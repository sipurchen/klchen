#!/usr/bin/env python3
"""Tier 2 driver: per-token expert sequences for the routing corpus, locality statistic, per-type affinity sets and the
base-vs-LoRA comparison (hypothesis H2). Runs under the machine profile (AFFINITY_PROFILE / --profile), i.e. under the same
restricted conditions as the Tier 1 pipeline (CPU cores, nice, -ngl, -ub, VRAM budget from the llama.cpp buffer log).

  python scripts/tier2_locality.py run --tag base --sessions coder_pilot,planner_pilot [--max-tokens N]
  python scripts/tier2_locality.py run --tag lora --sessions all --lora logs/lora/external.gguf --lora-scale 1.0
  python scripts/tier2_locality.py analyze --tag base
  python scripts/tier2_locality.py compare --base base --other lora

Outputs go to <root>/logs/tier2/<tag>/ : <session>.npz (uint8 [tokens, layers, k]), run_meta.json, summary.json, counts.npz.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from affinity_pipeline import tier2_stats as st  # noqa: E402
from affinity_pipeline.config import REPO_ROOT  # noqa: E402
from affinity_pipeline.llama_logs import BudgetVerdict, check_budget, parse_buffer_lines  # noqa: E402
from build_imatrix_jobs import get_profile  # noqa: E402

ALPHAS = (0.8, 0.9)
WINDOWS = (8, 32, 128)
SESSIONS_DIR = Path("data") / "routing_corpus_v2" / "sessions"


def session_names(root: Path, arg: str) -> list:
    if arg != "all":
        return [s for s in arg.split(",") if s]
    return sorted(p.name[: -len(".spans.json")] for p in (root / SESSIONS_DIR).glob("*.spans.json"))


def load_session(root: Path, name: str):
    body = (root / SESSIONS_DIR / f"{name}.body.txt").read_bytes()
    if b"\r" in body:
        raise SystemExit(f"{name}: body has CR characters; the corpus must be LF (autocrlf changes the token count)")
    spans = json.loads((root / SESSIONS_DIR / f"{name}.spans.json").read_text(encoding="utf-8"))
    return body.decode("utf-8"), spans


def default_layout(root: Path) -> Path:
    """Windows uses the b8679 layout, Linux the d327bf1 one (see config/tier2_layouts/)."""
    f = root / "config" / "tier2_layouts" / ("b8679.json" if sys.platform.startswith("win") else "d327bf1.json")
    if not f.exists():
        raise SystemExit(f"no layout file {f}; pass --layout")
    return f


def cmd_run(a) -> int:
    from affinity_pipeline.tier2_ctypes import Layout, Tier2Runner
    root = Path(a.root)
    profile = get_profile(a.profile)
    out = root / "logs" / "tier2" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    # llama.cpp logs the device buffer sizes to stderr: capture them so the VRAM budget can be checked like Tier 1 does
    log_path = out / "llama_stderr.log"
    saved = os.dup(2)
    fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 2)
    t0 = time.time()
    try:
        runner = Tier2Runner(profile.llama_bin_dir, profile.model_path, Layout.load(a.layout or default_layout(root)),
                             n_gpu_layers=a.ngl, n_ctx=a.n_ctx, n_ubatch=a.ubatch, n_threads=profile.cpu.threads,
                             lora_path=(a.lora.split(',') if a.lora else None), lora_scale=a.lora_scale)
    finally:
        os.dup2(saved, 2)
        os.close(fd)
    verdict = check_budget(parse_buffer_lines(log_path.read_text(encoding="utf-8", errors="replace")), profile.vram)
    report = parse_buffer_lines(log_path.read_text(encoding="utf-8", errors="replace"))
    print(f"model ready in {time.time() - t0:.1f}s; device buffers MiB={report.device_mb} verdict={verdict.value}", flush=True)
    if verdict == BudgetVerdict.ABORT:
        runner.close()
        print("VRAM budget exceeded: refusing to run")
        return 3
    meta = {"tag": a.tag, "profile": profile.name, "model": str(profile.model_path), "lora": a.lora, "lora_scale": a.lora_scale,
            "ngl": a.ngl, "ubatch": a.ubatch, "n_ctx": a.n_ctx, "threads": profile.cpu.threads,
            "device_buffers_mib": report.device_mb, "vram_verdict": verdict.value, "sessions": {}}
    names = session_names(root, a.sessions)
    for name in names:
        text, spans = load_session(root, name)
        ids = runner.tokenize(text)
        if len(ids) != spans["n_tokens"]:
            runner.close()
            print(f"{name}: tokenizer gives {len(ids)} tokens but spans.json has {spans['n_tokens']} (different tokenizer build?)")
            return 4
        if a.max_tokens:
            ids = ids[: a.max_tokens]
        t1 = time.time()
        last = [0.0]

        def progress(done, total, t1=t1, last=last):
            if time.time() - last[0] > 60:
                last[0] = time.time()
                print(f'  {name}: {done}/{total} tokens, {done / max(time.time() - t1, 1e-9):.2f} tok/s', flush=True)
        runner.progress = progress
        topk = runner.collect_per_token_experts(ids)
        np.savez_compressed(out / f"{name}.npz", topk=topk)
        meta["sessions"][name] = {"tokens": len(ids), "seconds": round(time.time() - t1, 1)}
        print(f"{name}: {len(ids)} tokens, topk {topk.shape}, {time.time() - t1:.1f}s", flush=True)
    runner.close()
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return 0


def _pool(root: Path, tag: str):
    """pooled locality accumulators and per-type counts over every session stored under a tag"""
    d = root / "logs" / "tier2" / tag
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    loc_pairs, loc_p, loc_inter, loc_shuf, counts, tokens = {}, {}, {}, {}, {}, 0
    win = {w: [0.0, 0.0, 0] for w in WINDOWS}   # hit sum, union-size sum, tokens
    win_layers = {}
    for name, info in meta["sessions"].items():
        topk = np.load(d / f"{name}.npz")["topk"]
        spans = json.loads((root / SESSIONS_DIR / f"{name}.spans.json").read_text(encoding="utf-8"))["spans"]
        spans = [s for s in spans if s["start"] < topk.shape[0]]
        for s in spans:
            s["end"] = min(s["end"], topk.shape[0])
        tokens += topk.shape[0]
        for w in WINDOWS:
            h, u = st.window_hit(topk, w)
            nw = topk.shape[0] - w
            win[w][0] += float(h.mean()) * nw
            win[w][1] += float(u.mean()) * nw
            win[w][2] += nw
        for g, v in st.locality(topk, spans).items():
            n = v["pairs"]
            loc_pairs[g] = loc_pairs.get(g, 0) + n
            loc_p[g] = loc_p.get(g, 0) + np.array(v["p_overlap_per_layer"]) * n
            loc_inter[g] = loc_inter.get(g, 0) + np.array(v["mean_intersection_per_layer"]) * n
            loc_shuf[g] = loc_shuf.get(g, 0) + np.array(v["p_overlap_shuffled_per_layer"]) * n
        for t, c in st.type_counts(topk, spans).items():
            counts[t] = counts.get(t, 0) + c
    return meta, tokens, loc_pairs, loc_p, loc_inter, loc_shuf, counts, win


def cmd_analyze(a) -> int:
    root = Path(a.root)
    meta, tokens, pairs, p_sum, inter_sum, shuf_sum, counts, win = _pool(root, a.tag)
    summary = {"tag": a.tag, "tokens": tokens, "sessions": list(meta["sessions"]), "lora": meta["lora"],
               "random_overlap_baseline": st.random_overlap_probability(), "locality": {}, "set_sizes": {}}
    for g in pairs:
        per_layer = (p_sum[g] / pairs[g])
        summary["locality"][g] = {"pairs": int(pairs[g]), "p_overlap_mean": float(per_layer.mean()),
                                  "p_overlap_median_layer": float(np.median(per_layer)),
                                  "p_overlap_min_layer": float(per_layer.min()), "p_overlap_max_layer": float(per_layer.max()),
                                  "p_overlap_shuffled_mean": float((shuf_sum[g] / pairs[g]).mean()),
                                  "mean_intersection": float((inter_sum[g] / pairs[g]).mean()),
                                  "layers_ge_0.85": int((per_layer >= 0.85).sum())}
    for al in ALPHAS:
        summary["set_sizes"][str(al)] = {t: {"median": float(np.median(st.set_sizes(c, al))),
                                             "per_layer": st.set_sizes(c, al)} for t, c in counts.items()}
    summary["type_conditioning"] = {str(al): st.type_conditioning(counts, al) for al in ALPHAS}
    summary["window_selection"] = {str(w): {"hit_rate": v[0] / v[2], "mean_union_size": v[1] / v[2]} for w, v in win.items()}
    np.savez_compressed(root / "logs" / "tier2" / a.tag / "counts.npz", **{t: c for t, c in counts.items()})
    (root / "logs" / "tier2" / a.tag / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"tag={a.tag} tokens={tokens} random-overlap baseline={summary['random_overlap_baseline']:.3f}")
    for g, v in summary["locality"].items():
        print(f"  locality[{g:10s}] pairs={v['pairs']:6d} P(overlap) mean={v['p_overlap_mean']:.3f} median-layer={v['p_overlap_median_layer']:.3f} "
              f"layers>=0.85: {v['layers_ge_0.85']}/30 shuffled-control={v['p_overlap_shuffled_mean']:.3f} mean|E_i&E_i+1|={v['mean_intersection']:.2f}")
    for al, d in summary["set_sizes"].items():
        print(f"  alpha={al}: " + ", ".join(f"{t}={v['median']:.1f}" for t, v in d.items()))
    for al, d in summary["type_conditioning"].items():
        print(f"  type conditioning alpha={al}: global set {d['global_median_size']:.1f}; per-type reduction "
              + ", ".join(f"{t}={100 * r:.0f}%" for t, r in d["per_type_reduction"].items()) + f"; mean {100 * d['mean_reduction']:.0f}%")
    for w, d in summary["window_selection"].items():
        print(f"  window W={w}: hit rate {d['hit_rate']:.3f} with mean union size {d['mean_union_size']:.1f} experts per layer")
    return 0


def _js(p, q):
    m = 0.5 * (p + q)
    def kl(x, y):
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / y[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cmd_compare(a) -> int:
    root = Path(a.root)
    cb = dict(np.load(root / "logs" / "tier2" / a.base / "counts.npz"))
    co = dict(np.load(root / "logs" / "tier2" / a.other / "counts.npz"))
    res = {"base": a.base, "other": a.other, "by_alpha": {}}
    for al in ALPHAS:
        res["by_alpha"][str(al)] = st.compare_runs(cb, co, al)
    # type separation proxy: per-layer Jensen-Shannon divergence between the type distributions (simplified V5)
    sep = {}
    types = sorted(set(cb) & set(co))
    for i, t1 in enumerate(types):
        for t2 in types[i + 1:]:
            def med(c):
                return float(np.median([_js(c[t1][l] / c[t1][l].sum(), c[t2][l] / c[t2][l].sum()) for l in range(c[t1].shape[0])]))
            sep[f"{t1}_vs_{t2}"] = {"jsd_base": med(cb), "jsd_other": med(co)}
    res["type_separation_jsd"] = sep
    d90 = res["by_alpha"]["0.9"]
    content = [t for t in st.CONTENT_TYPES if t in d90]
    mean_change = float(np.mean([d90[t]["relative_change"] for t in content]))
    res["h2"] = {"criterion": "mean over types of the relative change of the median per-layer alpha=0.9 set size <= -30% "
                              "and reasoning_vs_code / code_vs_factual JSD not below 0.9x base",
                 "mean_relative_change_alpha90": mean_change,
                 "per_type_relative_change": {t: d90[t]["relative_change"] for t in content},
                 "size_criterion_met": bool(mean_change <= -0.30),
                 "separation_held": bool(all(sep[k]["jsd_other"] >= 0.9 * sep[k]["jsd_base"]
                                             for k in ("code_vs_reasoning", "code_vs_factual") if k in sep)),
                 "mean_median_layer_jaccard": float(np.mean([d90[t]["median_layer_jaccard"] for t in content]))}
    out = root / "logs" / "tier2" / f"compare_{a.base}_vs_{a.other}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for al, d in res["by_alpha"].items():
        print(f"alpha={al}")
        for t, v in d.items():
            print(f"  {t:10s} median set size {v['median_size_base']:.1f} -> {v['median_size_other']:.1f} "
                  f"({100 * v['relative_change']:+.1f}%), median layer Jaccard {v['median_layer_jaccard']:.3f}")
    for k, v in sep.items():
        print(f"  JSD {k}: {v['jsd_base']:.4f} -> {v['jsd_other']:.4f}")
    print("H2:", json.dumps(res["h2"]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--sessions", default="all")
    r.add_argument("--profile")
    r.add_argument("--layout")
    r.add_argument("--lora")
    r.add_argument("--lora-scale", type=float, default=1.0, dest="lora_scale")
    r.add_argument("--ngl", type=int, default=3)
    r.add_argument("--ubatch", type=int, default=64)
    r.add_argument("--n-ctx", type=int, default=8192, dest="n_ctx")
    r.add_argument("--max-tokens", type=int, default=0, dest="max_tokens")
    an = sub.add_parser("analyze")
    an.add_argument("--tag", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True)
    c.add_argument("--other", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "analyze": cmd_analyze, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
