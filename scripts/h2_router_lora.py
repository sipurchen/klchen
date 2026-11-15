#!/usr/bin/env python3
"""H2 adapter: a LoRA on the MoE router (`router.proj`) trained so that tokens of a content type route into that type's
target expert set, plus the held-out evaluation (base vs student routing in the same engine).

Runs on gx10 with a python that has torch/transformers/peft and the frozen-expert offload engine (package `lora_offload`, imported
from $OFFLOAD_ENGINE_DIR; no bytecode is written there). Restricted preset: GPU 1.0 GiB, RAM pool 2.0 GiB.

  train: python scripts/h2_router_lora.py train --train s1,s2 --targets logs/h2/targets.json --out logs/h2/adapter_x
  eval : python scripts/h2_router_lora.py eval --adapter logs/h2/adapter_x --sessions coder_03,planner_03,writer_02 --out logs/h2/eval_x.json

Loss per layer: lam * mean(-log(router mass inside the target set of the token's type)) + mu * mean(KL(p_base || p_student)).
Training is teacher-forced (the body keeps the BASE routing); evaluation lets the student routing drive the body.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
CODE = os.path.expanduser(os.environ.get("OFFLOAD_ENGINE_DIR", "~/offload_engine"))   # directory that contains the `lora_offload` package
sys.path.insert(0, CODE)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from affinity_pipeline import tier2_stats as st  # noqa: E402
from lora_offload.engine import LayerWeightStreamer, build_gemma4_text_model, chunked_causal_lm_loss  # noqa: E402
from lora_offload.store import ThreeTierExpertStore  # noqa: E402

GIB = 1024 ** 3
MODEL_DIR = os.path.expanduser(os.environ.get("BASE_HF_DIR", "~/models/gemma-4-26B-A4B-it"))
SESSIONS_DIR = Path("data") / "routing_corpus_v2" / "sessions"
TYPES = ["reasoning", "code", "factual", "creative", "control", "user_prompt"]
TYPE_ID = {t: i for i, t in enumerate(TYPES)}
NONE_ID = len(TYPES)
N_LAYERS, N_EXPERTS, K = 30, 128, 8


class State:
    train_side = False        # compute the router-local training loss inside the router hook
    student_body = False      # let the student routing drive the body
    lam, mu = 1.0, 0.2
    nu = 0.0                 # weight of the keep-the-base-top-1-expert term
    hinge = 0.0              # >0: only penalise router mass inside the target set below this level (no pressure beyond it)
    type_ids = None           # LongTensor [tokens] of the current window
    losses = None
    mass_log = None
    record = None             # {layer: [uint8 [tokens, K]]} when collecting routing


def build_engine(gpu_gib: float = 1.0, ram_gib: float = 2.0):
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_gib * GIB / total), 0)
    torch.cuda.reset_peak_memory_stats()
    store = ThreeTierExpertStore(MODEL_DIR, device, gpu_cache_bytes=0, ram_pool_bytes=int(ram_gib * GIB))
    model, tc, info = build_gemma4_text_model(MODEL_DIR, device, store, "sdpa")
    for layer in model.model.layers:
        LayerWeightStreamer(layer, device, resident=False)
    for p in model.model.norm.parameters():
        p.data = p.data.to(device)
    embed = model.model.embed_tokens
    embed.register_forward_hook(lambda m, a, out: out.to(device))
    model.eval()
    return model, embed.weight.data, store, device, tc


def make_lora(rank: int, device):
    As, Bs = [], []
    for _ in range(N_LAYERS):
        a = torch.empty(rank, 2816, device=device)
        torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        As.append(a.requires_grad_(True))
        Bs.append(torch.zeros(N_EXPERTS, rank, device=device, requires_grad=True))
    return As, Bs


def target_masks(targets: dict, device) -> list:
    """per layer float mask [len(TYPES)+1, E]: row of a content type = its target set; other rows all zero"""
    masks = []
    for l in range(N_LAYERS):
        m = torch.zeros(len(TYPES) + 1, N_EXPERTS, device=device)
        for t, sets in targets["targets"].items():
            m[TYPE_ID[t], sets[l]] = 1.0
        masks.append(m)
    return masks


def patch_routers(model, As, Bs, scale: float, masks, state: State) -> None:
    for l, layer in enumerate(model.model.layers):
        router = layer.router
        A, B, tmask = As[l], Bs[l], masks[l]
        k = router.config.top_k_experts

        def fwd(hidden_states, router=router, A=A, B=B, tmask=tmask, l=l, k=k):
            h = router.norm(hidden_states)
            h = h * router.scale * router.scalar_root_size
            base_scores = router.proj(h).float()
            if state.train_side:
                with torch.enable_grad():
                    delta = ((h.detach().float() @ A.T) @ B.T) * scale
                    logp_s = torch.log_softmax(base_scores.detach() + delta, dim=-1)
                    logp_b = torch.log_softmax(base_scores.detach(), dim=-1)
                    kl = (logp_b.exp() * (logp_b - logp_s)).sum(-1)
                    loss = state.mu * kl.mean()
                    if state.nu > 0:
                        loss = loss + state.nu * (-logp_s.gather(-1, base_scores.argmax(-1, keepdim=True)).squeeze(-1)).mean()
                    m = tmask[state.type_ids]
                    has = m.sum(-1) > 0
                    mass_val = None
                    if bool(has.any()):
                        mass = (logp_s.exp() * m).sum(-1).clamp_min(1e-6)
                        conc = -torch.log(mass)
                        if state.hinge > 0:
                            conc = torch.relu(conc - (-math.log(state.hinge)))
                        loss = loss + state.lam * conc[has].mean()
                        mass_val = float(mass[has].mean())
                    state.losses.append(loss)
                    state.mass_log.append((l, mass_val, float(kl.mean())))
            scores = base_scores
            if state.student_body:
                with torch.no_grad():
                    scores = base_scores + ((h.float() @ A.T) @ B.T) * scale
            probs = torch.softmax(scores, dim=-1)
            w, idx = torch.topk(probs, k=k, dim=-1)
            w = w / w.sum(dim=-1, keepdim=True)
            w = w * router.per_expert_scale[idx]
            if state.record is not None:
                state.record[l].append(idx.detach().to("cpu", torch.uint8))
            return probs, w, idx

        router.forward = fwd


def load_session(tok, name: str):
    body = (SESSIONS_DIR / f"{name}.body.txt").read_bytes().decode("utf-8")
    spans = json.loads((SESSIONS_DIR / f"{name}.spans.json").read_text(encoding="utf-8"))
    ids = tok(body, add_special_tokens=False)["input_ids"]
    if len(ids) != spans["n_tokens"]:
        raise SystemExit(f"{name}: HF tokenizer gives {len(ids)} tokens, spans.json has {spans['n_tokens']}")
    types = st.type_array(spans["spans"], len(ids))
    tid = np.array([TYPE_ID.get(t, NONE_ID) for t in types], dtype=np.int64)
    return ids, tid, spans["spans"]


def windows(ids, tid, size: int):
    for s in range(0, len(ids), size):
        e = min(s + size, len(ids))
        if e - s >= 32:
            yield s, e


def forward(model, ids_slice, device):
    x = torch.tensor([ids_slice], dtype=torch.long)
    return model.model(input_ids=x, attention_mask=torch.ones_like(x).to(device), use_cache=False).last_hidden_state, x


def cmd_train(a) -> int:
    from transformers import AutoTokenizer
    random.seed(3407)
    torch.manual_seed(3407)
    targets = json.loads(Path(a.targets).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    t0 = time.time()
    model, embed_w, store, device, tc = build_engine()
    print(f"> engine built in {time.time() - t0:.1f}s", flush=True)
    As, Bs = make_lora(a.rank, device)
    state = State()
    state.lam, state.mu, state.hinge, state.nu = a.lam, a.mu, a.hinge, a.nu
    patch_routers(model, As, Bs, a.alpha / a.rank, target_masks(targets, device), state)
    params = As + Bs
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    items = []
    for name in a.train.split(","):
        ids, tid, _ = load_session(tok, name)
        for s, e in windows(ids, tid, a.window):
            items.append((name, ids[s:e], tid[s:e]))
    if a.max_windows:
        items = items[: a.max_windows]
    total_steps = math.ceil(len(items) / a.accum) * a.epochs
    print(f"> {len(items)} windows of <= {a.window} tokens, {total_steps} optimizer steps", flush=True)
    log, step = [], 0
    for ep in range(a.epochs):
        random.shuffle(items)
        for g in range(0, len(items), a.accum):
            group = items[g:g + a.accum]
            t1 = time.time()
            tot, mass_acc, kl_acc, n = 0.0, [], [], 0
            for name, ids, tid in group:
                state.train_side, state.losses, state.mass_log = True, [], []
                state.type_ids = torch.tensor(tid, device=device)
                with torch.no_grad():
                    forward(model, ids, device)
                loss = sum(state.losses) / N_LAYERS / len(group)
                loss.backward()
                tot += float(loss) * len(group)
                mass_acc += [m for _, m, _ in state.mass_log if m is not None]
                kl_acc += [k for _, _, k in state.mass_log]
                n += 1
            lr = a.lr * max(0.05, 1 - step / max(total_steps, 1))
            for pg in opt.param_groups:
                pg["lr"] = lr
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": ep + 1, "loss": tot / n, "mean_mass_in_target": float(np.mean(mass_acc)) if mass_acc else None,
                   "mean_kl": float(np.mean(kl_acc)), "grad_norm": gn, "seconds": round(time.time() - t1, 1),
                   "cuda_peak_gib": round(torch.cuda.max_memory_reserved() / GIB, 3)}
            log.append(rec)
            print(f"> step {step}/{total_steps} loss={rec['loss']:.4f} mass_in_target={rec['mean_mass_in_target']} "
                  f"kl={rec['mean_kl']:.4f} gn={gn:.3f} {rec['seconds']}s peak={rec['cuda_peak_gib']}GiB", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    tensors = {}
    for l in range(N_LAYERS):
        tensors[f"base_model.model.model.layers.{l}.router.proj.lora_A.weight"] = As[l].detach().cpu().contiguous()
        tensors[f"base_model.model.model.layers.{l}.router.proj.lora_B.weight"] = Bs[l].detach().cpu().contiguous()
    save_file(tensors, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": a.rank, "lora_alpha": a.alpha, "lora_dropout": 0.0,
        "target_modules": ["router.proj"], "bias": "none", "base_model_name_or_path": None, "fan_in_fan_out": False,
        "inference_mode": True, "use_rslora": False, "use_dora": False}, indent=1), encoding="utf-8")
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "steps": log, "train_seconds": round(time.time() - t0, 1)}, indent=1),
                                        encoding="utf-8")
    print("> adapter saved:", out, flush=True)
    store.release_all()
    return 0


def chunked_argmax(h, W, chunk=16384, cap=30.0):
    best_val = torch.full((h.shape[0],), float("-inf"), device=h.device)
    best_idx = torch.zeros((h.shape[0],), dtype=torch.long, device=h.device)
    for s in range(0, W.shape[0], chunk):
        wc = W[s:s + chunk].to(h.device)
        lg = cap * torch.tanh((h @ wc.T).float() / cap)
        v, i = lg.max(dim=1)
        m = v > best_val
        best_val, best_idx = torch.where(m, v, best_val), torch.where(m, i + s, best_idx)
    return best_idx


def cmd_eval(a) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model, embed_w, store, device, tc = build_engine()
    cfg = json.loads((Path(a.adapter) / "adapter_config.json").read_text(encoding="utf-8"))
    from safetensors.torch import load_file
    t = load_file(str(Path(a.adapter) / "adapter_model.safetensors"))
    As = [t[f"base_model.model.model.layers.{l}.router.proj.lora_A.weight"].to(device) for l in range(N_LAYERS)]
    Bs = [t[f"base_model.model.model.layers.{l}.router.proj.lora_B.weight"].to(device) for l in range(N_LAYERS)]
    state = State()
    patch_routers(model, As, Bs, cfg["lora_alpha"] / cfg["r"], [torch.zeros(len(TYPES) + 1, N_EXPERTS, device=device)] * N_LAYERS, state)
    res = {"adapter": a.adapter, "sessions": {}, "pooled": {}}
    pooled = {"base": {}, "student": {}}
    nll = {"base": [0.0, 0], "student": [0.0, 0]}
    agree = [0, 0]
    for name in a.sessions.split(","):
        ids, tid, spans = load_session(tok, name)
        topk_modes = {}
        for mode in ("base", "student"):
            state.student_body = mode == "student"
            state.record = {l: [] for l in range(N_LAYERS)}
            argmaxes, losses = [], []
            with torch.no_grad():
                for s, e in windows(ids, tid, a.window):
                    h, x = forward(model, ids[s:e], device)
                    mask = torch.ones_like(x).to(device)
                    loss = float(chunked_causal_lm_loss(h, x, mask, embed_w))
                    nt = e - s - 1
                    nll[mode][0] += loss * nt
                    nll[mode][1] += nt
                    argmaxes.append(chunked_argmax(h[0, :-1], embed_w).cpu())
            topk_modes[mode] = (np.stack([np.concatenate([r.numpy() for r in state.record[l]], axis=0) for l in range(N_LAYERS)], axis=1),
                                torch.cat(argmaxes))
            print(f"> {name} {mode}: done", flush=True)
        agree[0] += int((topk_modes["base"][1] == topk_modes["student"][1]).sum())
        agree[1] += len(topk_modes["base"][1])
        covered = topk_modes["base"][0].shape[0]
        sp = [dict(s_, end=min(s_["end"], covered)) for s_ in spans if s_["start"] < covered]
        # windows shorter than 32 tokens were skipped: the recorded array may be shorter than the session
        for mode in ("base", "student"):
            for typ, c in st.type_counts(topk_modes[mode][0], sp).items():
                pooled[mode][typ] = pooled[mode].get(typ, 0) + c
    res["nll_base"], res["nll_student"] = nll["base"][0] / nll["base"][1], nll["student"][0] / nll["student"][1]
    res["delta_nll"] = res["nll_student"] - res["nll_base"]
    res["top1_agreement"] = agree[0] / agree[1]
    for al in (0.8, 0.9):
        res["pooled"][str(al)] = st.compare_runs(pooled["base"], pooled["student"], al)
    d90 = res["pooled"]["0.9"]
    content = [x for x in st.CONTENT_TYPES if x in d90]
    res["mean_relative_change_alpha90"] = float(np.mean([d90[x]["relative_change"] for x in content]))
    res["h2_size_met"] = bool(res["mean_relative_change_alpha90"] <= -0.30)
    res["guard_nll_met"] = bool(res["delta_nll"] <= 0.15)
    res["guard_top1_met"] = bool(res["top1_agreement"] >= 0.85)
    res["h2_passed"] = bool(res["h2_size_met"] and res["guard_nll_met"] and res["guard_top1_met"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("nll_base", "nll_student", "delta_nll", "top1_agreement", "mean_relative_change_alpha90",
                                          "h2_size_met", "guard_nll_met", "guard_top1_met", "h2_passed")}), flush=True)
    for x in content:
        v = d90[x]
        print(f"  {x:10s} set size {v['median_size_base']:.1f} -> {v['median_size_other']:.1f} ({100 * v['relative_change']:+.1f}%) "
              f"jaccard {v['median_layer_jaccard']:.3f}", flush=True)
    store.release_all()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--targets", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--alpha", type=float, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--accum", type=int, default=2)
    t.add_argument("--lam", type=float, default=1.0)
    t.add_argument("--mu", type=float, default=0.2)
    t.add_argument("--hinge", type=float, default=0.0)
    t.add_argument("--nu", type=float, default=0.0)
    t.add_argument("--window", type=int, default=768)
    t.add_argument("--max-windows", type=int, default=0, dest="max_windows")
    e = sub.add_parser("eval")
    e.add_argument("--adapter", required=True)
    e.add_argument("--sessions", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--window", type=int, default=768)
    a = ap.parse_args()
    return cmd_train(a) if a.cmd == "train" else cmd_eval(a)


if __name__ == "__main__":
    sys.exit(main())
