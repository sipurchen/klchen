#!/usr/bin/env python3
"""H2 compensation adapter: keep the trained router LoRA (the part that shrinks the expert sets) FROZEN and train an
attention + dense-MLP LoRA on top of it, with the STUDENT routing driving the body, so that the model recovers the quality the
concentrated routing cost (next-token top-1 agreement with the base, NLL).

Loss per window = w * CE(student logits, real next tokens) + (1 - w) * CE(student logits, the BASE model's argmax tokens), w = --lm-weight (default 0.5).
The second term is hard-label distillation toward exactly the guard metric (top-1 agreement); the base argmax comes from a
no-grad pass with all adapters off and the base routing. Backward goes through the frozen experts (offload engine re-fetches them).

  python scripts/h2_comp.py train --router logs/h2/adapter_c1 --out logs/h2/adapter_f1 [--lr 1e-4 --epochs 2]
  python scripts/h2_comp.py eval  --router logs/h2/adapter_c1 --peft logs/h2/adapter_f1 --sessions coder_03,planner_03,writer_02 --out logs/h2/eval_f1.json
Same train / held-out split, same guards and same restricted preset (GPU 1.0 GiB, RAM pool 2.0 GiB) as h2_router_lora.py.
"""
import argparse
import contextlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import h2_router_lora as h2r
from h2_router_lora import GIB, MODEL_DIR, N_EXPERTS, N_LAYERS, State, TYPES
from lora_offload.engine import LayerWeightStreamer, build_gemma4_text_model, chunked_causal_lm_loss
from lora_offload.store import ThreeTierExpertStore

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRAIN = "coder_pilot,coder_01,coder_02,planner_pilot,planner_01,planner_02,writer_01"
HELD_OUT = "coder_03,planner_03,writer_02"


def build(peft_dir, train_cfg, gpu_gib=1.0, ram_gib=2.0):
    """PEFT is attached BEFORE the layer streamers (they skip trainable params), as the offload engine requires."""
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_gib * GIB / total), 0)
    torch.cuda.reset_peak_memory_stats()
    store = ThreeTierExpertStore(MODEL_DIR, device, gpu_cache_bytes=0, ram_pool_bytes=int(ram_gib * GIB))
    model, tc, info = build_gemma4_text_model(MODEL_DIR, device, store, "sdpa")
    from peft import LoraConfig, PeftModel, get_peft_model
    if train_cfg:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        pm = get_peft_model(model, LoraConfig(r=train_cfg["rank"], lora_alpha=train_cfg["alpha"], lora_dropout=0.0, bias="none",
                                              task_type="CAUSAL_LM", target_modules=TARGET_MODULES))
    else:
        pm = PeftModel.from_pretrained(model, peft_dir, is_trainable=False)
    for _, p in pm.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(device)
    inner = pm.base_model.model
    for layer in inner.model.layers:
        LayerWeightStreamer(layer, device, resident=False)
    for p in inner.model.norm.parameters():
        p.data = p.data.to(device)
    embed = inner.model.embed_tokens
    embed.register_forward_hook(lambda m, a, out: out.to(device))
    return pm, inner, embed.weight.data, store, device


def load_router(router_dir, device):
    from safetensors.torch import load_file
    cfg = json.loads((Path(router_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    t = load_file(str(Path(router_dir) / "adapter_model.safetensors"))
    As = [t[f"base_model.model.model.layers.{l}.router.proj.lora_A.weight"].to(device) for l in range(N_LAYERS)]
    Bs = [t[f"base_model.model.model.layers.{l}.router.proj.lora_B.weight"].to(device) for l in range(N_LAYERS)]
    return As, Bs, cfg["lora_alpha"] / cfg["r"]


def fwd(inner, ids, device):
    x = torch.tensor([ids], dtype=torch.long)
    return inner.model(input_ids=x, attention_mask=torch.ones_like(x).to(device), use_cache=False).last_hidden_state, x


def cmd_train(a) -> int:
    from transformers import AutoTokenizer
    random.seed(3407)
    torch.manual_seed(3407)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    t0 = time.time()
    pm, inner, embed_w, store, device = build(None, {"rank": a.rank, "alpha": a.alpha})
    As, Bs, rscale = load_router(a.router, device)
    state = State()
    h2r.patch_routers(inner, As, Bs, rscale, [torch.zeros(len(TYPES) + 1, N_EXPERTS, device=device)] * N_LAYERS, state)
    trainable = [p for p in pm.parameters() if p.requires_grad]
    print(f"> built in {time.time() - t0:.1f}s; trainable LoRA params {sum(p.numel() for p in trainable):,}", flush=True)
    pm.train()
    items = []
    for name in TRAIN.split(","):
        ids, tid, _ = h2r.load_session(tok, name)
        for s, e in h2r.windows(ids, tid, a.window):
            items.append(ids[s:e])
    if a.max_windows:
        items = items[: a.max_windows]
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.01)
    total_steps = math.ceil(len(items) / a.accum) * a.epochs
    print(f"> {len(items)} windows, {total_steps} optimizer steps", flush=True)
    log, step = [], 0
    for ep in range(a.epochs):
        random.shuffle(items)
        for g in range(0, len(items), a.accum):
            group = items[g:g + a.accum]
            t1, tot, lm_tot, kd_tot = time.time(), 0.0, 0.0, 0.0
            for ids in group:
                with torch.no_grad(), pm.disable_adapter():                  # base: adapters off, base routing
                    state.student_body = False
                    hb, x = fwd(inner, ids, device)
                    yb = h2r.chunked_argmax(hb[0, :-1], embed_w).cpu()
                state.student_body = True                                    # student: all adapters, student routing
                hs, x = fwd(inner, ids, device)
                mask = torch.ones_like(x).to(device)
                lm = chunked_causal_lm_loss(hs, x, mask, embed_w)
                fake = torch.cat([x[:, :1], yb.view(1, -1)], dim=1)
                kd = chunked_causal_lm_loss(hs, fake, mask, embed_w)
                loss = (a.lm_weight * lm + (1.0 - a.lm_weight) * kd) / len(group)
                loss.backward()
                tot += float(loss) * len(group)
                lm_tot += float(lm)
                kd_tot += float(kd)
            lr = a.lr * max(0.05, 1 - step / max(total_steps, 1))
            for pg in opt.param_groups:
                pg["lr"] = lr
            gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": ep + 1, "loss": tot / len(group), "lm": lm_tot / len(group), "kd": kd_tot / len(group),
                   "grad_norm": gn, "seconds": round(time.time() - t1, 1), "cuda_peak_gib": round(torch.cuda.max_memory_reserved() / GIB, 3)}
            log.append(rec)
            print(f"> step {step}/{total_steps} loss={rec['loss']:.4f} lm={rec['lm']:.3f} kd={rec['kd']:.3f} gn={gn:.3f} "
                  f"{rec['seconds']}s peak={rec['cuda_peak_gib']}GiB", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(str(out))
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "steps": log, "train_seconds": round(time.time() - t0, 1)}, indent=1),
                                        encoding="utf-8")
    print("> adapter saved:", out, flush=True)
    store.release_all()
    return 0


def cmd_eval(a) -> int:
    from transformers import AutoTokenizer
    from h2_router_lora import st
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    pm, inner, embed_w, store, device = build(a.peft, None)
    pm.eval()
    As, Bs, rscale = load_router(a.router, device)
    state = State()
    h2r.patch_routers(inner, As, Bs, rscale, [torch.zeros(len(TYPES) + 1, N_EXPERTS, device=device)] * N_LAYERS, state)
    pooled = {"base": {}, "student": {}}
    nll = {"base": [0.0, 0], "student": [0.0, 0]}
    agree = [0, 0]
    for name in a.sessions.split(","):
        ids, tid, spans = h2r.load_session(tok, name)
        res = {}
        for mode in ("base", "student"):
            state.student_body = mode == "student"
            state.record = {l: [] for l in range(N_LAYERS)}
            ams = []
            ctx = contextlib.nullcontext() if mode == "student" else pm.disable_adapter()
            with torch.no_grad(), ctx:
                for s, e in h2r.windows(ids, tid, a.window):
                    h, x = fwd(inner, ids[s:e], device)
                    loss = float(chunked_causal_lm_loss(h, x, torch.ones_like(x).to(device), embed_w))
                    nll[mode][0] += loss * (e - s - 1)
                    nll[mode][1] += e - s - 1
                    ams.append(h2r.chunked_argmax(h[0, :-1], embed_w).cpu())
            res[mode] = (np.stack([np.concatenate([r.numpy() for r in state.record[l]], axis=0) for l in range(N_LAYERS)], axis=1), torch.cat(ams))
            print(f"> {name} {mode}: done", flush=True)
        agree[0] += int((res["base"][1] == res["student"][1]).sum())
        agree[1] += len(res["base"][1])
        covered = res["base"][0].shape[0]
        sp = [dict(s_, end=min(s_["end"], covered)) for s_ in spans if s_["start"] < covered]
        for mode in ("base", "student"):
            for typ, c in st.type_counts(res[mode][0], sp).items():
                pooled[mode][typ] = pooled[mode].get(typ, 0) + c
    out = {"router": a.router, "peft": a.peft, "pooled": {}}
    out["nll_base"], out["nll_student"] = nll["base"][0] / nll["base"][1], nll["student"][0] / nll["student"][1]
    out["delta_nll"] = out["nll_student"] - out["nll_base"]
    out["top1_agreement"] = agree[0] / agree[1]
    for al in (0.8, 0.9):
        out["pooled"][str(al)] = st.compare_runs(pooled["base"], pooled["student"], al)
    d90 = out["pooled"]["0.9"]
    content = [x for x in st.CONTENT_TYPES if x in d90]
    out["mean_relative_change_alpha90"] = float(np.mean([d90[x]["relative_change"] for x in content]))
    out["h2_size_met"] = bool(out["mean_relative_change_alpha90"] <= -0.30)
    out["guard_nll_met"] = bool(out["delta_nll"] <= 0.15)
    out["guard_top1_met"] = bool(out["top1_agreement"] >= 0.85)
    out["h2_passed"] = bool(out["h2_size_met"] and out["guard_nll_met"] and out["guard_top1_met"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("nll_base", "nll_student", "delta_nll", "top1_agreement", "mean_relative_change_alpha90",
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
    t.add_argument("--router", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--lm-weight", type=float, default=0.5, dest="lm_weight")
    t.add_argument("--alpha", type=float, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--accum", type=int, default=2)
    t.add_argument("--window", type=int, default=768)
    t.add_argument("--max-windows", type=int, default=0, dest="max_windows")
    e = sub.add_parser("eval")
    e.add_argument("--router", required=True)
    e.add_argument("--peft", required=True)
    e.add_argument("--sessions", default=HELD_OUT)
    e.add_argument("--out", required=True)
    e.add_argument("--window", type=int, default=768)
    a = ap.parse_args()
    return cmd_train(a) if a.cmd == "train" else cmd_eval(a)


if __name__ == "__main__":
    sys.exit(main())
