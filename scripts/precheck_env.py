"""
Pre-inference environment check.
Standard: Chrome+Edge closed, LINE open.
Warns if VRAM is below threshold; recommends which ngl to use.
"""

import subprocess
import sys

STANDARD_MB = 850
WARN_MB = 700
MUST_CLOSE = {"chrome.exe", "msedge.exe"}
KEEP_OPEN = {"LINE.exe"}  # do not suggest closing


def get_free_vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip().split(",")
        return int(out[0].strip()), int(out[1].strip())
    except Exception:
        return None, None


def get_running_targets():
    try:
        out = subprocess.check_output(
            ["tasklist", "/fo", "csv", "/nh"], text=True, encoding="utf-8", errors="ignore"
        )
        running = set()
        for line in out.splitlines():
            parts = line.strip('"').split('","')
            if parts:
                name = parts[0].lower()
                for t in MUST_CLOSE:
                    if t.lower() in name:
                        running.add(t)
        return running
    except Exception:
        return set()


def main():
    free, total = get_free_vram()
    if free is None:
        print("[ERROR] nvidia-smi not available")
        sys.exit(1)

    running_bad = get_running_targets()

    print("=" * 52)
    print("  GT 1030 Pre-Inference Environment Check")
    print("=" * 52)
    print(f"  VRAM total : {total} MB")
    print(f"  VRAM free  : {free} MB  (standard target: {STANDARD_MB} MB)")
    print()

    if running_bad:
        print(f"  [WARN] Still running: {', '.join(running_bad)}")
        print(f"         Close these to free ~{len(running_bad) * 230}MB VRAM")
    else:
        print("  [OK] Chrome and Edge not detected")

    print()
    status = "OK" if free >= STANDARD_MB else ("WARN" if free >= WARN_MB else "LOW")
    print(f"  VRAM status: [{status}]  free={free}MB / target={STANDARD_MB}MB")

    # recommend ngl for common models
    models = {
        "Gemma4 E4B Q4 (MoE)":   (46, 42.0,  96.0, 6.0),
        "Phi-4-mini Q4 (3.8B)":  (32, 55.0,  64.0, 6.0),
        "Mixtral-8x7B Q2_K":     (32, 85.0, 256.0, 8.0),
        "DeepSeek-R1-7B Q4":     (28, 155.0, 128.0, 8.0),
    }
    budget = max(0, free - 100)  # 100MB Vulkan overhead
    print()
    print(f"  Recommended ngl (budget={budget}MB after Vulkan overhead):")
    for name, (layers, layer_mb, emb_mb, kv_mb) in models.items():
        kv_total = kv_mb * layers
        avail = budget - emb_mb - kv_total
        ngl = max(0, min(layers, int(avail / layer_mb)))
        note = "" if ngl > 0 else " (CPU-only, still runnable)"
        print(f"    {name:<32} ngl={ngl}{note}")

    print()
    if status == "LOW":
        print("  [ACTION] Close Chrome and Edge before starting inference.")
        sys.exit(2)
    elif status == "WARN":
        print("  [NOTICE] Below standard. Consider closing more apps.")
    else:
        print("  Environment ready.")


if __name__ == "__main__":
    main()
