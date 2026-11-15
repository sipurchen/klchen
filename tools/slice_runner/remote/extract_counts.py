"""Run on gx10 (via gx10_run.py run counts): pack the per-layer expert counts of every logs/imatrix_v2/*.gguf into one small
counts.npz so the acceptor does not have to download ~1.4 GB of imatrix files."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import aggregate_expert_affinity as agg  # noqa: E402

d = ROOT / "logs" / "imatrix_v2"
arrays = {f.stem: agg.read_counts(f) for f in sorted(d.glob("*.gguf"))}
np.savez(d / "counts.npz", **arrays)
print(f"packed {len(arrays)} jobs -> {d / 'counts.npz'}")
