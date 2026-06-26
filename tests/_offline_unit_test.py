import sys, io, pathlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from monitor.role_tag_parser import RoleTagParser
from segmentor.chunk_classifier import classify_chunk
import numpy as np
from kv.kv_serializer import save_kv_chunk, load_kv_chunk

print("=== Offline Unit Tests ===")
print()

# Chunk detection
cot = "<think>\nStep by step.\n</think>\n```python\nprint(42)\n```"
parser = RoleTagParser()
bounds = parser.feed_text(cot)
print(f"Boundaries found: {len(bounds)}")
for b in bounds:
    c = classify_chunk(b.token_index, b.tag, [1.5, 1.2])
    print(f"  tag={b.tag!r:15} type={c.chunk_type:12} route={c.route}")

ok1 = len(bounds) >= 3
print(f"Chunk detection: {'PASS' if ok1 else 'FAIL'}")
print()

# KV round-trip
kv = np.random.rand(2, 8, 32).astype(np.float16)
path = save_kv_chunk("ci_test", 0, "code", kv)
loaded = load_kv_chunk("ci_test", 0, "code")
ok2 = loaded is not None and loaded.shape == kv.shape
print(f"KV round-trip: {'PASS' if ok2 else 'FAIL'}  shape={kv.shape}")
path.unlink(missing_ok=True)
try:
    path.parent.rmdir()
except Exception:
    pass

print()
print(f"Results: {sum([ok1, ok2])}/2 passed")
sys.exit(0 if ok1 and ok2 else 1)
