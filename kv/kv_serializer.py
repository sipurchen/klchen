"""
P3.1 KV Serializer
Serialize chunk KV cache to .kvbin files for disk offload.
Format: [header: 64 bytes][tensor bytes]
Header: magic(4) + version(2) + session_id(16) + chunk_id(4) +
        layer_start(2) + layer_end(2) + dtype(2) + shape_len(2) + shape(N*4) + reserved
"""

import os
import struct
import numpy as np
from pathlib import Path

MAGIC = b"KVBN"
VERSION = 1
KV_BASE_DIR = Path("kv_store")


def _header(session_id: str, chunk_id: int, layer_start: int, layer_end: int,
            shape: tuple, dtype_code: int = 1) -> bytes:
    sid = session_id.encode()[:16].ljust(16, b"\x00")
    shape_bytes = struct.pack(f"<{len(shape)}I", *shape)
    hdr = struct.pack(
        "<4sH16sIHHHH",
        MAGIC, VERSION, sid, chunk_id,
        layer_start, layer_end,
        dtype_code, len(shape),
    )
    return hdr + shape_bytes


def save_kv_chunk(
    session_id: str,
    chunk_id: int,
    role: str,
    kv_tensor: np.ndarray,
    layer_start: int = 0,
    layer_end: int = 32,
):
    """Write chunk KV to {KV_BASE_DIR}/{session_id}/{chunk_id}_{role}.kvbin"""
    out_dir = KV_BASE_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{chunk_id:04d}_{role}.kvbin"

    dtype_code = 1 if kv_tensor.dtype == np.float16 else 2  # 1=f16 2=f32
    hdr = _header(session_id, chunk_id, layer_start, layer_end,
                  kv_tensor.shape, dtype_code)

    with open(path, "wb") as f:
        f.write(hdr)
        f.write(kv_tensor.tobytes())
    return path


def load_kv_chunk(session_id: str, chunk_id: int, role: str) -> np.ndarray | None:
    path = KV_BASE_DIR / session_id / f"{chunk_id:04d}_{role}.kvbin"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic}")
        f.read(2)  # version
        f.read(16)  # session_id
        f.read(4)   # chunk_id
        f.read(2); f.read(2)  # layer_start, layer_end
        dtype_code, shape_len = struct.unpack("<HH", f.read(4))
        shape = struct.unpack(f"<{shape_len}I", f.read(shape_len * 4))
        dtype = np.float16 if dtype_code == 1 else np.float32
        data = np.frombuffer(f.read(), dtype=dtype).reshape(shape)
    return data
