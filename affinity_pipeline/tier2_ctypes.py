"""Tier 2 harness: per-token MoE expert ids from the shipped llama.cpp shared libraries, through Python ctypes.

Tier 1 (`llama-imatrix`) only gives cumulative per-layer counts. The locality statistic Pr[E_i ∩ E_{i+1} ≠ ∅] and any per-token
analysis need the expert ids in token order. llama.cpp exposes a per-node eval callback (`cb_eval`); the router's top-k indices
of layer L are the tensor named `ffn_moe_topk-L`. This module loads `libllama` / `libggml*`, installs that callback, prefills a
token sequence and returns the ids as uint8 [n_tokens, n_layers, n_expert_used].

No struct is transcribed by hand: sizes and field offsets come from a layout JSON measured with the build's own headers
(`tools/slice_runner/remote/layout_probe.py`, stored in `config/tier2_layouts/<commit>.json`). Everything the harness
configures mirrors the CLI flags of the restricted profiles (`-ngl`, `--override-tensor "\\.ffn_.*_exps\\.=CPU"`, `-ub`,
`-t`), so observing the model does not change how it runs.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np

TOPK_PREFIX = b"ffn_moe_topk-"
EXPERT_PATTERN = rb"\.ffn_.*_exps\."


class Layout:
    """Byte layout of the llama.cpp / ggml structs the harness touches (see layout_probe.py)."""

    def __init__(self, d: dict):
        self.d = d

    @classmethod
    def load(cls, path) -> "Layout":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def size(self, struct: str) -> int:
        return int(self.d[struct]["size"])

    def off(self, struct: str, field: str) -> int:
        return int(self.d[struct][field])

    def const(self, name: str) -> int:
        return int(self.d["constants"][name])


def blob_type(n: int):
    """An opaque by-value struct of n bytes (what llama_*_default_params() returns and llama_* functions take)."""
    return type(f"Blob{n}", (ctypes.Structure,), {"_fields_": [("raw", ctypes.c_ubyte * n)]})


def poke(blob, off: int, ctype, value) -> None:
    ctype.from_address(ctypes.addressof(blob) + off).value = value


def peek(blob, off: int, ctype):
    return ctype.from_address(ctypes.addressof(blob) + off).value


def parse_topk(buf: bytes, ne0: int, ne1: int, nb1: int) -> np.ndarray:
    """The router top-k tensor is a strided view (row stride nb1 >= ne0*4): rebuild int32 [ne1 tokens, ne0 experts]."""
    arr = np.frombuffer(buf, dtype=np.int32)
    if nb1 % 4:
        raise ValueError(f"unexpected row stride {nb1}")
    return np.lib.stride_tricks.as_strided(arr, shape=(ne1, ne0), strides=(nb1, 4)).copy()


def _lib_names(platform: str) -> dict:
    if platform.startswith("win"):
        return {"ggml_base": "ggml-base.dll", "ggml": "ggml.dll", "llama": "llama.dll"}
    return {"ggml_base": "libggml-base.so", "ggml": "libggml.so", "llama": "libllama.so"}


class Tier2Runner:
    """Load a GGUF model (optionally with a GGUF LoRA) and collect per-token top-k expert ids while prefilling."""

    def __init__(self, lib_dir, model_path, layout: Layout, *, n_gpu_layers: int = 3, n_ctx: int = 8192,
                 n_ubatch: int = 64, n_threads: int = 5, lora_path=None, lora_scale: float = 1.0,
                 experts_on_cpu: bool = True):
        self.layout = layout
        self.progress = None            # optional callable(done_tokens, total_tokens)
        self.lib_dir = Path(lib_dir)
        self.n_ubatch = n_ubatch
        self._rows: dict[int, list] = {}
        self._err: str | None = None
        self._cb_calls = 0
        names = _lib_names(sys.platform)
        mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        if hasattr(os, "add_dll_directory"):      # Windows: let llama.dll find ggml*.dll and the backend DLLs
            os.add_dll_directory(str(self.lib_dir))
        self.ggml_base = ctypes.CDLL(str(self.lib_dir / names["ggml_base"]), mode=mode)
        self.ggml = ctypes.CDLL(str(self.lib_dir / names["ggml"]), mode=mode)
        self.llama = ctypes.CDLL(str(self.lib_dir / names["llama"]), mode=mode)
        self._bind()
        self._sym("ggml_backend_load_all_from_path")(str(self.lib_dir).encode())
        self.llama.llama_backend_init()
        L = layout
        ModelBlob, CtxBlob = blob_type(L.size("llama_model_params")), blob_type(L.size("llama_context_params"))
        self._BatchBlob = blob_type(L.size("llama_batch"))
        self.llama.llama_model_default_params.restype = ModelBlob
        self.llama.llama_context_default_params.restype = CtxBlob
        self.llama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, ModelBlob]
        self.llama.llama_init_from_model.argtypes = [ctypes.c_void_p, CtxBlob]
        self.llama.llama_batch_init.restype = self._BatchBlob
        self.llama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        self.llama.llama_batch_free.argtypes = [self._BatchBlob]
        self.llama.llama_decode.argtypes = [ctypes.c_void_p, self._BatchBlob]

        mp = self.llama.llama_model_default_params()
        poke(mp, L.off("llama_model_params", "n_gpu_layers"), ctypes.c_int32, n_gpu_layers)
        # tensor_buft_overrides: [{pattern, buft}, {NULL, NULL}]  (== --override-tensor "<pattern>=CPU")
        self._pattern = ctypes.create_string_buffer(EXPERT_PATTERN)
        OvBlob = blob_type(L.size("llama_model_tensor_buft_override"))
        self._overrides = (OvBlob * 2)()
        if experts_on_cpu:
            cpu_dev = self._sym("ggml_backend_dev_by_type")(L.const("GGML_BACKEND_DEVICE_TYPE_CPU"))
            buft = self._sym("ggml_backend_dev_buffer_type")(cpu_dev)
            poke(self._overrides[0], L.off("llama_model_tensor_buft_override", "pattern"), ctypes.c_void_p,
                 ctypes.addressof(self._pattern))
            poke(self._overrides[0], L.off("llama_model_tensor_buft_override", "buft"), ctypes.c_void_p, buft)
            poke(mp, L.off("llama_model_params", "tensor_buft_overrides"), ctypes.c_void_p,
                 ctypes.addressof(self._overrides))
        self.model = self.llama.llama_model_load_from_file(str(model_path).encode(), mp)
        if not self.model:
            raise RuntimeError(f"cannot load model {model_path}")
        self.vocab = self.llama.llama_model_get_vocab(self.model)

        self._CB = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p)
        self._cb = self._CB(self._on_tensor)
        cp = self.llama.llama_context_default_params()
        for field, ct, val in (("n_ctx", ctypes.c_uint32, n_ctx), ("n_batch", ctypes.c_uint32, n_ubatch),
                               ("n_ubatch", ctypes.c_uint32, n_ubatch), ("n_seq_max", ctypes.c_uint32, 1),
                               ("n_threads", ctypes.c_int32, n_threads), ("n_threads_batch", ctypes.c_int32, n_threads)):
            poke(cp, L.off("llama_context_params", field), ct, val)
        poke(cp, L.off("llama_context_params", "cb_eval"), ctypes.c_void_p, ctypes.cast(self._cb, ctypes.c_void_p).value)
        poke(cp, L.off("llama_context_params", "cb_eval_user_data"), ctypes.c_void_p, 0)
        self.ctx = self.llama.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("cannot create context")
        self.lora = None
        paths = [lora_path] if isinstance(lora_path, (str, Path)) else list(lora_path or [])
        scales_in = [lora_scale] * len(paths) if not isinstance(lora_scale, (list, tuple)) else list(lora_scale)
        if paths:
            self.llama.llama_adapter_lora_init.restype = ctypes.c_void_p
            self.llama.llama_adapter_lora_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            handles = []
            for pth in paths:
                h = self.llama.llama_adapter_lora_init(self.model, str(pth).encode())
                if not h:
                    raise RuntimeError(f"cannot load LoRA {pth}")
                handles.append(h)
            self.lora = handles
            adapters = (ctypes.c_void_p * len(handles))(*handles)
            scales = (ctypes.c_float * len(handles))(*scales_in)
            self.llama.llama_set_adapters_lora.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            rc = self.llama.llama_set_adapters_lora(self.ctx, ctypes.cast(adapters, ctypes.c_void_p), len(handles),
                                                    ctypes.cast(scales, ctypes.c_void_p))
            if rc != 0:
                raise RuntimeError(f"llama_set_adapters_lora failed: {rc}")

    # ------------------------------------------------------------------ binding
    def _sym(self, name: str):
        for lib in (self.llama, self.ggml, self.ggml_base):
            try:
                return getattr(lib, name)
            except AttributeError:
                continue
        raise AttributeError(name)

    def _bind(self) -> None:
        self.llama.llama_model_get_vocab.restype = ctypes.c_void_p
        self.llama.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
        self.llama.llama_model_load_from_file.restype = ctypes.c_void_p
        self.llama.llama_init_from_model.restype = ctypes.c_void_p
        self.llama.llama_get_memory.restype = ctypes.c_void_p
        self.llama.llama_get_memory.argtypes = [ctypes.c_void_p]
        self.llama.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        self.llama.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_void_p,
                                              ctypes.c_int32, ctypes.c_bool, ctypes.c_bool]
        self.llama.llama_free.argtypes = [ctypes.c_void_p]
        self.llama.llama_model_free.argtypes = [ctypes.c_void_p]
        f = self._sym("ggml_backend_dev_by_type")
        f.restype, f.argtypes = ctypes.c_void_p, [ctypes.c_int]
        f = self._sym("ggml_backend_dev_buffer_type")
        f.restype, f.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
        f = self._sym("ggml_backend_load_all_from_path")
        f.argtypes = [ctypes.c_char_p]
        f = self._sym("ggml_backend_tensor_get")
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]

    # ------------------------------------------------------------------ eval callback
    def _on_tensor(self, t, ask, _user) -> bool:
        try:
            L = self.layout
            name = ctypes.string_at(t + L.off("ggml_tensor", "name"), L.const("GGML_MAX_NAME")).split(b"\0", 1)[0]
            if not name.startswith(TOPK_PREFIX):
                return not ask   # not our tensor: no data wanted (ask) / keep going (data call)
            if ask:
                return True
            self._cb_calls += 1
            ttype = ctypes.c_int32.from_address(t + L.off("ggml_tensor", "type")).value
            if ttype != L.const("GGML_TYPE_I32"):
                raise RuntimeError(f"{name!r}: expected I32 router indices, got type {ttype}")
            ne0 = ctypes.c_int64.from_address(t + L.off("ggml_tensor", "ne")).value
            ne1 = ctypes.c_int64.from_address(t + L.off("ggml_tensor", "ne") + 8).value
            nb1 = ctypes.c_size_t.from_address(t + L.off("ggml_tensor", "nb") + 8).value
            nbytes = (ne1 - 1) * nb1 + ne0 * 4
            buf = ctypes.create_string_buffer(nbytes)
            self._sym("ggml_backend_tensor_get")(t, buf, 0, nbytes)
            layer = int(name[len(TOPK_PREFIX):])
            self._rows.setdefault(layer, []).append(parse_topk(buf.raw, ne0, ne1, nb1))
            return True
        except Exception as e:  # an exception must not cross the C boundary: abort the decode and report it
            self._err = f"{type(e).__name__}: {e}"
            return False

    # ------------------------------------------------------------------ public API
    def tokenize(self, text: str, *, add_special: bool = False, parse_special: bool = True) -> list:
        b = text.encode("utf-8")
        cap = len(b) + 16
        arr = (ctypes.c_int32 * cap)()
        n = self.llama.llama_tokenize(self.vocab, b, len(b), ctypes.addressof(arr), cap, add_special, parse_special)
        if n < 0:
            raise RuntimeError(f"llama_tokenize needs {-n} tokens (cap {cap})")
        return list(arr[:n])

    def _decode_chunk(self, token_ids, start: int) -> int:
        """Decode tokens[start:start+n] with logits requested for EVERY token. (llama_batch_get_one only requests the last
        token, and the last MoE layer is pruned to the requested outputs before its router runs, so its top-k would
        otherwise be missing for all but one token per ubatch.)"""
        L = self.layout
        m = len(token_ids)
        batch = self.llama.llama_batch_init(m, 0, 1)
        try:
            poke(batch, L.off("llama_batch", "n_tokens"), ctypes.c_int32, m)
            tok = (ctypes.c_int32 * m).from_address(peek(batch, L.off("llama_batch", "token"), ctypes.c_void_p))
            pos = (ctypes.c_int32 * m).from_address(peek(batch, L.off("llama_batch", "pos"), ctypes.c_void_p))
            nsq = (ctypes.c_int32 * m).from_address(peek(batch, L.off("llama_batch", "n_seq_id"), ctypes.c_void_p))
            sid = (ctypes.c_void_p * m).from_address(peek(batch, L.off("llama_batch", "seq_id"), ctypes.c_void_p))
            out = (ctypes.c_int8 * m).from_address(peek(batch, L.off("llama_batch", "logits"), ctypes.c_void_p))
            for j in range(m):
                tok[j], pos[j], nsq[j], out[j] = int(token_ids[j]), start + j, 1, 1
                ctypes.c_int32.from_address(sid[j]).value = 0
            return self.llama.llama_decode(self.ctx, batch)
        finally:
            self.llama.llama_batch_free(batch)

    def collect_per_token_experts(self, token_ids) -> np.ndarray:
        """Prefill the tokens (n_ubatch at a time) and return uint8 [n_tokens, n_layers, n_expert_used]."""
        n = len(token_ids)
        self._rows, self._err = {}, None
        self.llama.llama_memory_clear(self.llama.llama_get_memory(self.ctx), True)
        for i in range(0, n, self.n_ubatch):
            rc = self._decode_chunk(token_ids[i:i + self.n_ubatch], i)
            if self.progress is not None:
                self.progress(min(i + self.n_ubatch, n), n)
            if self._err or rc != 0:
                raise RuntimeError(f"llama_decode rc={rc} at token {i}: {self._err}")
        layers = sorted(self._rows)
        if not layers:
            raise RuntimeError("no ffn_moe_topk-* tensor was seen: model is not MoE or the tensor name differs in this build")
        out = []
        for l in layers:
            part = np.concatenate(self._rows[l], axis=0)
            if part.shape[0] != n:
                raise RuntimeError(f"layer {l}: collected {part.shape[0]} rows for {n} tokens")
            out.append(part)
        arr3 = np.stack(out, axis=1)
        if arr3.max() > 255 or arr3.min() < 0:
            raise RuntimeError("expert id out of uint8 range")
        return arr3.astype(np.uint8)

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.llama.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            self.llama.llama_model_free(self.model)
            self.model = None


def collect_per_token_experts(runner: Tier2Runner, token_ids) -> np.ndarray:
    """Module-level entry point named in docs/test_plan_vram_opt_streaming_experts.md (T6.1)."""
    return runner.collect_per_token_experts(token_ids)
