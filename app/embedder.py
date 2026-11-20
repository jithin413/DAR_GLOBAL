"""Tiny dependency-light sentence embedder (BAAI/bge-small-en-v1.5, ONNX, int8).

Uses only onnxruntime + tokenizers, so the Docker image stays small (no torch).
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

MODEL_REPO = "Xenova/bge-small-en-v1.5"
MODEL_FILES = {
    "model.onnx": "onnx/model_quantized.onnx",
    "tokenizer.json": "tokenizer.json",
}
MODEL_DIR = Path(os.getenv("EMBED_MODEL_DIR", Path(__file__).resolve().parent.parent / "models" / "bge-small"))
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def ensure_model() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for local, remote in MODEL_FILES.items():
        dst = MODEL_DIR / local
        if not dst.exists():
            url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{remote}"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dst)
    return MODEL_DIR


class Embedder:
    dim = 384

    def __init__(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = ensure_model()
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=512)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        self.sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def _encode(self, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(None, feed)[0]
        cls = out[:, 0, :]  # bge uses CLS pooling
        return cls / np.linalg.norm(cls, axis=1, keepdims=True).clip(min=1e-12)

    def embed_docs(self, texts: list[str], batch: int = 32) -> np.ndarray:
        vecs = [self._encode(texts[i:i + batch]) for i in range(0, len(texts), batch)]
        return np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([QUERY_PREFIX + text])[0].astype(np.float32)
