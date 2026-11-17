"""Sentence embedders: ONNX MiniLM (default) and a dependency-free hashing fallback."""

from __future__ import annotations

import hashlib
import itertools
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from . import download
from .tokenizer import WordPieceTokenizer, basic_tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class OnnxEmbedder:
    """all-MiniLM-L6-v2 through ONNX Runtime: mean pooling + L2 normalisation."""

    def __init__(
        self,
        model_path: Path,
        vocab_path: Path,
        providers: str = "cpu",
        name: str = "minilm",
        max_len: int = 256,
        batch: int = 32,
    ) -> None:
        from .providers import session

        self.name = name
        self.tokenizer = WordPieceTokenizer.from_file(vocab_path)
        self.session, self.skipped = session(str(model_path), providers)
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.dim = int(self.session.get_outputs()[0].shape[-1])
        self.max_len = max_len
        self.batch = batch

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch):
            enc = self.tokenizer.encode_batch(texts[i : i + self.batch], self.max_len)
            feed = {k: v for k, v in enc.items() if k in self.input_names}
            hidden = self.session.run(None, feed)[0]
            mask = enc["attention_mask"][..., None].astype(np.float32)
            out.append((hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None))
        return _l2(np.concatenate(out).astype(np.float32))


class HashingEmbedder:
    """Feature-hashed unigrams + bigrams. No model download; used in CI and as a fallback."""

    def __init__(self, dim: int = 512) -> None:
        self.name = "hash"
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = [t for t in basic_tokenize(text) if re.match(r"\w", t)]
            for tok in toks + [f"{a}_{b}" for a, b in itertools.pairwise(toks)]:
                idx, sign = self._bucket(tok)
                mat[row, idx] += sign
        return _l2(mat)


def load(name: str, models_dir: Path, providers: str = "cpu") -> Embedder:
    if name == "hash":
        return HashingEmbedder()
    if name not in download.EMBEDDERS:
        raise SystemExit(
            f"unknown embedder {name!r}; choose from hash, {', '.join(download.EMBEDDERS)}"
        )
    assets = download.EMBEDDERS[name]
    model = download.fetch(assets["model"], models_dir)
    vocab = download.fetch(assets["vocab"], models_dir)
    return OnnxEmbedder(model, vocab, providers, name=name)
