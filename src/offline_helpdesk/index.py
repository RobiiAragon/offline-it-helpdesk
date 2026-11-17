"""Hybrid retrieval index: dense embeddings + BM25 fused with reciprocal rank fusion."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .chunker import Chunk, load_kb
from .embedder import Embedder


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused RRF score (ranking)
    dense: float  # cosine similarity (used for the "do we know this?" guard)
    bm25: float


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.embedder = embedder
        self.bm25 = BM25([c.search_text for c in chunks])

    # ----------------------------------------------------------------- build/save
    @classmethod
    def build(cls, kb_dir: Path, embedder: Embedder) -> Index:
        chunks = load_kb(kb_dir)
        return cls(chunks, embedder.embed([c.search_text for c in chunks]), embedder)

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", self.vectors)
        meta = {
            "embedder": self.embedder.name,
            "dim": int(self.vectors.shape[1]),
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunks": [c.as_dict() for c in self.chunks],
        }
        (out / "chunks.json").write_text(
            json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> Index:
        meta_path = path / "chunks.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no index at {path}; run `helpdesk index` first")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["embedder"] != embedder.name:
            raise SystemExit(
                f"index was built with {meta['embedder']!r} but embedder is "
                f"{embedder.name!r}; rebuild with `helpdesk index`"
            )
        chunks = [Chunk(**c) for c in meta["chunks"]]
        return cls(chunks, np.load(path / "vectors.npy"), embedder)

    # --------------------------------------------------------------------- search
    def search(self, query: str, k: int = 4, rrf_k: int = 60) -> list[Hit]:
        q = self.embedder.embed([query])[0]
        dense = self.vectors @ q
        lexical = self.bm25.scores(query)
        fused = np.zeros(len(self.chunks))
        for scores in (dense, lexical):
            order = np.argsort(-scores, kind="stable")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(order))
            fused += 1.0 / (rrf_k + ranks + 1)
        top = np.argsort(-fused, kind="stable")[:k]
        return [
            Hit(self.chunks[i], float(fused[i]), float(dense[i]), float(lexical[i])) for i in top
        ]
