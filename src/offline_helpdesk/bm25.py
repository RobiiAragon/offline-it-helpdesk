"""Okapi BM25 over the same tokenizer used for embeddings."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .tokenizer import basic_tokenize

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    ]
)


def terms(text: str) -> list[str]:
    return [t for t in basic_tokenize(text) if t.isalnum() and t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = [Counter(terms(d)) for d in docs]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=np.float64)
        self.avgdl = float(self.lens.mean()) if len(self.lens) else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tfs), dtype=np.float64)
        norm = self.k1 * (1 - self.b + self.b * self.lens / (self.avgdl or 1.0))
        for term in set(terms(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.tfs], dtype=np.float64)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
