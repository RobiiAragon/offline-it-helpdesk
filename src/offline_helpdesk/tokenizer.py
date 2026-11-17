"""Pure-Python BERT (uncased) WordPiece tokenizer.

Matches HuggingFace ``BertTokenizer(do_lower_case=True)`` for the MiniLM embedding model
without needing the Rust ``tokenizers`` wheel, which keeps the project installable on
Android/Termux where no wheel exists.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np


def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


def _clean(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if cp in (0, 0xFFFD) or (unicodedata.category(ch).startswith("C") and ch not in "\t\n\r"):
            continue
        if _is_cjk(cp):
            out.append(f" {ch} ")
        elif ch in "\t\n\r" or unicodedata.category(ch) == "Zs":
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def basic_tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for word in _clean(text).lower().split():
        word = _strip_accents(word)
        current = ""
        for ch in word:
            if _is_punct(ch):
                if current:
                    tokens.append(current)
                    current = ""
                tokens.append(ch)
            else:
                current += ch
        if current:
            tokens.append(current)
    return tokens


class WordPieceTokenizer:
    def __init__(self, vocab: dict[str, int], max_word_chars: int = 100) -> None:
        self.vocab = vocab
        self.max_word_chars = max_word_chars
        self.cls_id = vocab["[CLS]"]
        self.sep_id = vocab["[SEP]"]
        self.pad_id = vocab["[PAD]"]
        self.unk_id = vocab["[UNK]"]

    @classmethod
    def from_file(cls, path: str | Path) -> WordPieceTokenizer:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls({tok: i for i, tok in enumerate(lines) if tok})

    def _wordpiece(self, word: str) -> list[int]:
        if len(word) > self.max_word_chars:
            return [self.unk_id]
        ids: list[int] = []
        start = 0
        while start < len(word):
            end = len(word)
            match = None
            while start < end:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.vocab:
                    match = self.vocab[piece]
                    break
                end -= 1
            if match is None:
                return [self.unk_id]
            ids.append(match)
            start = end
        return ids

    def tokenize_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in basic_tokenize(text):
            ids.extend(self._wordpiece(word))
        return ids

    def encode_batch(self, texts: list[str], max_len: int = 256) -> dict[str, np.ndarray]:
        rows = [[self.cls_id, *self.tokenize_ids(t)[: max_len - 2], self.sep_id] for t in texts]
        width = max(len(r) for r in rows)
        ids = np.full((len(rows), width), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = r
            mask[i, : len(r)] = 1
        return {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
