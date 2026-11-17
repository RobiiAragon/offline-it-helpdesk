from __future__ import annotations

import pytest

from offline_helpdesk.bm25 import BM25, terms
from offline_helpdesk.chunker import chunk_markdown, load_kb
from offline_helpdesk.tokenizer import WordPieceTokenizer, basic_tokenize

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "vpn",
    "connect",
    "##ed",
    "##ing",
    "the",
    "is",
    "cafe",
    "naive",
    ".",
    ",",
    "!",
    "un",
    "##able",
    "to",
    "c",
    ":",
    "\\",
    "temp",
    "%",
]


@pytest.fixture()
def vocab_file(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    return path


def test_basic_tokenize_lowercases_strips_accents_splits_punct() -> None:
    assert basic_tokenize("Café, NAÏVE! C:\\Temp") == [
        "cafe",
        ",",
        "naive",
        "!",
        "c",
        ":",
        "\\",
        "temp",
    ]


def test_wordpiece_matches_reference_tokenizer(vocab_file) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    ref = tokenizers.BertWordPieceTokenizer(str(vocab_file), lowercase=True)
    ours = WordPieceTokenizer.from_file(vocab_file)
    for text in [
        "The VPN is connected.",
        "Unable to connect, the vpn!",
        "Café naïve %temp%",
        "unknownword vpnconnecting",
        "  spaces\tand\nnewlines  ",
    ]:
        assert ours.tokenize_ids(text) == ref.encode(text, add_special_tokens=False).ids, text


def test_encode_batch_pads_and_truncates(vocab_file) -> None:
    tok = WordPieceTokenizer.from_file(vocab_file)
    enc = tok.encode_batch(["vpn", "the vpn is connected"], max_len=4)
    assert enc["input_ids"].shape == (2, 4)
    assert enc["input_ids"][0].tolist() == [2, 4, 3, 0]
    assert enc["input_ids"][1].tolist() == [2, 8, 4, 3]  # truncated to fit [CLS] .. [SEP]
    assert enc["attention_mask"].sum(1).tolist() == [3, 4]


def test_chunker_tracks_title_and_sections() -> None:
    md = "# VPN Guide\nintro text\n## Connecting\nstep one\n### Details\nmore\n"
    chunks = chunk_markdown("vpn.md", md)
    assert [(c.title, c.section) for c in chunks] == [
        ("VPN Guide", ""),
        ("VPN Guide", "Connecting"),
        ("VPN Guide", "Details"),
    ]
    assert chunks[1].search_text.startswith("VPN Guide - Connecting")
    assert chunks[0].id == "vpn#0"


def test_chunker_splits_long_sections() -> None:
    body = "\n\n".join(f"paragraph {i} " + "x" * 300 for i in range(6))
    chunks = chunk_markdown("long.md", f"# Long\n## Part\n{body}", max_chars=700)
    assert len(chunks) > 1 and all(len(c.text) <= 700 for c in chunks)


def test_bundled_kb_loads() -> None:
    from conftest import KB

    chunks = load_kb(KB)
    assert len({c.doc for c in chunks}) >= 15
    assert all(c.text.strip() for c in chunks)


def test_bm25_prefers_matching_document() -> None:
    bm = BM25(["printer toner jam", "vpn portal address", "password reset portal"])
    scores = bm.scores("what is the vpn portal?")
    assert scores.argmax() == 1
    assert "the" not in terms("the vpn")
