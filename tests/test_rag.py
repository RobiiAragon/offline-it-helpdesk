from __future__ import annotations

import numpy as np
from conftest import FakeLLM

from offline_helpdesk.embedder import HashingEmbedder
from offline_helpdesk.index import Index
from offline_helpdesk.llm import ExtractiveLLM
from offline_helpdesk.rag import NOT_FOUND, Assistant, build_prompt


def test_index_roundtrip_and_hybrid_search(kb_index, tmp_path) -> None:
    kb_index.save(tmp_path)
    loaded = Index.load(tmp_path, HashingEmbedder())
    assert len(loaded.chunks) == len(kb_index.chunks)
    assert np.allclose(loaded.vectors, kb_index.vectors)
    hits = loaded.search("GlobalProtect VPN portal address", k=3)
    assert hits[0].chunk.doc == "vpn.md"
    assert hits[0].score >= hits[-1].score


def test_index_rejects_other_embedder(kb_index, tmp_path) -> None:
    kb_index.save(tmp_path)
    other = HashingEmbedder()
    other.name = "minilm-int8"
    try:
        Index.load(tmp_path, other)
    except SystemExit as exc:
        assert "rebuild" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_answer_with_valid_and_invalid_citations(kb_index) -> None:
    llm = FakeLLM("Wait 30 minutes [1]. Call extension 4357 [9].")
    ans = Assistant(kb_index, llm, top_k=3, min_score=0.0).ask("my account is locked out")
    assert not ans.refused and ans.reason == "answered"
    assert [c.n for c in ans.citations] == [1]
    assert ans.invalid_citations == [9]
    assert ans.completion_tokens == 12
    assert "[1] (" in llm.prompts[0] and "Question: my account is locked out" in llm.prompts[0]


def test_low_retrieval_score_refuses_without_calling_llm(kb_index) -> None:
    llm = FakeLLM()
    ans = Assistant(kb_index, llm, min_score=0.99).ask("what is the capital of france")
    assert ans.refused and ans.reason == "low_retrieval_score"
    assert ans.answer == NOT_FOUND and llm.prompts == []


def test_llm_decline_falls_back_to_extractive_when_confident(kb_index) -> None:
    llm = FakeLLM("I don't know based on the knowledge base.")
    confident = Assistant(kb_index, llm, min_score=0.0, fallback_score=0.0)
    ans = confident.ask("account lockout failed sign-in attempts")
    assert not ans.refused and ans.reason == "extractive_fallback" and ans.citations
    strict = Assistant(kb_index, llm, min_score=0.0, fallback_score=None)
    ans = strict.ask("account lockout failed sign-in attempts")
    assert ans.refused and ans.reason == "llm_declined"


def test_extractive_llm_quotes_sources_with_citations(kb_index) -> None:
    hits = kb_index.search("print job badge release", k=2)
    gen = ExtractiveLLM().generate("", build_prompt("q", hits), 100)
    assert "[1]" in gen.text and hits[0].chunk.text.splitlines()[0].strip()[:20] in gen.text
