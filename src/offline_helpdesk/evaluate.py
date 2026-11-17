"""Evaluate retrieval, answer correctness, citations and refusals on a question set."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rag import Assistant


@dataclass
class Question:
    id: str
    question: str
    expected_doc: str | None
    must_include: list[str]
    answerable: bool


def load_questions(path: Path) -> list[Question]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Question(**json.loads(line)) for line in lines if line.strip()]


def facts_found(answer: str, must_include: list[str]) -> tuple[int, list[str]]:
    """Each entry may list alternatives separated by '|'. Returns (#found, missing)."""
    text = answer.lower()
    missing = [
        group
        for group in must_include
        if not any(alt.strip().lower() in text for alt in group.split("|"))
    ]
    return len(must_include) - len(missing), missing


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def run(assistant: Assistant, questions: list[Question], verbose: bool = False) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for q in questions:
        t0 = time.perf_counter()
        ans = assistant.ask(q.question)
        total_ms = (time.perf_counter() - t0) * 1000
        docs = [r["doc"] for r in ans.retrieved]
        rank = docs.index(q.expected_doc) + 1 if q.expected_doc in docs else None
        found, missing = (
            facts_found(ans.answer, q.must_include) if not ans.refused else (0, q.must_include)
        )
        cited_docs = sorted({c.doc for c in ans.citations})
        row = {
            "id": q.id,
            "question": q.question,
            "answerable": q.answerable,
            "expected_doc": q.expected_doc,
            "retrieval_rank": rank,
            "refused": ans.refused,
            "reason": ans.reason,
            "facts_found": found,
            "facts_total": len(q.must_include),
            "missing": missing,
            "cited_docs": cited_docs,
            "cites_expected": q.expected_doc in cited_docs if q.expected_doc else None,
            "invalid_citations": ans.invalid_citations,
            "reason_detail": ans.reason,
            "top_dense": max((r["dense"] for r in ans.retrieved), default=0.0),
            "retrieve_ms": ans.retrieve_ms,
            "generate_ms": ans.generate_ms,
            "total_ms": total_ms,
            "completion_tokens": ans.completion_tokens,
            "answer": ans.answer,
        }
        rows.append(row)
        if verbose:
            status = "REFUSED" if ans.refused else f"{found}/{len(q.must_include)} facts"
            print(f"{q.id:<6} rank={rank or '-'} {status:<12} {total_ms:7.0f} ms  {q.question}")

    ans_rows = [r for r in rows if r["answerable"]]
    oos_rows = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans_rows if not r["refused"]]
    ranks = [r["retrieval_rank"] for r in ans_rows]
    gen = [r["generate_ms"] for r in rows if r["generate_ms"] > 0]
    summary = {
        "n_answerable": len(ans_rows),
        "n_out_of_scope": len(oos_rows),
        "retrieval_hit@1": _mean([r == 1 for r in ranks]),
        "retrieval_hit@k": _mean([r is not None for r in ranks]),
        "retrieval_mrr": _mean([1 / r if r else 0.0 for r in ranks]),
        "answer_fact_recall": _mean([r["facts_found"] / r["facts_total"] for r in ans_rows]),
        "answer_fully_correct": _mean([r["facts_found"] == r["facts_total"] for r in ans_rows]),
        "cites_expected_doc": _mean([bool(r["cites_expected"]) for r in answered]),
        "answers_with_citation": _mean([bool(r["cited_docs"]) for r in answered]),
        "invalid_citation_rate": _mean([bool(r["invalid_citations"]) for r in answered]),
        "false_refusal_rate": _mean([r["refused"] for r in ans_rows]),
        "oos_refusal_rate": _mean([r["refused"] for r in oos_rows]),
        "retrieve_ms_p50": _pct([r["retrieve_ms"] for r in rows], 50),
        "generate_ms_p50": _pct(gen, 50),
        "generate_ms_p95": _pct(gen, 95),
        "total_ms_p50": _pct([r["total_ms"] for r in rows], 50),
        "tokens_per_s_p50": _pct(
            [
                r["completion_tokens"] / (r["generate_ms"] / 1000)
                for r in rows
                if r["completion_tokens"] and r["generate_ms"]
            ],
            50,
        ),
        "fallback_rate": _mean([r["reason"] == "extractive_fallback" for r in ans_rows]),
        "wall_s": time.perf_counter() - t_start,
    }
    return {"summary": summary, "rows": rows}


def _mean(values: list[Any]) -> float | None:
    return float(np.mean([float(v) for v in values])) if values else None


def format_summary(summary: dict[str, Any]) -> str:
    def pct(key: str) -> str:
        v = summary.get(key)
        return "n/a" if v is None else f"{v * 100:.1f}%"

    def ms(key: str) -> str:
        v = summary.get(key)
        return "n/a" if v is None else f"{v:,.0f} ms"

    return "\n".join(
        [
            f"questions: {summary['n_answerable']} answerable + "
            f"{summary['n_out_of_scope']} out-of-scope",
            f"retrieval hit@1 {pct('retrieval_hit@1')} | hit@k {pct('retrieval_hit@k')} | "
            f"MRR {summary['retrieval_mrr']:.3f}",
            f"answers: fact recall {pct('answer_fact_recall')} | fully correct "
            f"{pct('answer_fully_correct')}",
            f"citations: cites expected doc {pct('cites_expected_doc')} | has citation "
            f"{pct('answers_with_citation')} | invalid {pct('invalid_citation_rate')}",
            f"refusals: false refusals {pct('false_refusal_rate')} | out-of-scope refused "
            f"{pct('oos_refusal_rate')}",
            f"latency: retrieve p50 {ms('retrieve_ms_p50')} | generate p50 {ms('generate_ms_p50')} "
            f"p95 {ms('generate_ms_p95')} | end-to-end p50 {ms('total_ms_p50')}",
            f"generation speed p50 {summary.get('tokens_per_s_p50') or 0:,.0f} tok/s | "
            f"extractive fallback used {pct('fallback_rate')}",
        ]
    )
