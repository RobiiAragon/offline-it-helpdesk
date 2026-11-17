"""Retrieve -> guard -> generate -> verify citations."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .index import Hit, Index
from .llm import LLM, ExtractiveLLM

SYSTEM_PROMPT = """You are the IT Service Desk assistant for a company.
Answer the user's question using ONLY the numbered sources provided.
Rules:
- Be concise and practical. Use short numbered steps when the answer is a procedure.
- After every sentence or step that uses a source, cite it like [1] or [2].
- Never invent URLs, phone numbers, extensions, limits or time frames that are not in the sources.
- If the sources do not contain the answer, reply exactly: "I don't know based on the knowledge base." and nothing else.
- Never ask for or repeat passwords.

Example of the required format:
1. Open the self-service portal at https://sspr.example.com [1].
2. Wait up to 15 minutes for the new password to sync [1]."""

NOT_FOUND = (
    "I couldn't find this in the IT knowledge base, so I won't guess. "
    "Please open a ticket with the IT Service Desk."
)
DECLINE_MARKER = "i don't know based on the knowledge base"
CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    n: int
    doc: str
    title: str
    section: str
    snippet: str
    dense: float


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    refused: bool
    # "answered" | "low_retrieval_score" | "llm_declined" | "extractive_fallback"
    reason: str
    backend: str
    retrieve_ms: float
    generate_ms: float
    invalid_citations: list[int] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    completion_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_prompt(question: str, hits: list[Hit]) -> str:
    parts = ["Sources:"]
    for n, hit in enumerate(hits, 1):
        c = hit.chunk
        where = f"{c.title} > {c.section}" if c.section else c.title
        parts.append(f"\n[{n}] ({c.doc} - {where})\n{c.text}")
    parts.append(
        f"\n\nQuestion: {question}\n"
        "Answer using only the sources above. End every sentence or step with its source "
        "number in square brackets, e.g. [1]."
    )
    return "\n".join(parts)


class Assistant:
    def __init__(
        self,
        index: Index,
        llm: LLM,
        top_k: int = 3,
        min_score: float = 0.35,
        max_tokens: int = 350,
        fallback_score: float | None = 0.5,
    ) -> None:
        """``fallback_score``: if the LLM declines although the best passage is at least this
        similar, quote the passages verbatim instead of refusing (None disables it)."""
        self.index = index
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self.max_tokens = max_tokens
        self.fallback_score = fallback_score

    def ask(self, question: str) -> Answer:
        question = question.strip()
        t0 = time.perf_counter()
        hits = self.index.search(question, self.top_k)
        retrieve_ms = (time.perf_counter() - t0) * 1000
        retrieved = [
            {
                "doc": h.chunk.doc,
                "section": h.chunk.section,
                "dense": round(h.dense, 4),
                "bm25": round(h.bm25, 3),
            }
            for h in hits
        ]
        best = max((h.dense for h in hits), default=0.0)
        if not hits or best < self.min_score:
            return Answer(
                question,
                NOT_FOUND,
                [],
                True,
                "low_retrieval_score",
                self.llm.name,
                retrieve_ms,
                0.0,
                retrieved=retrieved,
            )

        prompt = build_prompt(question, hits)
        gen = self.llm.generate(SYSTEM_PROMPT, prompt, self.max_tokens)
        reason = "answered"
        if DECLINE_MARKER in gen.text.lower() and len(gen.text) < 120:
            if self.fallback_score is None or best < self.fallback_score:
                return Answer(
                    question,
                    NOT_FOUND,
                    [],
                    True,
                    "llm_declined",
                    gen.backend,
                    retrieve_ms,
                    gen.elapsed_ms,
                    retrieved=retrieved,
                )
            # Small LLMs sometimes decline answerable questions; retrieval is confident,
            # so show the evidence verbatim rather than refusing.
            elapsed = gen.elapsed_ms
            gen = ExtractiveLLM(max_sources=1).generate(SYSTEM_PROMPT, prompt, self.max_tokens)
            gen.elapsed_ms += elapsed
            reason = "extractive_fallback"

        cited = sorted({int(n) for n in CITATION.findall(gen.text)})
        valid = [n for n in cited if 1 <= n <= len(hits)]
        invalid = [n for n in cited if n not in valid]
        citations = [
            Citation(
                n,
                hits[n - 1].chunk.doc,
                hits[n - 1].chunk.title,
                hits[n - 1].chunk.section,
                _snippet(hits[n - 1].chunk.text),
                round(hits[n - 1].dense, 4),
            )
            for n in valid
        ]
        return Answer(
            question,
            gen.text,
            citations,
            False,
            reason,
            gen.backend,
            retrieve_ms,
            gen.elapsed_ms,
            invalid,
            retrieved,
            gen.completion_tokens,
        )


def _snippet(text: str, limit: int = 220) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rsplit(" ", 1)[0] + "…"
