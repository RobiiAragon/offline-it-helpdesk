from __future__ import annotations

from pathlib import Path

import pytest

from offline_helpdesk.embedder import HashingEmbedder
from offline_helpdesk.index import Index
from offline_helpdesk.llm import Generation

REPO = Path(__file__).resolve().parents[1]
KB = REPO / "kb"


class FakeLLM:
    """Returns a canned reply; records the prompt it received."""

    name = "fake"

    def __init__(self, reply: str = "Wait 30 minutes or call extension 4357 [1].") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, system: str, user: str, max_tokens: int) -> Generation:
        self.prompts.append(user)
        return Generation(self.reply, self.name, "fake-model", 5.0, 12)


@pytest.fixture(scope="session")
def kb_index() -> Index:
    return Index.build(KB, HashingEmbedder())
