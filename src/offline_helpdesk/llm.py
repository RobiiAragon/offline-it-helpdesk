"""Local LLM backends.

* ``llamacpp`` - llama.cpp's ``llama-server`` (OpenAI-compatible, default :8080)
* ``ollama``   - Ollama's OpenAI-compatible endpoint (default :11434)
* ``extractive`` - no LLM: quotes the best passages verbatim (CI, low-end devices)

Only the Python standard library is used for HTTP, so this also runs in Termux.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

DEFAULT_URLS = {"llamacpp": "http://127.0.0.1:8080/v1", "ollama": "http://127.0.0.1:11434/v1"}
DEFAULT_MODELS = {"llamacpp": "local", "ollama": "qwen2.5:1.5b"}


@dataclass
class Generation:
    text: str
    backend: str
    model: str
    elapsed_ms: float
    completion_tokens: int | None = None


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str, max_tokens: int) -> Generation: ...


class LLMUnavailable(RuntimeError):
    pass


class OpenAICompatLLM:
    def __init__(self, backend: str, url: str = "", model: str = "", timeout: float = 120) -> None:
        self.name = backend
        self.url = (url or DEFAULT_URLS[backend]).rstrip("/")
        self.model = model or DEFAULT_MODELS[backend]
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise LLMUnavailable(f"{self.name} at {self.url} is not reachable: {exc}") from exc

    def health(self) -> str:
        try:
            with urllib.request.urlopen(f"{self.url}/models", timeout=3) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
            raise LLMUnavailable(f"{self.name} at {self.url} is not reachable: {exc}") from exc
        ids = [m.get("id", "?") for m in data.get("data", [])]
        return ", ".join(ids) or "(no models listed)"

    def generate(self, system: str, user: str, max_tokens: int) -> Generation:
        t0 = time.perf_counter()
        data = self._post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage") or {}
        return Generation(
            text,
            self.name,
            data.get("model", self.model),
            (time.perf_counter() - t0) * 1000,
            usage.get("completion_tokens"),
        )


class ExtractiveLLM:
    """Quotes the most relevant lines of the top sources with citations. Deterministic."""

    name = "extractive"

    def __init__(self, max_sources: int = 2, max_lines: int = 6) -> None:
        self.max_sources = max_sources
        self.max_lines = max_lines

    def generate(self, system: str, user: str, max_tokens: int) -> Generation:
        t0 = time.perf_counter()
        blocks = user.split("\n[")
        lines: list[str] = ["Here is what the knowledge base says:"]
        for block in blocks[1 : self.max_sources + 1]:
            n = block.split("]", 1)[0]
            body = block.split("\n", 1)[1] if "\n" in block else ""
            body = body.split("\n\nQuestion:")[0]
            kept = [ln.strip() for ln in body.splitlines() if ln.strip()][: self.max_lines]
            if kept:
                lines.append(f"- {' '.join(kept)} [{n}]")
        return Generation("\n".join(lines), self.name, "none", (time.perf_counter() - t0) * 1000)


def make(backend: str, url: str = "", model: str = "") -> LLM:
    if backend in ("none", "extractive"):
        return ExtractiveLLM()
    if backend not in DEFAULT_URLS:
        raise SystemExit(f"unknown LLM backend {backend!r} (llamacpp, ollama, extractive)")
    return OpenAICompatLLM(backend, url, model)
