"""Runtime settings, overridable with HELPDESK_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"HELPDESK_{name}", default)


def default_cache() -> Path:
    return Path(_env("CACHE", str(Path.home() / ".cache" / "offline-it-helpdesk")))


@dataclass
class Settings:
    kb_dir: Path = field(default_factory=lambda: Path(_env("KB_DIR", "kb")))
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", "index")))
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", str(default_cache()))))
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "minilm-int8"))
    providers: str = field(default_factory=lambda: _env("PROVIDERS", "cpu"))
    llm: str = field(default_factory=lambda: _env("LLM", "llamacpp"))
    llm_url: str = field(default_factory=lambda: _env("LLM_URL", ""))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "3")))
    min_score: float = field(default_factory=lambda: float(_env("MIN_SCORE", "0.35")))
    max_tokens: int = field(default_factory=lambda: int(_env("MAX_TOKENS", "350")))
    fallback_score: float = field(default_factory=lambda: float(_env("FALLBACK_SCORE", "0.5")))
