"""Split Markdown runbooks into heading-aware chunks that keep their provenance."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Chunk:
    id: str
    doc: str  # file name, e.g. "vpn.md"
    title: str  # document H1
    section: str  # nearest H2/H3, or "" for the intro
    text: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def search_text(self) -> str:
        """Title and section are prepended so short sections still match on topic."""
        head = f"{self.title} - {self.section}" if self.section else self.title
        return f"{head}\n{self.text}"


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        if current and len(current) + len(para) > max_chars:
            parts.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_markdown(doc: str, markdown: str, max_chars: int = 1200) -> list[Chunk]:
    title, section, buf = Path(doc).stem.replace("-", " ").title(), "", []
    sections: list[tuple[str, str, str]] = []  # (title, section, body)

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((title, section, body))
        buf.clear()

    in_fence = False
    for line in markdown.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        # "# comment" lines inside ``` code blocks are not headings.
        m = None if in_fence else HEADING.match(line)
        if m and len(m.group(1)) == 1:
            flush()
            title = m.group(2).strip()
            section = ""  # a new document title starts with its intro, not the old section
        elif m:
            flush()
            section = m.group(2).strip()
        else:
            buf.append(line)
    flush()

    chunks: list[Chunk] = []
    for head, sec, body in sections:
        for part in _split_long(body, max_chars):
            chunks.append(Chunk(f"{Path(doc).stem}#{len(chunks)}", doc, head, sec, part))
    return chunks


def load_kb(kb_dir: Path) -> list[Chunk]:
    files = sorted(kb_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"no Markdown files found in {kb_dir}")
    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(chunk_markdown(path.name, path.read_text(encoding="utf-8")))
    return chunks
