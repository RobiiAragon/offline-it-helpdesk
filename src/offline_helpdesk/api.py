"""FastAPI service: POST /ask, GET /health, GET /kb/{doc}, and the web UI at /."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, PlainTextResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise SystemExit('the API needs FastAPI: pip install -e ".[server]"') from exc

from . import __version__
from .rag import Assistant


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def create_app(assistant: Assistant, kb_dir: Path) -> FastAPI:
    app = FastAPI(
        title="Offline IT Helpdesk",
        version=__version__,
        description="Local RAG assistant over IT runbooks. Nothing leaves the machine.",
    )
    kb_root = kb_dir.resolve()
    ui = files("offline_helpdesk").joinpath("web/index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return ui

    @app.get("/health")
    def health() -> dict[str, Any]:
        llm = assistant.llm
        status = "ok"
        detail = ""
        check = getattr(llm, "health", None)
        if check is not None:
            try:
                detail = check()
            except Exception as exc:  # report, don't fail the health endpoint
                status, detail = "llm_unreachable", str(exc)
        return {
            "status": status,
            "llm_backend": llm.name,
            "llm_detail": detail,
            "embedder": assistant.index.embedder.name,
            "chunks": len(assistant.index.chunks),
            "version": __version__,
        }

    @app.post("/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        try:
            return assistant.ask(req.question).as_dict()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/kb")
    def list_docs() -> list[str]:
        return sorted(p.name for p in kb_root.glob("*.md"))

    @app.get("/kb/{doc}", response_class=PlainTextResponse)
    def get_doc(doc: str) -> str:
        path = (kb_root / doc).resolve()
        if path.parent != kb_root or path.suffix != ".md" or not path.is_file():
            raise HTTPException(status_code=404, detail="document not found")
        return path.read_text(encoding="utf-8")

    return app
