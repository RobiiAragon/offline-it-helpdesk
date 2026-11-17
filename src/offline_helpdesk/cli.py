"""``helpdesk`` command-line interface."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from . import __version__, download
from .config import Settings


def _assistant(args: argparse.Namespace, s: Settings):
    from . import embedder, llm
    from .index import Index
    from .rag import Assistant

    emb = embedder.load(args.embedder or s.embedder, s.models_dir, args.providers or s.providers)
    index = Index.load(Path(args.index_dir or s.index_dir), emb)
    model = llm.make(args.llm or s.llm, args.llm_url or s.llm_url, args.llm_model or s.llm_model)
    fallback = None if args.no_fallback else s.fallback_score
    return Assistant(
        index,
        model,
        top_k=args.top_k or s.top_k,
        min_score=args.min_score or s.min_score,
        max_tokens=s.max_tokens,
        fallback_score=fallback,
    )


def _served_model(model: object) -> str | None:
    """Ask the LLM server which model it is serving (llama.cpp reports the GGUF path)."""
    check = getattr(model, "health", None)
    if check is None:
        return None
    try:
        return Path(check()).name
    except Exception:
        return getattr(model, "model", None)


def cmd_download(args: argparse.Namespace, s: Settings) -> int:
    for asset in download.EMBEDDERS[args.embedder].values():
        print(" ->", download.fetch(asset, s.models_dir))
    if args.llm:
        print(" ->", download.fetch(download.LLMS[args.llm], s.models_dir))
    return 0


def cmd_index(args: argparse.Namespace, s: Settings) -> int:
    import time

    from . import embedder
    from .index import Index

    emb = embedder.load(args.embedder or s.embedder, s.models_dir, args.providers or s.providers)
    t0 = time.perf_counter()
    idx = Index.build(Path(args.kb_dir or s.kb_dir), emb)
    out = Path(args.index_dir or s.index_dir)
    idx.save(out)
    docs = len({c.doc for c in idx.chunks})
    print(
        f"indexed {len(idx.chunks)} chunks from {docs} documents with {emb.name} "
        f"in {time.perf_counter() - t0:.2f}s -> {out}"
    )
    return 0


def cmd_ask(args: argparse.Namespace, s: Settings) -> int:
    from .llm import LLMUnavailable

    assistant = _assistant(args, s)
    try:
        ans = assistant.ask(" ".join(args.question))
    except LLMUnavailable as exc:
        print(
            f"{exc}\nStart it with scripts/start_llm.sh, or use --llm extractive.", file=sys.stderr
        )
        return 3
    if args.json:
        print(json.dumps(ans.as_dict(), indent=2, ensure_ascii=False))
        return 0
    print(ans.answer)
    for c in ans.citations:
        where = f"{c.title} > {c.section}" if c.section else c.title
        print(f"  [{c.n}] {c.doc}: {where} (similarity {c.dense:.2f})")
    print(
        f"\n({ans.backend}; retrieval {ans.retrieve_ms:.0f} ms, generation "
        f"{ans.generate_ms:.0f} ms)"
    )
    return 0


def cmd_eval(args: argparse.Namespace, s: Settings) -> int:
    from . import device_info
    from .evaluate import format_summary, load_questions, run

    assistant = _assistant(args, s)
    result = run(assistant, load_questions(args.questions), verbose=True)
    print("\n" + format_summary(result["summary"]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result["meta"] = {
            "created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "version": __version__,
            "device": device_info.collect(),
            "embedder": assistant.index.embedder.name,
            "llm": assistant.llm.name,
            "llm_model": _served_model(assistant.llm),
            "top_k": assistant.top_k,
            "min_score": assistant.min_score,
            "fallback_score": assistant.fallback_score,
        }
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}")
    return 0


def cmd_serve(args: argparse.Namespace, s: Settings) -> int:
    import uvicorn

    from .api import create_app

    app = create_app(_assistant(args, s), Path(args.kb_dir or s.kb_dir))
    print(f"Offline IT Helpdesk on http://{args.host}:{args.port}  (API docs at /docs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_doctor(args: argparse.Namespace, s: Settings) -> int:
    from . import llm, providers

    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
    try:
        import onnxruntime as ort

        print(f"onnxruntime {ort.__version__}: {', '.join(ort.get_available_providers())}")
        _, skipped = providers.resolve(args.providers or s.providers)
        for alias, why in skipped.items():
            print(f"  [skip] {alias}: {why}")
    except ImportError:
        print("onnxruntime: NOT INSTALLED")
    model = llm.make(args.llm or s.llm, args.llm_url or s.llm_url, args.llm_model or s.llm_model)
    check = getattr(model, "health", None)
    if check is None:
        print(f"llm: {model.name} (no server needed)")
        return 0
    try:
        print(f"llm: {model.name} OK -> {check()}")
    except llm.LLMUnavailable as exc:
        print(f"llm: {exc}\n  start one with scripts/start_llm.sh")
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="helpdesk", description="Offline IT helpdesk assistant")
    p.add_argument("--version", action="version", version=__version__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--kb-dir")
    common.add_argument("--index-dir")
    common.add_argument("--embedder", help="minilm-int8 (default), minilm-fp32 or hash")
    common.add_argument("--providers", help="ONNX EPs for embeddings, e.g. cpu,coreml,qnn")
    common.add_argument("--llm", help="llamacpp (default), ollama or extractive")
    common.add_argument("--llm-url")
    common.add_argument("--llm-model")
    common.add_argument("--min-score", type=float)
    common.add_argument("--top-k", type=int)
    common.add_argument(
        "--no-fallback",
        action="store_true",
        help="refuse when the LLM declines, even if retrieval is confident",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download the embedding model (and optionally an LLM)")
    d.add_argument("--embedder", default="minilm-int8", choices=list(download.EMBEDDERS))
    d.add_argument("--llm", choices=list(download.LLMS))
    d.set_defaults(func=cmd_download)

    sub.add_parser(
        "index", parents=[common], help="chunk and embed the knowledge base"
    ).set_defaults(func=cmd_index)

    a = sub.add_parser("ask", parents=[common], help="ask one question")
    a.add_argument("question", nargs="+")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval", parents=[common], help="run the evaluation question set")
    e.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    e.add_argument("--out", type=Path)
    e.set_defaults(func=cmd_eval)

    sv = sub.add_parser("serve", parents=[common], help="run the FastAPI service + web UI")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)

    sub.add_parser(
        "doctor", parents=[common], help="check ONNX Runtime and the LLM server"
    ).set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args, Settings()) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
