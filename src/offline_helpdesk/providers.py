"""ONNX Runtime execution-provider selection (CPU always; others when present)."""

from __future__ import annotations

import sys
from typing import Any

EP_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "nnapi": "NnapiExecutionProvider",
    "xnnpack": "XnnpackExecutionProvider",
}


def _ort() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "onnxruntime is missing: pip install onnxruntime "
            "(Termux: pkg install python-onnxruntime)"
        ) from exc
    return ort


def options(alias: str) -> dict[str, Any]:
    if alias == "coreml":
        return {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
    if alias == "qnn":
        lib = "QnnHtp.dll" if sys.platform == "win32" else "libQnnHtp.so"
        return {"backend_path": lib}
    return {}


def resolve(requested: str) -> tuple[list[Any], dict[str, str]]:
    """Return (ORT providers list, {alias: reason skipped}). CPU is always appended."""
    available = set(_ort().get_available_providers())
    chosen: list[Any] = []
    skipped: dict[str, str] = {}
    for alias in [a.strip().lower() for a in requested.split(",") if a.strip()]:
        name = EP_ALIASES.get(alias)
        if name is None:
            skipped[alias] = "unknown provider"
        elif name not in available:
            skipped[alias] = f"{name} is not in this onnxruntime build"
        elif name != "CPUExecutionProvider":
            chosen.append((name, options(alias)))
    chosen.append("CPUExecutionProvider")
    return chosen, skipped


def session(model_path: str, requested: str = "cpu") -> tuple[Any, dict[str, str]]:
    ort = _ort()
    providers, skipped = resolve(requested)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(model_path, sess_options=so, providers=providers), skipped
