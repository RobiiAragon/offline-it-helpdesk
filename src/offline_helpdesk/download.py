"""Small, pinned, checksum-verified downloads (embedding model and optional GGUF LLMs)."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HF = "https://huggingface.co"
MINILM_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"


@dataclass(frozen=True)
class Asset:
    url: str
    filename: str
    sha256: str | None
    size_mb: float
    license: str


EMBEDDERS: dict[str, dict[str, Asset]] = {
    "minilm-int8": {
        "model": Asset(
            f"{HF}/sentence-transformers/all-MiniLM-L6-v2/resolve/{MINILM_REV}/onnx/model_qint8_arm64.onnx",
            "all-MiniLM-L6-v2.qint8.onnx",
            "4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474",
            23.0,
            "Apache-2.0",
        ),
        "vocab": Asset(
            f"{HF}/sentence-transformers/all-MiniLM-L6-v2/resolve/{MINILM_REV}/vocab.txt",
            "all-MiniLM-L6-v2.vocab.txt",
            VOCAB_SHA256,
            0.2,
            "Apache-2.0",
        ),
    },
    "minilm-fp32": {
        "model": Asset(
            f"{HF}/sentence-transformers/all-MiniLM-L6-v2/resolve/{MINILM_REV}/onnx/model.onnx",
            "all-MiniLM-L6-v2.fp32.onnx",
            "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
            90.4,
            "Apache-2.0",
        ),
        "vocab": Asset(
            f"{HF}/sentence-transformers/all-MiniLM-L6-v2/resolve/{MINILM_REV}/vocab.txt",
            "all-MiniLM-L6-v2.vocab.txt",
            VOCAB_SHA256,
            0.2,
            "Apache-2.0",
        ),
    },
}

LLMS: dict[str, Asset] = {
    "qwen2.5-1.5b": Asset(
        f"{HF}/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
        1117.3,
        "Apache-2.0",
    ),
    "qwen2.5-0.5b": Asset(
        f"{HF}/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
        491.4,
        "Apache-2.0",
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(asset: Asset, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / asset.filename
    if target.exists() and (asset.sha256 is None or sha256_file(target) == asset.sha256):
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"downloading {asset.filename} ({asset.size_mb:,.1f} MB, {asset.license})")
    try:
        with urllib.request.urlopen(asset.url, timeout=60) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total and sys.stdout.isatty():
                    print(f"\r  {done / total:6.1%}", end="", flush=True)
    except BaseException:  # interrupted or failed: don't leave a half-written .part file
        tmp.unlink(missing_ok=True)
        raise
    if sys.stdout.isatty():
        print()
    if asset.sha256 and (got := sha256_file(tmp)) != asset.sha256:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {asset.filename}: {got}")
    tmp.replace(target)
    return target
