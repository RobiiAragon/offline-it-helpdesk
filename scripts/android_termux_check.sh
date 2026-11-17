#!/data/data/com.termux/files/usr/bin/bash
# Run the helpdesk fully offline on Android via Termux (target: Galaxy Tab S9 Ultra, SD 8 Gen 2).
# Embeddings: ONNX Runtime (CPU/XNNPACK) | LLM: llama.cpp on CPU with Qwen2.5-0.5B (0.5 GB).
#
#   pkg install -y git && git clone https://github.com/RobiiAragon/offline-it-helpdesk
#   cd offline-it-helpdesk && bash scripts/android_termux_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "${PREFIX:-}" == *com.termux* ]] || { echo "This script is meant for Termux."; exit 1; }

MODEL="${MODEL:-qwen2.5-0.5b}"
echo "==> Termux packages (prebuilt numpy, onnxruntime and llama.cpp)"
pkg update -y >/dev/null
pkg install -y python python-numpy python-onnxruntime llama-cpp git >/dev/null

echo "==> install (no pip deps; FastAPI is optional on Termux because pydantic-core needs Rust)"
python -m pip install -q --no-deps -e .

helpdesk download
helpdesk index --providers "${PROVIDERS:-xnnpack,cpu}"

LLM=llamacpp
if command -v llama-server >/dev/null; then
  echo "==> starting llama-server with $MODEL (CPU)"
  MODEL="$MODEL" LLAMA_ARGS="-t ${THREADS:-4}" bash scripts/start_llm.sh > .llama-server.log 2>&1 &
  SERVER=$!
  trap 'kill "$SERVER" 2>/dev/null || true' EXIT
  for _ in $(seq 1 600); do curl -sf http://127.0.0.1:8080/v1/models >/dev/null && break; sleep 1; done
else
  echo "llama-server not found; falling back to the extractive (no-LLM) mode"
  LLM=extractive
fi
helpdesk doctor --llm "$LLM"

MODEL_TAG=$(getprop ro.product.model | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
OUT="eval/results/android-$MODEL_TAG-$MODEL.json"
echo "==> evaluation -> $OUT"
helpdesk eval --llm "$LLM" --providers "${PROVIDERS:-xnnpack,cpu}" --out "$OUT"
helpdesk ask --llm "$LLM" "My account is locked, what do I do?"
echo "done. Share $OUT (termux-setup-storage && cp $OUT ~/storage/downloads/)"
