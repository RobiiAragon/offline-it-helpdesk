#!/usr/bin/env bash
# Start a local OpenAI-compatible LLM server with llama.cpp (macOS: Metal; Termux: CPU).
#   scripts/start_llm.sh                 # Qwen2.5-1.5B-Instruct Q4_K_M (1.1 GB)
#   MODEL=qwen2.5-0.5b scripts/start_llm.sh   # smaller/faster (0.5 GB), good for tablets
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-qwen2.5-1.5b}"
PORT="${PORT:-8080}"
CTX="${CTX:-4096}"

command -v llama-server >/dev/null || {
  echo "llama-server not found. macOS: brew install llama.cpp | Termux: pkg install llama-cpp"; exit 1; }

GGUF=$(python3 - "$MODEL" <<'PY'
import sys
from pathlib import Path
from offline_helpdesk.config import Settings
from offline_helpdesk.download import LLMS, fetch
print(fetch(LLMS[sys.argv[1]], Path(Settings().models_dir)))
PY
)
GGUF=$(echo "$GGUF" | tail -1)
echo "serving $GGUF on http://127.0.0.1:$PORT/v1"
exec llama-server -m "$GGUF" --host 127.0.0.1 --port "$PORT" -c "$CTX" --temp 0 ${LLAMA_ARGS:-}
