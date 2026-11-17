#!/usr/bin/env bash
# macOS end-to-end check: venv, lint, tests, embeddings, index, local LLM, full evaluation.
#   scripts/mac_check.sh                 # uses llama.cpp + Qwen2.5-1.5B (downloads 1.1 GB once)
#   LLM=extractive scripts/mac_check.sh  # no LLM download
set -euo pipefail
cd "$(dirname "$0")/.."
LLM="${LLM:-llamacpp}"
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in python3.13 python3.12 python3.11; do command -v "$c" >/dev/null && PY="$c" && break; done
fi
[[ -n "$PY" ]] || { echo "Python 3.11+ not found (brew install python@3.12)"; exit 1; }

[[ -d .venv ]] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip --version >/dev/null 2>&1 || python -m ensurepip >/dev/null
python -m pip install -q --upgrade pip
python -m pip install -q -e ".[dev]"

echo "==> lint + tests"
ruff check . && ruff format --check . && pytest -q

echo "==> embeddings + index"
helpdesk download
helpdesk index

STARTED=""
if [[ "$LLM" == "llamacpp" ]] && ! curl -sf http://127.0.0.1:8080/v1/models >/dev/null; then
  command -v llama-server >/dev/null || { echo "brew install llama.cpp (or LLM=extractive)"; exit 1; }
  echo "==> starting llama-server (Qwen2.5-1.5B-Instruct Q4_K_M)"
  scripts/start_llm.sh > .llama-server.log 2>&1 &
  STARTED=$!
  trap '[[ -n "$STARTED" ]] && kill "$STARTED" 2>/dev/null || true' EXIT
  for _ in $(seq 1 600); do curl -sf http://127.0.0.1:8080/v1/models >/dev/null && break; sleep 1; done
fi
helpdesk doctor --llm "$LLM"

echo "==> evaluation ($LLM)"
helpdesk eval --llm "$LLM" --out "eval/results/local-mac-$LLM.json"

echo "==> API smoke test"
helpdesk serve --llm "$LLM" --port 8765 > /dev/null 2>&1 &
API=$!
for _ in $(seq 1 40); do curl -sf http://127.0.0.1:8765/health >/dev/null && break; sleep 0.5; done
curl -sf http://127.0.0.1:8765/health && echo
curl -sf -X POST http://127.0.0.1:8765/ask -H 'Content-Type: application/json' \
  -d '{"question":"How long are print jobs kept?"}' | python -c \
  "import json,sys; a=json.load(sys.stdin); print(a['answer']); assert not a['refused']"
kill "$API"
echo "OK - run 'helpdesk serve' and open http://127.0.0.1:8000"
