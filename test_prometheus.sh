#!/usr/bin/env bash
# Quick end-to-end Prometheus runs (no need to edit tests/ first).
# Add more e2e scenarios below as the app grows. Unit/integration pytest
# still lives under tests/.
set -euo pipefail

cd "$(dirname "$0")"

# Prefer active env; otherwise use conda env "cmu"
if ! command -v fastmcp >/dev/null 2>&1 || ! command -v python >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda activate cmu
  fi
fi

echo "== fastmcp inspect (summary) =="
fastmcp inspect prometheus/mcp/server.py

echo ""
echo "== main.py (LocalLLM + enter_symbol elicitation) =="
echo "   At the prompt, enter a symbol (e.g. TEST, AAPL, or \$AAPL)."
python main.py

echo ""
echo "OK: Prometheus e2e smoke finished"
# TODO: add more e2e cases here (e.g. --skip-llm, multi-symbol, Redis-backed quotes)
