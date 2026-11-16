#!/usr/bin/env bash
# The Prometheus web UI (design/ui_v1_spec.md, design choices entry 99): the page in ui/ (React) and its bridge in prometheus/application/web/ (FastAPI).
#
#   ./run_ui.sh                      build the UI if ui/dist is missing or older than ui/src, then start the server on http://127.0.0.1:8000
#   ./run_ui.sh --replay             the same, replaying tests/fixtures/session_real/bob_jones_oil.json (FREE: no keys, no network, no database)
#   ./run_ui.sh --replay FILE        another capture (a file made with `session --save`)
#   ./run_ui.sh --dev                the API on 8000 and Vite on 5173 with hot reload (add --replay to replay)
#   ./run_ui.sh --port 8100          another port for the server (the default is 8000)
#   ./run_ui.sh --replay-speed 0     no pause between the stages of a replay (the default 1.0 is about 10 seconds for a whole run)
#
# Starting the server costs nothing. LIVE MODE (no --replay): pressing Go! is a PAID run (OpenRouter and Finnhub) that writes the UI's own account in the local Postgres, AND the page's "Refresh market data first
# (paid)" box is TICKED BY DEFAULT: Go! then also refreshes the market data with the paid data APIs and writes the data files (untick the box to use the stored data). The API itself (POST /api/runs) fetches only when it is
# sent `fetch: true`. The Trader is simulated either way: nothing is sent to a broker. Needs the cmu conda env (Python) and `corepack` (npm 10.9.2) for the page.
set -euo pipefail
cd "$(dirname "$0")"
PY="${HOME}/miniconda3/envs/cmu/bin/python"
NPM=(corepack npm@10.9.2)
DEV=0
SERVER_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) DEV=1; shift ;;
    --replay)
      if [[ $# -gt 1 && "$2" != --* ]]; then SERVER_ARGS+=(--replay "$2"); shift 2; else SERVER_ARGS+=(--replay); shift; fi ;;
    --port|--replay-speed|--question-timeout|--host) SERVER_ARGS+=("$1" "$2"); shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see ./run_ui.sh --help)" >&2; exit 2 ;;
  esac
done
if [[ ! -x "$PY" ]]; then echo "ERROR: $PY not found (the cmu conda env)" >&2; exit 2; fi
if [[ " ${SERVER_ARGS[*]:-} " != *" --replay"* ]]; then
  echo "LIVE MODE: pressing Go! is a PAID run (OpenRouter, Finnhub) that writes the UI's own account in Postgres."
  echo "           'Refresh market data first (paid)' is TICKED by default: Go! also calls the paid data APIs and writes the data files. Untick it to use the stored data. --replay is the free recorded session."
fi
if [[ ! -d ui/node_modules ]]; then
  echo "installing the page's packages (corepack npm@10.9.2 ci) ..."
  (cd ui && "${NPM[@]}" ci)
fi
if [[ $DEV -eq 1 ]]; then
  "$PY" -m prometheus.application.web --dev "${SERVER_ARGS[@]}" &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null || true' EXIT
  echo "Vite on http://localhost:5173 (hot reload), API on http://127.0.0.1:8000"
  cd ui && exec "${NPM[@]}" run dev
fi
if [[ ! -f ui/dist/index.html || -n "$(find ui/src ui/index.html ui/package.json -newer ui/dist/index.html -print -quit 2>/dev/null)" ]]; then
  echo "building the page (ui/dist) ..."
  (cd ui && "${NPM[@]}" run build)
fi
exec "$PY" -m prometheus.application.web "${SERVER_ARGS[@]}"
