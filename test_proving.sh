#!/usr/bin/env bash
# The Proving Ground (design/proving_ground_spec.md): ./test_proving.sh COMMAND [options]
#   ./test_proving.sh track                       free: every recommendation of the saved runs scored against the stored bars
#   ./test_proving.sh live --account N            Finnhub quotes: the paper account against its plan (sells nothing)
#   ./test_proving.sh coverage | index | show RUN_ID                free
#   ./test_proving.sh time-machine --prompt "..." --cash 10000 --from 2026-10-19 --to 2026-11-13 --cadence weekly      PAID (the models; never a fetch): prints the plan and asks first
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$HOME/miniconda3/envs/cmu/bin/python}"
exec "$PY" -m prometheus.proving "$@"
