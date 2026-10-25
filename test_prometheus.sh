#!/usr/bin/env bash
# Prometheus end to end: ask for the request, then run it through the desk (a LangGraph that passes ONE work order from the
# Account Manager to the Data Clerk to the Advisor). Unit/integration pytest still lives under tests/.
#
#   ./test_prometheus.sh                    prompts for the four parameters below
#   ./test_prometheus.sh --symbol MSFT --from-days-ago 14 --yes          no prompts for what is given
#   ./test_prometheus.sh --term "flying car"                              a term with no symbol: searches the stored news, fetches nothing
#   ./test_prometheus.sh --legacy           the old demo: LocalLLM + enter_symbol elicitation (main.py)
#
# THE FOUR PARAMETERS
#   symbol         a ticker, for example MSFT. A symbol makes the request LIVE: fresh news is fetched with the PAID Finnhub/Polygon APIs and
#                  written to news/<SYMBOL>.txt (you are asked to confirm, or pass --yes; --no-fetch searches the stored news instead).
#   term           a topic, for example "flying car". With no symbol it is searched in the stored news. With a symbol it narrows the
#                  question inside that symbol's news.
#   source         only "news" is wired, so it is the default and the only value accepted.
#   from_days_ago  how many days back (the from_date of the fetch and the window of the search). Default 7, from 1 to 90.
# At least one of symbol and term is required.
#
# FLAGS  --symbol S  --term T  --source news  --from-days-ago N  --yes (do not ask to confirm a fetch)  --no-fetch  --no-advisor
#        --model ID (Advisor model)  --skip-inspect (skip the fastmcp smoke check)  --legacy  --help
# PROMETHEUS_DRY_RUN=1 prints the command instead of running it (used by tests/test_prometheus_script.py).
set -euo pipefail

cd "$(dirname "$0")"

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

SYMBOL=""; TERM_TEXT=""; SOURCE=""; DAYS=""; YES=0; NO_FETCH=0; NO_ADVISOR=0; MODEL=""; SKIP_INSPECT=0; LEGACY=0
HAVE_SYMBOL=0; HAVE_TERM=0; HAVE_SOURCE=0; HAVE_DAYS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbol)         [[ $# -ge 2 ]] || { echo "ERROR: --symbol needs a value" >&2; exit 2; }; SYMBOL="$2"; HAVE_SYMBOL=1; shift 2 ;;
    --term)           [[ $# -ge 2 ]] || { echo "ERROR: --term needs a value" >&2; exit 2; }; TERM_TEXT="$2"; HAVE_TERM=1; shift 2 ;;
    --source)         [[ $# -ge 2 ]] || { echo "ERROR: --source needs a value" >&2; exit 2; }; SOURCE="$2"; HAVE_SOURCE=1; shift 2 ;;
    --from-days-ago)  [[ $# -ge 2 ]] || { echo "ERROR: --from-days-ago needs a value" >&2; exit 2; }; DAYS="$2"; HAVE_DAYS=1; shift 2 ;;
    --model)          [[ $# -ge 2 ]] || { echo "ERROR: --model needs a value" >&2; exit 2; }; MODEL="$2"; shift 2 ;;
    --yes|-y)         YES=1; shift ;;
    --no-fetch)       NO_FETCH=1; shift ;;
    --no-advisor)     NO_ADVISOR=1; shift ;;
    --skip-inspect)   SKIP_INSPECT=1; shift ;;
    --legacy)         LEGACY=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

# Prefer active env; otherwise use conda env "cmu"
if [[ -z "${PROMETHEUS_DRY_RUN:-}" ]]; then
  if ! command -v fastmcp >/dev/null 2>&1 || ! command -v python >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
      # shellcheck disable=SC1091
      eval "$(conda shell.bash hook)"
      conda activate cmu
    fi
  fi
fi

if [[ $SKIP_INSPECT -eq 0 && -z "${PROMETHEUS_DRY_RUN:-}" ]]; then
  echo "== fastmcp inspect (summary) =="
  fastmcp inspect prometheus/mcp/server.py
  echo ""
fi

if [[ $LEGACY -eq 1 ]]; then
  echo "== main.py (LocalLLM + enter_symbol elicitation) =="
  echo "   At the prompt, enter a symbol (e.g. TEST, AAPL, or \$AAPL)."
  if [[ -n "${PROMETHEUS_DRY_RUN:-}" ]]; then echo "DRY RUN: python main.py"; exit 0; fi
  python main.py
  echo ""
  echo "OK: Prometheus legacy smoke finished"
  exit 0
fi

# ---- the four parameters -------------------------------------------------------------------------------------------------------------
# ask VARIABLE PROMPT DEFAULT: prints the prompt on stderr, reads one line into VARIABLE (the default if the line is blank). It does not use a
# subshell, so that it can set INPUT_ENDED when the input has run out (a pipe that ended): the loops below must stop then, not spin. When
# stdin is not a terminal the answer is echoed, so that a transcript reads properly.
INPUT_ENDED=0
ask() {
  local __answer=""
  printf '%s' "$2" >&2
  if ! IFS= read -r __answer; then [[ -n "$__answer" ]] || INPUT_ENDED=1; fi
  [[ -t 0 ]] || printf '%s\n' "$__answer" >&2
  printf -v "$1" '%s' "${__answer:-$3}"
}

valid_symbol() { [[ -z "$1" || "$1" =~ ^\$?[A-Za-z0-9][A-Za-z0-9.-]{0,9}$ ]]; }

echo "== Prometheus request (press Enter to accept a default) =="
while true; do
  [[ $HAVE_SYMBOL -eq 0 ]] && ask SYMBOL 'Symbol  (a ticker such as MSFT; blank for none): ' ''
  if ! valid_symbol "$SYMBOL"; then
    echo "  '$SYMBOL' is not a valid ticker symbol." >&2
    if [[ $HAVE_SYMBOL -eq 1 || $INPUT_ENDED -eq 1 ]]; then exit 2; fi
    continue
  fi
  [[ $HAVE_TERM -eq 0 ]] && ask TERM_TEXT 'Term    (a topic such as flying car; blank for none): ' ''
  if [[ -n "${SYMBOL// /}" || -n "${TERM_TEXT// /}" ]]; then break; fi
  echo "  Give a symbol, a term, or both." >&2
  if [[ $HAVE_SYMBOL -eq 1 || $HAVE_TERM -eq 1 || $INPUT_ENDED -eq 1 ]]; then exit 2; fi   # nothing more can be asked: do not loop forever
done
SYMBOL="${SYMBOL#\$}"; SYMBOL="${SYMBOL^^}"

while true; do
  [[ $HAVE_SOURCE -eq 0 ]] && ask SOURCE 'Source  [news] (only news is wired): ' 'news'
  SOURCE="${SOURCE:-news}"
  [[ "$SOURCE" == "news" ]] && break
  echo "  Source '$SOURCE' is not wired yet (only: news)." >&2
  if [[ $HAVE_SOURCE -eq 1 || $INPUT_ENDED -eq 1 ]]; then exit 2; fi
done

while true; do
  [[ $HAVE_DAYS -eq 0 ]] && ask DAYS 'From how many days ago  [7] (1 to 90): ' '7'
  DAYS="${DAYS:-7}"
  if [[ "$DAYS" =~ ^[0-9]+$ ]] && (( DAYS >= 1 && DAYS <= 90 )); then break; fi
  echo "  '$DAYS' is not a whole number of days from 1 to 90." >&2
  if [[ $HAVE_DAYS -eq 1 || $INPUT_ENDED -eq 1 ]]; then exit 2; fi
done

# A symbol means fresh news from the paid APIs and a file written: say so, and ask. If there is nobody to ask (the input ended) and --yes was
# not given, the answer is NO: a script that is not being watched must not spend the paid APIs by default.
FETCH_NOTE="no fetch (the stored news is searched)"
if [[ -n "$SYMBOL" && $NO_FETCH -eq 0 ]]; then
  if [[ $YES -eq 1 ]]; then
    FETCH_NOTE="FETCH fresh news for $SYMBOL (paid APIs; writes news/$SYMBOL.txt)"
  else
    ask CONFIRM "Fetch fresh news for $SYMBOL with the PAID Finnhub/Polygon APIs and write news/$SYMBOL.txt? [Y/n]: " 'Y'
    if [[ $INPUT_ENDED -eq 1 ]]; then
      echo "  (no answer: not fetching. Pass --yes to fetch without being asked.)" >&2
      NO_FETCH=1
    elif [[ "$CONFIRM" =~ ^[Nn] ]]; then
      NO_FETCH=1
    else
      FETCH_NOTE="FETCH fresh news for $SYMBOL (paid APIs; writes news/$SYMBOL.txt)"
    fi
  fi
fi

echo ""
echo "== The desk (LangGraph, one work order: account_manager -> data_clerk -> advisor) =="
echo "   symbol: ${SYMBOL:-none} | term: ${TERM_TEXT:-none} | source: $SOURCE | from_days_ago: $DAYS | $FETCH_NOTE"

CMD=(python -m prometheus.langgraph.desk --symbol "$SYMBOL" --term "$TERM_TEXT" --source "$SOURCE" --from-days-ago "$DAYS")
[[ $NO_FETCH -eq 1 ]] && CMD+=(--no-fetch)
[[ $NO_ADVISOR -eq 1 ]] && CMD+=(--no-advisor)
[[ -n "$MODEL" ]] && CMD+=(--model "$MODEL")

if [[ -n "${PROMETHEUS_DRY_RUN:-}" ]]; then
  printf 'DRY RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'
  exit 0
fi

STARTED=$SECONDS
status=0
"${CMD[@]}" || status=$?
echo ""
if [[ $status -eq 0 ]]; then
  echo "OK: Prometheus desk finished in $((SECONDS - STARTED)) s"
else
  echo "FAILED: the desk stopped with exit code $status after $((SECONDS - STARTED)) s (see the report above)" >&2
fi
exit $status
