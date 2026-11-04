#!/usr/bin/env bash
# Prometheus end to end: ask for the request, then run it through the desk (a LangGraph that passes ONE work order from the
# Account Manager to the Data Clerk to the Advisor). Unit/integration pytest still lives under tests/.
#
#   ./test_prometheus.sh                    prompts for the four parameters below
#   ./test_prometheus.sh --symbol MSFT --from-days-ago 14 --yes          no prompts for what is given
#   ./test_prometheus.sh --symbol "Procter&Gamble" --yes                  a company name: looked up to its ticker (PG), then as above
#   ./test_prometheus.sh --term "flying car"                              a term with no symbol: searches the stored news, fetches nothing
#   ./test_prometheus.sh --source press_releases --symbol AAPL --yes      the company's OWN press releases (design choices entry 34): fetches a year of them
#   ./test_prometheus.sh --source press_releases --company "Apple Inc" --yes   the same, by company name (exact; the symbol wins if both are given)
#   ./test_prometheus.sh --source filings --symbol AAPL --yes             the company's SEC filings (design choices entry 35): fetches a year of them (paid Massive/Polygon API)
#   ./test_prometheus.sh --source filings --company "Apple Inc" --yes     the same, by company name (the same lookup rules as press releases)
#   ./test_prometheus.sh --legacy           the old demo: LocalLLM + enter_symbol elicitation (main.py)
#
# THE FOUR PARAMETERS (and a fifth, company, for press releases and filings only)
#   symbol         a ticker, for example MSFT, or a company name, for example "Procter & Gamble". A ticker in capitals is used as it is; a name
#                  (or a ticker in lower case) is LOOKED UP with Finnhub's symbol search, never guessed, and an ambiguous name is refused (type
#                  the ticker). A symbol makes the request LIVE: fresh news is fetched with the PAID Finnhub/Polygon APIs and written to
#                  news/<TICKER>.txt (you are asked to confirm, or pass --yes; --no-fetch searches the stored news instead).
#   term           a topic, for example "flying car". With no symbol it is searched in the stored news. With a symbol it narrows the
#                  question inside that symbol's news.
#   source         what is searched: news (the default), press_releases or filings. A press release is the company's own announcement, fetched from Finnhub's
#                  /press-releases and kept in data/press_releases/<TICKER>_PRs.txt, apart from the news. A filing is a government (SEC EDGAR) filing, fetched from
#                  the Massive (Polygon) filings API and kept in data/filings/<TICKER>.txt: only 10-K and 8-K come with text, a 10-Q or a Form 144 is known by its
#                  form, date and link only. Nothing else is wired.
#   from_days_ago  how many days back (the from_date of the fetch and the window of the search). Default 7, from 1 to 90 for news;
#                  default 365, from 1 to 730 for press_releases and filings.
#   company        press_releases and filings only: a company name, for example "Apple Inc". The ticker comes first: it is looked up, and if it is found it is
#                  used (even if a company name is also given); if not, the company name is looked up in Finnhub's directory. The name is matched
#                  on its words, never on its spelling: case and punctuation do not matter ("Apple Inc" is "Apple Inc."), and neither do the generic
#                  company words Inc, Corp, Co, Company, The and "and" ("Pepsico" is "PepsiCo Inc", "Apple" is "Apple Inc."). A typo is NOT matched
#                  ("Proctor & Gamble", "Pepsi" for Pepsico). A name that fits several listings is refused, not guessed. Nothing found is
#                  refused: "Symbol: X is not found" or "Company Name not found." --company alone implies --source press_releases (give --source filings for filings).
#                  At the prompts a press release or filings request needs a ticker at the Symbol prompt; to ask by company name alone, pass --company.
# News: at least one of symbol and term is required. Press releases and filings: at least one of symbol and company.
#
# FLAGS  --symbol S  --term T  --source news|press_releases|filings  --company NAME  --from-days-ago N  --yes (do not ask to confirm a fetch)  --no-fetch  --no-advisor
#        --model ID (Advisor model)  --skip-inspect (skip the fastmcp smoke check)  --legacy  --help
# PROMETHEUS_DRY_RUN=1 prints the command instead of running it (used by tests/test_prometheus_script.py).
set -euo pipefail

cd "$(dirname "$0")"

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

SYMBOL=""; TERM_TEXT=""; SOURCE=""; DAYS=""; COMPANY=""; YES=0; NO_FETCH=0; NO_ADVISOR=0; MODEL=""; SKIP_INSPECT=0; LEGACY=0
HAVE_SYMBOL=0; HAVE_TERM=0; HAVE_SOURCE=0; HAVE_DAYS=0; HAVE_COMPANY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbol)         [[ $# -ge 2 ]] || { echo "ERROR: --symbol needs a value" >&2; exit 2; }; SYMBOL="$2"; HAVE_SYMBOL=1; shift 2 ;;
    --term)           [[ $# -ge 2 ]] || { echo "ERROR: --term needs a value" >&2; exit 2; }; TERM_TEXT="$2"; HAVE_TERM=1; shift 2 ;;
    --source)         [[ $# -ge 2 ]] || { echo "ERROR: --source needs a value" >&2; exit 2; }; SOURCE="$2"; HAVE_SOURCE=1; shift 2 ;;
    --company)        [[ $# -ge 2 ]] || { echo "ERROR: --company needs a value" >&2; exit 2; }; COMPANY="$2"; HAVE_COMPANY=1; shift 2 ;;
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

# --company belongs to press releases. Alone it means that source; with another source it is a mistake, and it is refused, not silently ignored.
if [[ -n "${COMPANY// /}" ]]; then
  if [[ $HAVE_SOURCE -eq 0 ]]; then
    SOURCE="press_releases"; HAVE_SOURCE=1
  elif [[ "$SOURCE" != "press_releases" && "$SOURCE" != "filings" ]]; then
    echo "ERROR: --company is only used with --source press_releases or filings (not '$SOURCE')" >&2; exit 2
  fi
fi

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

# A ticker or a company name: letters, digits, spaces and & . , ' - only (the desk looks a name up, and refuses what it cannot resolve).
NAME_RE='^\$?[A-Za-z0-9][A-Za-z0-9 &.,’'"'"'-]{0,59}$'
valid_symbol() { [[ -z "$1" || "$1" =~ $NAME_RE ]]; }
# A ticker in capitals is used as it is. Anything else is looked up first, which the confirmation says.
plain_ticker() { [[ "${1#\$}" =~ ^[A-Z0-9][A-Z0-9.-]{0,9}$ ]]; }

echo "== Prometheus request (press Enter to accept a default) =="
while true; do
  [[ $HAVE_SYMBOL -eq 0 ]] && ask SYMBOL 'Symbol  (a ticker such as MSFT, or a company name such as Procter & Gamble; blank for none): ' ''
  if ! valid_symbol "$SYMBOL"; then
    echo "  '$SYMBOL' is not a valid ticker symbol or company name." >&2
    if [[ $HAVE_SYMBOL -eq 1 || $INPUT_ENDED -eq 1 ]]; then exit 2; fi
    continue
  fi
  [[ $HAVE_TERM -eq 0 ]] && ask TERM_TEXT 'Term    (a topic such as flying car; blank for none): ' ''
  if [[ -n "${SYMBOL// /}" || -n "${TERM_TEXT// /}" || ( $HAVE_COMPANY -eq 1 && -n "${COMPANY// /}" ) ]]; then break; fi   # a --company flag is a request on its own (press releases)
  echo "  Give a symbol, a term, or both." >&2
  if [[ $HAVE_SYMBOL -eq 1 || $HAVE_TERM -eq 1 || $INPUT_ENDED -eq 1 ]]; then exit 2; fi   # nothing more can be asked: do not loop forever
done
SYMBOL="${SYMBOL#\$}"   # the case is kept: the desk uses a ticker in capitals as it is and looks up anything else
if [[ -z "$SYMBOL" ]]; then WHAT=""; elif plain_ticker "$SYMBOL"; then WHAT="$SYMBOL"; else WHAT="'$SYMBOL' (its ticker is looked up first)"; fi
FILE="news/${SYMBOL}.txt"; plain_ticker "$SYMBOL" || FILE="news/<its ticker>.txt"

while true; do
  [[ $HAVE_SOURCE -eq 0 ]] && ask SOURCE 'Source  [news] (news; press_releases for the company'"'"'s own announcements; filings for its SEC filings): ' 'news'
  SOURCE="${SOURCE:-news}"
  [[ "$SOURCE" == "news" || "$SOURCE" == "press_releases" || "$SOURCE" == "filings" ]] && break
  echo "  Source '$SOURCE' is not wired yet (only: news, press_releases, filings)." >&2
  if [[ $HAVE_SOURCE -eq 1 || $INPUT_ENDED -eq 1 ]]; then exit 2; fi
done

# Press releases (design choices entry 34) and filings (entry 35) take a company name as well. It is asked only for these sources, so the news prompts are exactly
# as they were. The ticker is looked up first; the company name only if the ticker is not found. A request needs one of them.
NOUN="news"; APIS="Finnhub/Polygon APIs"; DAYS_DEFAULT=7; DAYS_MAX=90
if [[ "$SOURCE" == "press_releases" || "$SOURCE" == "filings" ]]; then
  DAYS_DEFAULT=365; DAYS_MAX=730
  if [[ "$SOURCE" == "press_releases" ]]; then
    NOUN="press releases"; APIS="Finnhub API"; REQUEST_NAME="press release"; FOLDER="data/press_releases"; SUFFIX="_PRs.txt"
  else
    NOUN="filings"; APIS="Massive (Polygon) API"; REQUEST_NAME="filings"; FOLDER="data/filings"; SUFFIX=".txt"
  fi
  while true; do
    [[ $HAVE_COMPANY -eq 0 ]] && ask COMPANY "Company name  ($SOURCE only, used when the symbol is not given or not found; blank for none): " ''
    if valid_symbol "$COMPANY"; then break; fi
    echo "  '$COMPANY' is not a valid company name." >&2
    if [[ $HAVE_COMPANY -eq 1 || $INPUT_ENDED -eq 1 ]]; then exit 2; fi
  done
  COMPANY="${COMPANY#\$}"
  if [[ -z "${SYMBOL// /}" && -z "${COMPANY// /}" ]]; then
    echo "  A $REQUEST_NAME request needs a symbol, a company name, or both." >&2
    exit 2
  fi
  if [[ -n "$SYMBOL" && -n "$COMPANY" ]]; then WHAT="$SYMBOL (if it is not found: '$COMPANY')"
  elif [[ -n "$SYMBOL" ]]; then WHAT="$SYMBOL"
  else WHAT="'$COMPANY' (its ticker is looked up first)"; fi
  FILE="$FOLDER/<its ticker>$SUFFIX"
  [[ "$SYMBOL" =~ ^[A-Za-z0-9][A-Za-z0-9.-]{0,9}$ ]] && FILE="$FOLDER/${SYMBOL^^}$SUFFIX"
else
  COMPANY=""   # only press releases and filings use it
fi

while true; do
  [[ $HAVE_DAYS -eq 0 ]] && ask DAYS "From how many days ago  [$DAYS_DEFAULT] (1 to $DAYS_MAX): " "$DAYS_DEFAULT"
  DAYS="${DAYS:-$DAYS_DEFAULT}"
  if [[ "$DAYS" =~ ^[0-9]+$ ]] && (( DAYS >= 1 && DAYS <= DAYS_MAX )); then break; fi
  echo "  '$DAYS' is not a whole number of days from 1 to $DAYS_MAX." >&2
  if [[ $HAVE_DAYS -eq 1 || $INPUT_ENDED -eq 1 ]]; then exit 2; fi
done

# A symbol means fresh news from the paid APIs and a file written: say so, and ask. If there is nobody to ask (the input ended) and --yes was
# not given, the answer is NO: a script that is not being watched must not spend the paid APIs by default.
FETCH_NOTE="no fetch (the stored $NOUN are searched)"
[[ "$NOUN" == "news" ]] && FETCH_NOTE="no fetch (the stored news is searched)"
[[ "$SOURCE" == "press_releases" || "$SOURCE" == "filings" ]] && FETCH_NOTE="no fetch (the stored $NOUN are searched; the symbol or name is still looked up in Finnhub)"
if [[ ( -n "$SYMBOL" || -n "$COMPANY" ) && $NO_FETCH -eq 0 ]]; then
  if [[ $YES -eq 1 ]]; then
    FETCH_NOTE="FETCH fresh $NOUN for $WHAT (paid APIs; writes $FILE)"
  else
    ask CONFIRM "Fetch fresh $NOUN for $WHAT with the PAID $APIS and write $FILE? [Y/n]: " 'Y'
    if [[ $INPUT_ENDED -eq 1 ]]; then
      echo "  (no answer: not fetching. Pass --yes to fetch without being asked.)" >&2
      NO_FETCH=1
    elif [[ "$CONFIRM" =~ ^[Nn] ]]; then
      NO_FETCH=1
    else
      FETCH_NOTE="FETCH fresh $NOUN for $WHAT (paid APIs; writes $FILE)"
    fi
  fi
fi

echo ""
echo "== The desk (LangGraph, one work order: account_manager -> data_clerk -> advisor) =="
COMPANY_NOTE=""; [[ -n "$COMPANY" ]] && COMPANY_NOTE=" | company: $COMPANY"
echo "   symbol: ${SYMBOL:-none} | term: ${TERM_TEXT:-none} | source: $SOURCE | from_days_ago: $DAYS${COMPANY_NOTE} | $FETCH_NOTE"

CMD=(python -m prometheus.langgraph.desk --symbol "$SYMBOL" --term "$TERM_TEXT" --source "$SOURCE" --from-days-ago "$DAYS")
[[ -n "$COMPANY" ]] && CMD+=(--company "$COMPANY")
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
