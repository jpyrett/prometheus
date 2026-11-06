#!/usr/bin/env bash
# Prometheus recommendations end to end: ask what to look at, then find the companies, fetch their data, score and RANK them best to worst, and print a report.
# Unit/integration pytest still lives under tests/. This is the sibling of ./test_prometheus.sh (which runs ONE work order through the desk).
#
#   ./test_recommendations.sh                              prompts for what to look at, how many to return, and whether to fetch
#   ./test_recommendations.sh "Pizza"                      a SEARCH PHRASE: the Data Clerk finds the pizza companies, the Advisor ranks them (the top 5)
#   ./test_recommendations.sh "Pizza" --find 3 --yes       the top 3, without being asked to confirm the paid fetch
#   ./test_recommendations.sh DPZ                          ONE SYMBOL
#   ./test_recommendations.sh "DPZ PZZA YUM"               a LIST of symbols (spaces or commas), ranked best to worst
#   ./test_recommendations.sh "Domino's Pizza"             a COMPANY NAME: looked up to its ticker (the exact name of a listed company)
#   ./test_recommendations.sh --phrase "weight loss drugs" --no-fetch      force the kind; use only the data already stored
#   ./test_recommendations.sh 'MCD, YUM, "Burger King"'    COMPARE a list: tickers, company names or brands, separated by commas (a brand's listed owner is found: Burger King is QSR)
#   ./test_recommendations.sh "Pizza" --cash 1000          size the positions for $1,000 of cash instead of the sample portfolio
#   ./test_recommendations.sh "Pizza" --model advisor      use the Advisor model of .env (ADVISOR_MODEL) instead of gpt-oss-120b
#
# WHAT THE REQUEST CAN BE (it is worked out for you; --symbols, --company and --phrase force it)
#   symbols   every word is a ticker in capitals: DPZ, "DPZ PZZA YUM", "DPZ,PZZA". At most 15.
#   list      items separated by commas (or "and"), with at least one ticker and one name or brand: 'MCD, YUM, "Burger King"'. A ticker is itself; any other item goes to the Data Clerk, which names
#             the listed owner of the brand (Burger King -> Restaurant Brands International, QSR) and code confirms the ticker in Finnhub. The companies are compared and ranked as an industry.
#             --list forces it (without a ticker in the list too).
#   company   text that is EXACTLY a listed company's name (the same words as Finnhub's listing; Inc, Corp, Co and "the" ignored): "Domino's Pizza" is DPZ. A name that is not exact is a phrase.
#   phrase    anything else: "Pizza", "weight loss drugs", "AI memory shortage". The Data Clerk's MODEL proposes the leading brands and the companies that own them (Pizza Hut -> Yum! Brands);
#             CODE looks each one up in Finnhub's directory and uses what the lookup says. A privately held company (Little Caesars, Donatos) has no stock and no filings: it is listed as
#             NOT SCORED. Only listed companies are ranked. The Advisor ranks them as an INDUSTRY (leaders included: a large leader is not left out), and ChromaDB shows how well each company's stored
#             text matches the phrase (a relevance column; it never changes the ranking).
#
# WHAT IT COSTS. Finding the companies of a phrase calls the Data Clerk's model (OpenRouter) and Finnhub's symbol search. Scoring calls Finnhub's metrics and the Advisor's model twice (a
# plan call and a thesis call). FETCHING (the default unless you answer n or pass --no-fetch) refreshes each company's daily bars (Massive), news, press releases and SEC filings (Finnhub and
# Massive; the 10-Q text from sec.gov) and WRITES news/, data/press_releases/, data/filings/ and data/historical/ticker_bars_v1/. You are asked first; --yes skips the question. If there is nobody
# to ask (the input ended) and --yes was not given, the answer is NO: a script that is not being watched must not spend the paid APIs. Without a fetch the stored data is used, and a company with no
# stored bars fails the Advisor's freshness and liquidity gates (the report says so).
#
# THE MODEL. The Data Clerk uses DEFAULT_MODEL of .env (openai/gpt-oss-120b). The Advisor uses openai/gpt-oss-120b here too, on OpenRouter's fast hosts (Cerebras, then Groq; the others as fallback) at
# reasoning effort low, the settings of the Account Manager (entries 46 and 50): --model ID picks another OpenRouter model, --model advisor the ADVISOR_MODEL of .env, --effort and --providers change the settings.
#
# FLAGS  --symbols S  --company NAME  --phrase TEXT  --list "A, B, C"  --find N (1 to 20, default 5)  --case funds|industry|theme  --sector XLY  --portfolio FILE  --cash N  --from-days-ago N (1 to 90, default 14)
#        --today YYYY-MM-DD  --yes  --no-fetch  --no-ledger (explore: ignore and do not write the real ledger, whose cooldown hides a company already recommended)  --model ID|advisor (the Advisor's model; default openai/gpt-oss-120b)  --effort low|medium|high  --providers a,b|none  --save FILE (the finished work order, JSON)  --help
# PROMETHEUS_DRY_RUN=1 prints the command instead of running it (used by tests/test_recommendations_script.py).
set -euo pipefail

cd "$(dirname "$0")"

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

WHAT=""; KIND=""; KIND_TEXT=""; FIND=""; CASE=""; SECTOR=""; PORTFOLIO=""; CASH=""; DAYS=""; TODAY=""; YES=0; NO_FETCH=0; NO_LEDGER=0; MODEL="openai/gpt-oss-120b"; EFFORT=""; PROVIDERS=""; SAVE=""
HAVE_FIND=0

need() { [[ $2 -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbols|--symbol) need "$1" $#; KIND="symbols"; KIND_TEXT="$2"; shift 2 ;;
    --company)          need "$1" $#; KIND="company"; KIND_TEXT="$2"; shift 2 ;;
    --phrase|--term)    need "$1" $#; KIND="phrase";  KIND_TEXT="$2"; shift 2 ;;
    --list)             need "$1" $#; KIND="list";    KIND_TEXT="$2"; shift 2 ;;
    --find)             need "$1" $#; FIND="$2"; HAVE_FIND=1; shift 2 ;;
    --case)             need "$1" $#; CASE="$2"; shift 2 ;;
    --sector)           need "$1" $#; SECTOR="$2"; shift 2 ;;
    --portfolio)        need "$1" $#; PORTFOLIO="$2"; shift 2 ;;
    --cash)             need "$1" $#; CASH="$2"; shift 2 ;;
    --from-days-ago)    need "$1" $#; DAYS="$2"; shift 2 ;;
    --today)            need "$1" $#; TODAY="$2"; shift 2 ;;
    --model)            need "$1" $#; MODEL="$2"; shift 2 ;;
    --effort)           need "$1" $#; EFFORT="$2"; shift 2 ;;
    --providers)        need "$1" $#; PROVIDERS="$2"; shift 2 ;;
    --save)             need "$1" $#; SAVE="$2"; shift 2 ;;
    --yes|-y)           YES=1; shift ;;
    --no-fetch)         NO_FETCH=1; shift ;;
    --no-ledger)        NO_LEDGER=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    --*)                echo "ERROR: unknown option: $1 (try --help)" >&2; exit 2 ;;
    *)                  [[ -z "$WHAT" ]] || { echo "ERROR: give the request once, in quotes (got '$WHAT' and '$1')" >&2; exit 2; }; WHAT="$1"; shift ;;
  esac
done
if [[ -n "$KIND" && -n "$WHAT" ]]; then echo "ERROR: give the request once: the text, or one of --symbols, --company, --phrase, --list" >&2; exit 2; fi
[[ -n "$KIND" ]] && WHAT="$KIND_TEXT"

# Prefer active env; otherwise use conda env "cmu"
if [[ -z "${PROMETHEUS_DRY_RUN:-}" ]]; then
  if ! command -v python >/dev/null 2>&1 || ! python -c "import prometheus" >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
      # shellcheck disable=SC1091
      eval "$(conda shell.bash hook)"
      conda activate cmu
    fi
  fi
fi

# ask VARIABLE PROMPT DEFAULT: the prompt on stderr, one line into VARIABLE (the default if blank). No subshell, so INPUT_ENDED can be set when the input has run out (a pipe that ended).
# When stdin is not a terminal the answer is echoed, so that a transcript reads properly.
INPUT_ENDED=0
ask() {
  local __answer=""
  printf '%s' "$2" >&2
  if ! IFS= read -r __answer; then [[ -n "$__answer" ]] || INPUT_ENDED=1; fi
  [[ -t 0 ]] || printf '%s\n' "$__answer" >&2
  printf -v "$1" '%s' "${__answer:-$3}"
}

# Letters, digits, spaces and & . , ' " - $ only (a symbol, a list, a company name or a phrase: nothing that is a path, a url or a command).
REQUEST_RE='^[A-Za-z0-9$][A-Za-z0-9 $&.,’"'"'"'-]{0,119}$'

echo "== Prometheus recommendations (press Enter to accept a default) =="
while [[ -z "${WHAT// /}" ]]; do
  ask WHAT 'What to look at  (a symbol such as DPZ, a list such as "DPZ PZZA YUM", a company name such as Domino'"'"'s Pizza, or a search phrase such as Pizza): ' ''
  if [[ -z "${WHAT// /}" ]]; then
    echo "  Give a symbol, a list of symbols, a company name or a search phrase." >&2
    [[ $INPUT_ENDED -eq 1 ]] && exit 2
  fi
done
if ! [[ "$WHAT" =~ $REQUEST_RE ]]; then
  echo "  '$WHAT' is not a valid request (letters, digits, spaces and & . , quotes - \$ only, at most 120 characters)." >&2
  exit 2
fi

while true; do
  [[ $HAVE_FIND -eq 0 ]] && ask FIND 'How many top recommendations  [5] (1 to 20): ' '5'
  FIND="${FIND:-5}"
  if [[ "$FIND" =~ ^[0-9]+$ ]] && (( FIND >= 1 && FIND <= 20 )); then break; fi
  echo "  '$FIND' is not a whole number from 1 to 20." >&2
  if [[ $HAVE_FIND -eq 1 || $INPUT_ENDED -eq 1 ]]; then exit 2; fi
done
if [[ -n "$CASE" && "$CASE" != "funds" && "$CASE" != "industry" && "$CASE" != "theme" ]]; then echo "ERROR: --case is funds, industry or theme" >&2; exit 2; fi
if [[ -n "$DAYS" ]] && ! { [[ "$DAYS" =~ ^[0-9]+$ ]] && (( DAYS >= 1 && DAYS <= 90 )); }; then echo "ERROR: --from-days-ago is a whole number from 1 to 90" >&2; exit 2; fi
if [[ -n "$SECTOR" && ! "$SECTOR" =~ ^[A-Za-z]{2,5}$ ]]; then echo "ERROR: --sector is a fund ticker such as XLY" >&2; exit 2; fi
if [[ -n "$EFFORT" && "$EFFORT" != "low" && "$EFFORT" != "medium" && "$EFFORT" != "high" ]]; then echo "ERROR: --effort is low, medium or high" >&2; exit 2; fi
if [[ -n "$PROVIDERS" && ! "$PROVIDERS" =~ ^[A-Za-z0-9,._-]+$ ]]; then echo "ERROR: --providers is a comma separated list of provider names, or none" >&2; exit 2; fi
if [[ ! "$MODEL" =~ ^[A-Za-z0-9._:/-]+$ ]]; then echo "ERROR: --model is an OpenRouter model id such as openai/gpt-oss-120b, or advisor" >&2; exit 2; fi
if [[ -n "$CASH" && -n "$PORTFOLIO" ]]; then echo "ERROR: give --cash or --portfolio, not both" >&2; exit 2; fi
if [[ -n "$CASH" ]] && ! [[ "$CASH" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "ERROR: --cash is a number of dollars" >&2; exit 2; fi

# Fetching uses the paid APIs and writes files: say so, and ask. Nobody to ask (the input ended) and no --yes: NO.
FETCH=0
FETCH_NOTE="no fetch (the stored data is used)"
if [[ $NO_FETCH -eq 0 ]]; then
  WRITES="news/, data/press_releases/, data/filings/ and data/historical/ticker_bars_v1/"
  if [[ $YES -eq 1 ]]; then
    FETCH=1
  else
    ask CONFIRM "Refresh the data (daily bars, news, press releases, SEC filings) of the companies with the PAID APIs and write $WRITES? [Y/n]: " 'Y'
    if [[ $INPUT_ENDED -eq 1 ]]; then
      echo "  (no answer: not fetching. Pass --yes to fetch without being asked.)" >&2
    elif [[ ! "$CONFIRM" =~ ^[Nn] ]]; then
      FETCH=1
    fi
  fi
  [[ $FETCH -eq 1 ]] && FETCH_NOTE="FETCH the companies' data (paid APIs; writes $WRITES)"
fi

echo ""
echo "== Recommendations: find the companies -> refresh their data -> the Data Clerk's order -> the Advisor ranks them =="
echo "   request: $WHAT${KIND:+ (forced as $KIND)} | top $FIND${CASE:+ | case $CASE}${SECTOR:+ | sector $SECTOR} | Advisor model: $([[ "$MODEL" == advisor ]] && echo "ADVISOR_MODEL of .env" || echo "$MODEL") | $FETCH_NOTE"

CMD=(python -m prometheus.application.recommendations)
case "$KIND" in
  symbols) CMD+=(--symbols "$WHAT") ;;
  company) CMD+=(--company "$WHAT") ;;
  phrase)  CMD+=(--phrase "$WHAT") ;;
  list)    CMD+=(--list "$WHAT") ;;
  *)       CMD+=("$WHAT") ;;
esac
CMD+=(--find "$FIND")
[[ -n "$CASE" ]] && CMD+=(--case "$CASE")
[[ -n "$SECTOR" ]] && CMD+=(--sector "${SECTOR^^}")
[[ -n "$PORTFOLIO" ]] && CMD+=(--portfolio "$PORTFOLIO")
[[ -n "$CASH" ]] && CMD+=(--cash "$CASH")
[[ -n "$DAYS" ]] && CMD+=(--from-days-ago "$DAYS")
[[ -n "$TODAY" ]] && CMD+=(--today "$TODAY")
[[ "$MODEL" != "advisor" ]] && CMD+=(--model "$MODEL")
[[ -n "$EFFORT" ]] && CMD+=(--effort "$EFFORT")
[[ -n "$PROVIDERS" ]] && CMD+=(--providers "$PROVIDERS")
[[ -n "$SAVE" ]] && CMD+=(--save "$SAVE")
[[ $NO_LEDGER -eq 1 ]] && CMD+=(--no-ledger)
[[ $FETCH -eq 1 ]] && CMD+=(--fetch)

if [[ -n "${PROMETHEUS_DRY_RUN:-}" ]]; then
  printf 'DRY RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'
  exit 0
fi

STARTED=$SECONDS
status=0
"${CMD[@]}" || status=$?
echo ""
if [[ $status -eq 0 ]]; then
  echo "OK: Prometheus recommendations finished in $((SECONDS - STARTED)) s"
else
  echo "FAILED: the run stopped with exit code $status after $((SECONDS - STARTED)) s (see the report above)" >&2
fi
exit $status
