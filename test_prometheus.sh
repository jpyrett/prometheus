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
#   ./test_prometheus.sh --session          A USER SESSION, END TO END (design choices entry 78): prompts for the user (name, age, risk, money, Ask or Auto) and for what they want in their OWN WORDS,
#                                           creates the account in Postgres, adds the money, and starts the LangGraph desk with the request: the Account Manager reads it, the Data Clerk finds the
#                                           companies, the Advisor ranks them, the Critic reviews them against the profile, the Account Manager decides (and spreads the money evenly when asked),
#                                           the Trader executes (simulated) and books the fills. It prints the status of the recommendations, the Trader's results and the portfolio before and after.
#                                           ASK MODE (the default): after the Critic, the Account Manager shows the trades it proposes (final shares and cost) and asks you, Y/N, about each one;
#                                           only the ones you approve go on to the Trader, exactly as shown. --yes does NOT approve trades. AUTO mode trades what the Critic approved above the mark.
#   ./test_prometheus.sh --session --name "Bob Jones" --age 60 --risk conservative --cash 15000 --mode auto --yes --prompt "I want to buy oil stocks. Buy the most amount of oil stocks shares from the 3 most recommended companies. Balance the purchased stocks evenly among the 3 recommended companies."
#   Every agent is REAL (Data Clerk, Advisor, Critic, Trader, Account Manager, the Postgres account book) and the data are the real stored data: it costs OpenRouter and Finnhub calls, and with a fetch (the
#   default unless --no-fetch; you are asked, --yes answers) the PAID Massive/Finnhub APIs, which write files. Nothing is sent to a broker. Exit code 0: the session did what was asked (the portfolio
#   changed); 1: it could not trade (the reasons are printed; --allow-no-trade accepts that); 2: a bad flag, the account book or the request could not be used.
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
# SESSION FLAGS (with --session; anything not given is asked)  --name N  --age N  --risk conservative|moderate|aggressive  --cash DOLLARS  --mode ask|auto  --prompt TEXT  --account N|KEY (an existing account: no
#        user is created)  --yes (ask nothing: not the fetch, not the understanding; in Ask mode you are still asked to approve each trade)  --no-fetch  --no-trader-model  --cleanup (delete the new account at the end)  --allow-no-trade  --real-ledger
#        --auto-min-score N  --max-position-pct X  --max-exposure-pct X  --max-loss-pct X  --default-count N  --model ID (the Advisor's)  --save FILE (what the desk made, as JSON: a real capture)
# PROMETHEUS_DRY_RUN=1 prints the command instead of running it (used by tests/test_prometheus_script.py).
set -euo pipefail

cd "$(dirname "$0")"

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

SYMBOL=""; TERM_TEXT=""; SOURCE=""; DAYS=""; COMPANY=""; YES=0; NO_FETCH=0; NO_ADVISOR=0; MODEL=""; SKIP_INSPECT=0; LEGACY=0
HAVE_SYMBOL=0; HAVE_TERM=0; HAVE_SOURCE=0; HAVE_DAYS=0; HAVE_COMPANY=0
SESSION=0; S_NAME=""; S_AGE=""; S_RISK=""; S_CASH=""; S_MODE=""; S_PROMPT=""; S_ACCOUNT=""; S_CLEANUP=0; S_ALLOW_NO_TRADE=0; S_NO_TRADER_MODEL=0; S_REAL_LEDGER=0
S_AUTO_MIN=""; S_MAX_POS=""; S_MAX_EXP=""; S_MAX_LOSS=""; S_COUNT=""; S_SAVE=""

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
    --session)        SESSION=1; shift ;;
    --name)           [[ $# -ge 2 ]] || { echo "ERROR: --name needs a value" >&2; exit 2; }; S_NAME="$2"; shift 2 ;;
    --age)            [[ $# -ge 2 ]] || { echo "ERROR: --age needs a value" >&2; exit 2; }; S_AGE="$2"; shift 2 ;;
    --risk)           [[ $# -ge 2 ]] || { echo "ERROR: --risk needs a value" >&2; exit 2; }; S_RISK="$2"; shift 2 ;;
    --cash)           [[ $# -ge 2 ]] || { echo "ERROR: --cash needs a value" >&2; exit 2; }; S_CASH="$2"; shift 2 ;;
    --mode)           [[ $# -ge 2 ]] || { echo "ERROR: --mode needs a value" >&2; exit 2; }; S_MODE="$2"; shift 2 ;;
    --prompt)         [[ $# -ge 2 ]] || { echo "ERROR: --prompt needs a value" >&2; exit 2; }; S_PROMPT="$2"; shift 2 ;;
    --account)        [[ $# -ge 2 ]] || { echo "ERROR: --account needs a value" >&2; exit 2; }; S_ACCOUNT="$2"; shift 2 ;;
    --auto-min-score) [[ $# -ge 2 ]] || { echo "ERROR: --auto-min-score needs a value" >&2; exit 2; }; S_AUTO_MIN="$2"; shift 2 ;;
    --max-position-pct) [[ $# -ge 2 ]] || { echo "ERROR: --max-position-pct needs a value" >&2; exit 2; }; S_MAX_POS="$2"; shift 2 ;;
    --max-exposure-pct) [[ $# -ge 2 ]] || { echo "ERROR: --max-exposure-pct needs a value" >&2; exit 2; }; S_MAX_EXP="$2"; shift 2 ;;
    --max-loss-pct)   [[ $# -ge 2 ]] || { echo "ERROR: --max-loss-pct needs a value" >&2; exit 2; }; S_MAX_LOSS="$2"; shift 2 ;;
    --default-count)  [[ $# -ge 2 ]] || { echo "ERROR: --default-count needs a value" >&2; exit 2; }; S_COUNT="$2"; shift 2 ;;
    --save)           [[ $# -ge 2 ]] || { echo "ERROR: --save needs a value" >&2; exit 2; }; S_SAVE="$2"; shift 2 ;;
    --cleanup)        S_CLEANUP=1; shift ;;
    --allow-no-trade) S_ALLOW_NO_TRADE=1; shift ;;
    --no-trader-model) S_NO_TRADER_MODEL=1; shift ;;
    --real-ledger)    S_REAL_LEDGER=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

# The session flags belong to --session. Without it they are refused, not silently ignored.
if [[ $SESSION -eq 0 ]] && [[ -n "$S_NAME$S_AGE$S_RISK$S_CASH$S_MODE$S_PROMPT$S_ACCOUNT$S_AUTO_MIN$S_MAX_POS$S_MAX_EXP$S_MAX_LOSS$S_COUNT$S_SAVE" || $S_CLEANUP -eq 1 || $S_ALLOW_NO_TRADE -eq 1 || $S_NO_TRADER_MODEL -eq 1 || $S_REAL_LEDGER -eq 1 ]]; then
  echo "ERROR: --name, --age, --risk, --cash, --mode, --prompt, --account, --cleanup and the other session flags go with --session" >&2; exit 2
fi
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

# ---- a user session, end to end (design choices entry 78) --------------------------------------------------------------------------------
if [[ $SESSION -eq 1 ]]; then
  INPUT_ENDED=0
  ask() {   # ask VARIABLE PROMPT DEFAULT: prompts on stderr, reads one line (the default when blank); sets INPUT_ENDED when the input has run out
    local __answer=""
    printf '%s' "$2" >&2
    if ! IFS= read -r __answer; then [[ -n "$__answer" ]] || INPUT_ENDED=1; fi
    [[ -t 0 ]] || printf '%s\n' "$__answer" >&2
    printf -v "$1" '%s' "${__answer:-$3}"
  }
  need_value() {   # need_value VARIABLE PROMPT DEFAULT VALIDATOR MESSAGE: asked until valid; a value given as a flag is checked once and refused, never re-asked
    local __var="$1" __given="${!1}"
    while true; do
      if [[ -z "$__given" ]]; then
        ask "$__var" "$2" "$3"
        # A required answer that gets no input at all (the input ended) is NOT the default: a script nobody is typing at must not create a user and start a paid run on answers nobody gave.
        if [[ $INPUT_ENDED -eq 1 ]]; then echo "  (no answer: the input ended, so nothing was started. Give it as a flag, or run the script in a terminal.)" >&2; exit 2; fi
      fi
      if "$4" "${!__var}"; then return 0; fi
      echo "  $5" >&2
      if [[ -n "$__given" || $INPUT_ENDED -eq 1 ]]; then exit 2; fi
    done
  }
  valid_name() { [[ "$1" =~ ^[A-Za-z][A-Za-z\ .\'’-]{0,98}$ ]]; }
  valid_age() { [[ "$1" =~ ^[0-9]{2,3}$ ]] && (( 10#$1 >= 18 && 10#$1 <= 120 )); }
  valid_risk() { [[ "$1" == "conservative" || "$1" == "moderate" || "$1" == "aggressive" ]]; }
  valid_cash() { [[ "$1" =~ ^[0-9]+(\.[0-9]{1,2})?$ ]] && awk -v v="$1" 'BEGIN { exit !(v > 0) }'; }
  valid_mode() { [[ "$1" == "ask" || "$1" == "auto" ]]; }
  valid_prompt() { [[ -n "${1// /}" && ${#1} -le 600 ]]; }

  valid_account() { [[ "$1" =~ ^[0-9]+$ || "$1" =~ ^[a-z][a-z0-9_]{0,40}$ ]]; }
  valid_yes_no() { [[ "$1" =~ ^[YyNn]([Ee][Ss]|[Oo])?$ ]]; }

  echo "== Prometheus: a user session (press Enter to accept a default) =="
  # 1. the user: a new one (asked, field by field) or an existing account. A flag settles it without asking: --account is an existing one, any of the user's fields is a new one.
  CREATE=""
  if [[ -n "$S_ACCOUNT" ]]; then CREATE="n"
  elif [[ -n "$S_NAME$S_AGE$S_RISK$S_CASH$S_MODE" ]]; then CREATE="y"
  else
    need_value CREATE 'Create a new user?  [Y/n]: ' 'Y' valid_yes_no "Answer Y or N."
    [[ "$CREATE" =~ ^[Yy] ]] && CREATE="y" || CREATE="n"
  fi
  if [[ "$CREATE" == "y" ]]; then
    [[ -z "$S_ACCOUNT" ]] || { echo "ERROR: --account uses an existing account: do not give a new user's fields with it" >&2; exit 2; }
    echo "-- the new user (an account is created in the Postgres account book, and the money is added to it) --"
    need_value S_NAME  'Name  [Bob Jones]: ' 'Bob Jones' valid_name "A name is letters, spaces and . ' - (at most 99 characters)."
    need_value S_AGE   'Age  [60] (18 to 120; an older user is held to a lower risk): ' '60' valid_age "Age is a whole number from 18 to 120."
    need_value S_RISK  'Risk tolerance  [conservative] (conservative, moderate or aggressive): ' 'conservative' valid_risk "Risk tolerance is conservative, moderate or aggressive."
    need_value S_CASH  'Money to add to the account, in dollars  [15000]: ' '15000' valid_cash "Money is a number of dollars above zero, with at most two decimals (no $ or commas)."
    need_value S_MODE  'Trade mode  [ask] (ask: you approve each trade before it is made; auto: the desk trades what the Critic approves): ' 'ask' valid_mode "Trade mode is ask or auto."
  else
    if [[ -n "$S_NAME$S_AGE$S_RISK$S_CASH$S_MODE" ]]; then echo "ERROR: --account uses an existing account: do not give --name, --age, --risk, --cash or --mode with it" >&2; exit 2; fi
    echo "-- an existing account (no user is created and no money is added) --"
    need_value S_ACCOUNT 'Account number, or a sample key such as jacob_sample: ' '' valid_account "An account is a number, or a sample key in lower case letters, digits and _."
  fi
  # 2. the main prompt: what the user wants, in their own words
  echo "-- the request --"
  need_value S_PROMPT 'Tell me what you want to do?:  [Enter: the sample request, buy oil stocks]: ' 'I want to buy oil stocks. Use as much of the money in my account as you can to buy shares of the 3 most recommended oil companies, and split the money evenly among the 3.' valid_prompt "Say what you want (1 to 600 characters)."
  [[ -z "$S_COUNT" ]] || { [[ "$S_COUNT" =~ ^[0-9]+$ ]] || { echo "ERROR: --default-count is a whole number" >&2; exit 2; }; }

  # Fetching uses the PAID APIs and writes files: say so, and ask. Nobody to ask (the input ended) and no --yes: NO, so a script that is not watched never spends by default.
  FETCH_FLAG=""
  if [[ $NO_FETCH -eq 0 ]]; then
    if [[ $YES -eq 1 ]]; then
      FETCH_FLAG="--fetch"
    else
      ask CONFIRM "Refresh the data of the companies (daily bars, news, press releases, SEC filings) with the PAID APIs and write news/, data/press_releases/, data/filings/ and data/historical/ticker_bars_v1/? [Y/n]: " 'Y'
      if [[ $INPUT_ENDED -eq 1 ]]; then echo "  (no answer: not fetching. Pass --yes to fetch without being asked.)" >&2
      elif [[ ! "$CONFIRM" =~ ^[Nn] ]]; then FETCH_FLAG="--fetch"; fi
    fi
  fi

  CMD=(python -m prometheus.application.session --prompt "$S_PROMPT")
  if [[ -n "$S_ACCOUNT" ]]; then CMD+=(--account "$S_ACCOUNT"); else CMD+=(--name "$S_NAME" --age "$S_AGE" --risk "$S_RISK" --cash "$S_CASH" --mode "$S_MODE"); fi
  [[ -n "$FETCH_FLAG" ]] && CMD+=(--fetch)
  [[ $YES -eq 1 ]] && CMD+=(--yes)
  [[ -n "$MODEL" ]] && CMD+=(--model "$MODEL")
  [[ -n "$S_COUNT" ]] && CMD+=(--default-count "$S_COUNT")
  [[ -n "$S_AUTO_MIN" ]] && CMD+=(--auto-min-score "$S_AUTO_MIN")
  [[ -n "$S_MAX_POS" ]] && CMD+=(--max-position-pct "$S_MAX_POS")
  [[ -n "$S_MAX_EXP" ]] && CMD+=(--max-exposure-pct "$S_MAX_EXP")
  [[ -n "$S_MAX_LOSS" ]] && CMD+=(--max-loss-pct "$S_MAX_LOSS")
  [[ $S_NO_TRADER_MODEL -eq 1 ]] && CMD+=(--no-trader-model)
  [[ $S_REAL_LEDGER -eq 1 ]] && CMD+=(--real-ledger)
  [[ -n "$S_SAVE" ]] && CMD+=(--save "$S_SAVE")
  [[ $S_CLEANUP -eq 1 ]] && CMD+=(--cleanup)
  [[ $S_ALLOW_NO_TRADE -eq 1 ]] && CMD+=(--allow-no-trade)
  echo ""
  if [[ -n "${PROMETHEUS_DRY_RUN:-}" ]]; then
    printf 'DRY RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'
    exit 0
  fi
  set +e
  "${CMD[@]}"
  CODE=$?
  set -e
  echo ""
  if [[ $CODE -eq 0 ]]; then echo "OK: the Prometheus session finished"; else echo "Prometheus session: exit code $CODE" >&2; fi
  exit $CODE
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
