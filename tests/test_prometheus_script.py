"""Tests for ./test_prometheus.sh: the prompts for symbol, term, source and from_days_ago, and the command it builds for the desk.

Run:  pytest tests/test_prometheus_script.py -q

The script is run for real with bash, with PROMETHEUS_DRY_RUN=1 so that it prints the command it WOULD run instead of running it: no network, no
key, no cost, nothing fetched or written. Answers are piped in, the way a person would type them.
"""
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "test_prometheus.sh"


def run(stdin: str = "", *args: str, dry: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "PROMETHEUS_DRY_RUN": "1"} if dry else dict(os.environ)
    return subprocess.run(["bash", str(SCRIPT), "--skip-inspect", *args], input=stdin, capture_output=True, text=True, cwd=REPO, env=env, timeout=30)


def command(result: subprocess.CompletedProcess) -> list[str]:
    """The command the script said it would run, as a list of arguments."""
    line = next(line for line in result.stdout.splitlines() if line.startswith("DRY RUN:"))
    return shlex.split(line.removeprefix("DRY RUN:"))


def test_the_script_is_executable_and_valid_bash():
    assert os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0


def test_it_asks_for_the_four_parameters_in_order_and_the_defaults_are_news_and_seven_days():
    result = run("\nflying car\n\n\n")  # no symbol, the term, the default source, the default days
    assert result.returncode == 0, result.stderr
    prompts = result.stderr
    order = [prompts.index(p) for p in ("Symbol", "Term", "Source", "From how many days ago")]
    assert order == sorted(order)
    assert "[news]" in prompts and "[7]" in prompts and "press_releases for the company's own announcements" in prompts and "filings for its SEC filings" in prompts  # news is still the default; press_releases (entry 34) and filings (entry 35) follow
    assert command(result) == ["python", "-m", "prometheus.langgraph.desk", "--symbol", "", "--term", "flying car", "--source", "news", "--from-days-ago", "7"]
    assert "Fetch fresh news" not in prompts  # a term has no ticker to fetch, so it is not asked


def test_a_symbol_asks_before_it_fetches_and_the_default_answer_is_yes():
    result = run("MSFT\n\n\n14\n\n")
    assert "Fetch fresh news for MSFT with the PAID Finnhub/Polygon APIs and write news/MSFT.txt? [Y/n]" in result.stderr
    assert "FETCH fresh news for MSFT (paid APIs; writes news/MSFT.txt)" in result.stdout
    assert command(result) == ["python", "-m", "prometheus.langgraph.desk", "--symbol", "MSFT", "--term", "", "--source", "news", "--from-days-ago", "14"]


def test_when_the_input_ends_before_the_fetch_question_is_answered_nothing_is_fetched():
    """A script that nobody is watching must not spend the paid APIs by default."""
    result = run("msft\n\n\n7\n")  # the confirmation gets end-of-input
    assert "--no-fetch" in command(result)
    assert "no answer: not fetching. Pass --yes to fetch without being asked." in result.stderr and "FETCH fresh news" not in result.stdout


def test_yes_fetches_without_asking_and_gives_the_symbol_a_fetch_note():
    result = run("", "--symbol", "$AAPL", "--from-days-ago", "3", "--yes")
    assert "Fetch fresh news for" not in result.stderr
    assert command(result) == ["python", "-m", "prometheus.langgraph.desk", "--symbol", "AAPL", "--term", "", "--source", "news", "--from-days-ago", "3"]
    assert "FETCH fresh news for AAPL" in result.stdout


def test_a_company_name_is_accepted_at_the_symbol_prompt_and_passed_on_as_typed():
    result = run("Procter&Gamble\n\n\n7\n\n")
    assert result.returncode == 0 and "or a company name such as Procter & Gamble" in result.stderr
    assert command(result) == ["python", "-m", "prometheus.langgraph.desk", "--symbol", "Procter&Gamble", "--term", "", "--source", "news", "--from-days-ago", "7"]
    # the confirmation names what will happen: a lookup first, and a file named after the ticker the lookup finds
    assert "Fetch fresh news for 'Procter&Gamble' (its ticker is looked up first) with the PAID Finnhub/Polygon APIs and write news/<its ticker>.txt? [Y/n]" in result.stderr
    assert "FETCH fresh news for 'Procter&Gamble' (its ticker is looked up first) (paid APIs; writes news/<its ticker>.txt)" in result.stdout


def test_parameters_given_as_flags_are_not_asked_for():
    result = run("", "--symbol", "MSFT", "--term", "copilot", "--source", "news", "--from-days-ago", "14", "--yes")
    assert result.returncode == 0 and "Symbol  (" not in result.stderr and "Term    (" not in result.stderr
    assert command(result)[3:] == ["--symbol", "MSFT", "--term", "copilot", "--source", "news", "--from-days-ago", "14"]


def test_with_no_input_at_all_it_stops_instead_of_looping():
    result = run("")
    assert result.returncode == 2 and "Give a symbol, a term, or both." in result.stderr


def test_an_invalid_symbol_given_as_a_flag_is_refused_without_asking_again():
    result = run("", "--symbol", "../etc")
    assert result.returncode == 2 and "not a valid ticker symbol or company name" in result.stderr


def test_the_other_flags_reach_the_desk():
    cmd = command(run("", "--term", "flying car", "--no-advisor", "--no-fetch", "--model", "x/y"))
    assert cmd[-4:] == ["--no-fetch", "--no-advisor", "--model", "x/y"]


def test_the_old_demo_is_still_there_behind_legacy():
    result = run("", "--legacy")
    assert result.returncode == 0 and "DRY RUN: python main.py" in result.stdout and "Symbol  (" not in result.stderr


def test_help_explains_the_four_parameters_and_unknown_options_are_refused():
    help_text = run("", "--help").stdout
    for part in ("symbol", "term", "source", "from_days_ago", "Default 7", "news (the default), press_releases or filings", "company", "--company", "--legacy", "--yes"):
        assert part in help_text, part
    bad = run("", "--nope")
    assert bad.returncode == 2 and "unknown option: --nope" in bad.stderr
    assert run("", "--symbol").returncode == 2  # a flag without its value


# ================================================================ press releases (design choices entry 34)
#
# source press_releases: the company's own announcements. A company name is asked for only for this source, so the news prompts are exactly as they were.
# The ticker is looked up first, the company name only if the ticker is not found (that is the Data Clerk's rule: the script only passes both on).

DESK = ["python", "-m", "prometheus.langgraph.desk"]


def test_press_releases_are_asked_for_in_the_same_order_with_a_company_prompt_after_the_source_and_a_year_by_default():
    result = run("AAPL\n\npress_releases\n\n\n\n")  # symbol, no term, the source, no company, the default days, the default (yes) to the fetch question
    assert result.returncode == 0, result.stderr
    prompts = result.stderr
    order = [prompts.index(p) for p in ("Symbol", "Term", "Source", "Company name", "From how many days ago")]
    assert order == sorted(order) and "[365]" in prompts and "(1 to 730)" in prompts
    assert command(result) == [*DESK, "--symbol", "AAPL", "--term", "", "--source", "press_releases", "--from-days-ago", "365"]  # no --company: none was given
    assert "Fetch fresh press releases for AAPL with the PAID Finnhub API and write data/press_releases/AAPL_PRs.txt? [Y/n]" in prompts
    assert "FETCH fresh press releases for AAPL (paid APIs; writes data/press_releases/AAPL_PRs.txt)" in result.stdout


def test_company_with_the_news_source_is_refused_and_not_silently_ignored():
    result = run("", "--symbol", "MSFT", "--source", "news", "--company", "Microsoft", "--yes")
    assert result.returncode == 2 and "--company is only used with --source press_releases or filings (not 'news')" in result.stderr and "DRY RUN" not in result.stdout


def test_the_fetch_question_for_press_releases_follows_the_same_safety_rules_as_news():
    no = run("AAPL\n\npress_releases\n\n\nn\n")
    assert "--no-fetch" in command(no) and "no fetch (the stored press releases are searched; the symbol or name is still looked up in Finnhub)" in no.stdout
    ended = run("AAPL\n\npress_releases\n\n\n")  # the input ends at the confirmation: nothing is fetched
    assert "--no-fetch" in command(ended) and "no answer: not fetching. Pass --yes to fetch without being asked." in ended.stderr and "FETCH fresh" not in ended.stdout
    flagged = run("", "--source", "press_releases", "--symbol", "AAPL", "--no-fetch")
    assert "--no-fetch" in command(flagged) and "Fetch fresh" not in flagged.stderr  # --no-fetch: not even asked


# ================================================================ filings (design choices entry 35)
#
# source filings: the SEC's filings. Asked for like press releases (a symbol and/or a company name, the ticker first), with the same prompts, window (365, 1 to 730) and fetch
# rules. What differs: the paid API named in the question (Massive/Polygon), the file (data/filings/<TICKER>.txt) and the words.

def test_filings_are_asked_for_in_the_same_order_with_a_company_prompt_after_the_source_and_a_year_by_default():
    result = run("AAPL\n\nfilings\n\n\n\n")  # symbol, no term, the source, no company, the default days, the default (yes) to the fetch question
    assert result.returncode == 0, result.stderr
    prompts = result.stderr
    order = [prompts.index(p) for p in ("Symbol", "Term", "Source", "Company name", "From how many days ago")]
    assert order == sorted(order) and "[365]" in prompts and "(1 to 730)" in prompts
    assert "Company name  (filings only, used when the symbol is not given or not found; blank for none)" in prompts
    assert command(result) == [*DESK, "--symbol", "AAPL", "--term", "", "--source", "filings", "--from-days-ago", "365"]  # no --company: none was given
    assert "Fetch fresh filings for AAPL with the PAID Massive (Polygon) API and write data/filings/AAPL.txt? [Y/n]" in prompts
    assert "FETCH fresh filings for AAPL (paid APIs; writes data/filings/AAPL.txt)" in result.stdout


def test_the_fetch_question_for_filings_follows_the_same_safety_rules_as_news():
    no = run("AAPL\n\nfilings\n\n\nn\n")
    assert "--no-fetch" in command(no) and "no fetch (the stored filings are searched; the symbol or name is still looked up in Finnhub)" in no.stdout
    ended = run("AAPL\n\nfilings\n\n\n")  # the input ends at the confirmation: nothing is fetched
    assert "--no-fetch" in command(ended) and "no answer: not fetching. Pass --yes to fetch without being asked." in ended.stderr and "FETCH fresh" not in ended.stdout
    flagged = run("", "--source", "filings", "--symbol", "AAPL", "--no-fetch")
    assert "--no-fetch" in command(flagged) and "Fetch fresh" not in flagged.stderr  # --no-fetch: not even asked


# ---------------------------------------------------------------- ./test_prometheus.sh --session: a user session, end to end (design choices entry 78)

BOB = ["--name", "Bob Jones", "--age", "60", "--risk", "conservative", "--cash", "15000", "--mode", "auto"]
OIL = "I want to buy oil stocks. Buy the most amount of oil stocks shares from the 3 most recommended companies. Balance the purchased stocks evenly among the 3 recommended companies."
SESSION = ["python", "-m", "prometheus.application.session"]


def test_a_session_with_every_answer_given_builds_the_command_for_the_python_session():
    result = run("", "--session", *BOB, "--prompt", OIL, "--yes")
    assert result.returncode == 0, result.stderr
    assert command(result) == [*SESSION, "--prompt", OIL, "--name", "Bob Jones", "--age", "60", "--risk", "conservative", "--cash", "15000", "--mode", "auto", "--fetch", "--yes"]


def test_a_session_asks_for_the_user_and_the_request_in_order_and_enter_takes_each_default():
    result = run("\n\n\n\n\n\n\nn\n", "--session")                       # create a user? (Enter is yes), name, age, risk, money, mode, the request: Enter each; then no to the paid fetch
    assert result.returncode == 0, result.stderr
    prompts = result.stderr
    order = [prompts.index(p) for p in ("Create a new user?", "Name", "Age", "Risk tolerance", "Money to add", "Trade mode", "Tell me what you want to do?:", "Refresh the data")]
    assert order == sorted(order) and "-- the new user (an account is created in the Postgres account book, and the money is added to it) --" in result.stdout and "-- the request --" in result.stdout
    cmd = command(result)
    assert cmd[:3] == SESSION and cmd[cmd.index("--name") + 1] == "Bob Jones" and cmd[cmd.index("--age") + 1] == "60" and cmd[cmd.index("--risk") + 1] == "conservative"
    assert cmd[cmd.index("--cash") + 1] == "15000" and cmd[cmd.index("--mode") + 1] == "ask" and "I want to buy oil stocks." in cmd[cmd.index("--prompt") + 1]      # Ask is the default: a person approves each trade
    assert "--fetch" not in cmd and "--yes" not in cmd                       # the answer to the fetch question was no


def test_a_person_can_type_any_user_and_any_request_the_script_knows_no_company_or_sector():
    result = run("y\nAda Lovelace\n41\naggressive\n2500.50\nask\nShow me the best semiconductor companies, I might buy 2 of them.\ny\n", "--session")
    assert result.returncode == 0, result.stderr
    cmd = command(result)
    assert cmd[cmd.index("--name") + 1] == "Ada Lovelace" and cmd[cmd.index("--risk") + 1] == "aggressive" and cmd[cmd.index("--cash") + 1] == "2500.50" and cmd[cmd.index("--mode") + 1] == "ask"
    assert cmd[cmd.index("--prompt") + 1] == "Show me the best semiconductor companies, I might buy 2 of them." and "--fetch" in cmd
    source = SCRIPT.read_text(encoding="utf-8")
    session_block = source[source.index("if [[ $SESSION -eq 1 ]]; then"):source.index("if [[ $SKIP_INSPECT -eq 0")]
    for word in ("XOM", "CVX", "Chevron", "Exxon", "AAPL", "MSFT"):
        assert word not in session_block, word


@pytest.mark.parametrize("flags, message", [
    (["--name", "1bad"], "A name is letters"), (["--age", "17"], "Age is a whole number from 18 to 120"), (["--age", "abc"], "Age is a whole number"), (["--risk", "low"], "Risk tolerance is conservative"),
    (["--cash", "$5,000"], "Money is a number of dollars"), (["--cash", "-5"], "Money is a number of dollars"), (["--mode", "manual"], "Trade mode is ask or auto"), (["--prompt", " "], "Say what you want"),
])
def test_a_flag_that_is_not_valid_is_refused_once_and_never_asked_again(flags, message):
    given = {"--name": "Bob Jones", "--age": "60", "--risk": "conservative", "--cash": "15000", "--mode": "auto", "--prompt": OIL}
    for key, value in zip(flags[::2], flags[1::2]):
        given[key] = value
    result = run("", "--session", *[x for kv in given.items() for x in kv])
    assert result.returncode == 2 and message in result.stderr and "DRY RUN" not in result.stdout


def test_an_existing_account_takes_no_user_and_a_new_user_flag_with_it_is_a_mistake():
    result = run("", "--session", "--account", "7", "--prompt", OIL, "--yes")
    assert result.returncode == 0 and command(result) == [*SESSION, "--prompt", OIL, "--account", "7", "--fetch", "--yes"] and "-- an existing account (no user is created and no money is added) --" in result.stdout
    assert "Create a new user?" not in result.stderr                         # the flag settled it
    clash = run("", "--session", "--account", "7", "--name", "Bob Jones", "--prompt", OIL)
    assert clash.returncode == 2 and "do not give --name, --age, --risk, --cash or --mode" in clash.stderr


def test_the_paid_fetch_is_asked_about_and_nobody_answering_is_a_no():
    asked = run("", "--session", *BOB, "--prompt", OIL)                       # nothing to read: the question gets no answer
    assert asked.returncode == 0 and "no answer: not fetching" in asked.stderr and "--fetch" not in command(asked)
    assert "Refresh the data of the companies" in asked.stderr and "PAID APIs" in asked.stderr and "data/historical/ticker_bars_v1/" in asked.stderr
    assert "--fetch" in command(run("y\n", "--session", *BOB, "--prompt", OIL)) and "--fetch" in command(run("\n", "--session", *BOB, "--prompt", OIL))   # Enter is yes
    assert "--fetch" not in command(run("n\n", "--session", *BOB, "--prompt", OIL))
    assert "--fetch" not in command(run("", "--session", *BOB, "--prompt", OIL, "--no-fetch")) and "--fetch" in command(run("", "--session", *BOB, "--prompt", OIL, "--yes"))


# ---- the session as the Account Manager reads it now: a trade the user names (design choices entry 87) does no research, so the paid refresh is not asked about

DIRECT_REQUESTS = json.loads((REPO / "tests" / "fixtures" / "session_real" / "direct_order_forms.json").read_text(encoding="utf-8"))["requests"]


def fetched(prompt: str, *more: str) -> bool:
    """Whether the script would pass `--fetch` for this request when the user said --yes (so a request that does no research must not)."""
    return "--fetch" in command(run("", "--session", *BOB, "--prompt", prompt, "--yes", *more))


def test_a_request_that_names_the_trade_asks_no_paid_question_and_passes_no_fetch_even_with_yes():
    result = run("", "--session", *BOB, "--prompt", "Buy 50 shares of MSFT", "--yes")
    assert result.returncode == 0 and "names the trade itself, a direct order" in result.stdout
    assert command(result) == [*SESSION, "--prompt", "Buy 50 shares of MSFT", "--name", "Bob Jones", "--age", "60", "--risk", "conservative", "--cash", "15000", "--mode", "auto", "--yes"]      # no --fetch
    asked = run("", "--session", *BOB, "--prompt", "Sell 10 shares of TSLA")                  # without --yes: the paid question is not asked either
    assert asked.returncode == 0 and "Refresh the data" not in asked.stderr and "--fetch" not in command(asked) and "no answer: not fetching" not in asked.stderr
    assert "--fetch" in command(run("", "--session", *BOB, "--prompt", OIL, "--yes")) and "Refresh the data" in run("", "--session", *BOB, "--prompt", OIL).stderr      # a request for recommendations is asked as before


@pytest.mark.parametrize("typed, names_one", [
    ("Buy 50 shares of MSFT", True), ("sell 10 shares of tsla", False), ("Sell 10 shares of TSLA", True), ("Please purchase twenty-five shares of ABCD", True), ("Please purchase five shares of ABCD", True),
    ("Buy $5,000 of MSFT", True), ("I want to buy MSFT", True), ("I want to buy MU, SNDK and AMD if the recommendations pass", False), ("I want to buy the three most recommended oil stocks", False),
    ("Buy the 3 best US oil stocks", False), ("I want to buy oil stocks. Buy the 3 most recommended companies.", False), ("Buy 3 shares of oil stocks", False),
    ("Show me 50 shares of MSFT", False),               # no buying or selling word: nothing is asked of the desk
])
def test_what_the_script_takes_for_a_trade_you_name_yourself(typed, names_one):
    assert (not fetched(typed)) is names_one, typed


def test_the_help_says_which_way_in_is_the_end_to_end_one_and_what_a_trade_you_name_does():
    result = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True, cwd=REPO, timeout=30)
    text = result.stdout
    assert result.returncode == 0 and "TWO WAYS IN" in text and "THE END-TO-END PATH, through the Account Manager's own interface" in text
    assert "NAMES THE TRADE" in text and "DIRECT ORDER" in text and "Buy 50 shares of MSFT" in text and 'asks (a percentage, dollars, shares or "as much as I can"' in text


def test_the_other_session_flags_are_passed_on_exactly_as_given():
    result = run("", "--session", *BOB, "--prompt", OIL, "--no-fetch", "--cleanup", "--allow-no-trade", "--no-trader-model", "--real-ledger", "--auto-min-score", "70",
                 "--default-count", "4", "--model", "openai/gpt-oss-120b")
    cmd = command(result)
    assert result.returncode == 0 and "--cleanup" in cmd and "--allow-no-trade" in cmd and "--no-trader-model" in cmd and "--real-ledger" in cmd
    assert cmd[cmd.index("--auto-min-score") + 1] == "70"
    assert cmd[cmd.index("--default-count") + 1] == "4" and cmd[cmd.index("--model") + 1] == "openai/gpt-oss-120b" and "--fetch" not in cmd and "--yes" not in cmd


def test_the_session_flags_without_session_are_refused_not_ignored():
    for flags in (["--name", "Bob"], ["--prompt", "x"], ["--cleanup"], ["--account", "5"], ["--allow-no-trade"]):
        result = run("", "--symbol", "MSFT", "--no-fetch", *flags)
        assert result.returncode == 2 and "go with --session" in result.stderr, flags


def test_the_old_modes_are_exactly_what_they_were():
    result = run("\nflying car\n\n\n")
    assert result.returncode == 0 and command(result)[:3] == ["python", "-m", "prometheus.langgraph.desk"] and "Symbol" in result.stderr


def test_the_first_question_is_whether_to_create_a_user_and_no_asks_for_an_existing_account_instead():
    result = run("n\njacob_sample\n\nn\n", "--session")                      # no, the account, the request (Enter: the sample), no to the paid fetch
    assert result.returncode == 0, result.stderr
    prompts = result.stderr
    order = [prompts.index(p) for p in ("Create a new user?", "Account number, or a sample key", "Tell me what you want to do?:", "Refresh the data")]
    assert order == sorted(order) and "Name" not in prompts and "Money to add" not in prompts                        # no user fields were asked
    cmd = command(result)
    assert cmd[cmd.index("--account") + 1] == "jacob_sample" and "--name" not in cmd and "--cash" not in cmd and "--mode" not in cmd and "I want to buy oil stocks." in cmd[cmd.index("--prompt") + 1]


def test_a_flag_for_any_field_of_the_user_means_a_new_user_without_asking_whether_to_create_one():
    result = run("\n\n\n\n\n\nn\n", "--session", "--name", "Ada Lovelace")
    assert result.returncode == 0 and "Create a new user?" not in result.stderr and "Name  [" not in result.stderr and "Age  [60]" in result.stderr
    assert command(result)[command(result).index("--name") + 1] == "Ada Lovelace"


def test_no_answer_at_all_to_the_first_question_starts_nothing_instead_of_creating_a_user_on_the_defaults():
    result = run("", "--session")
    assert result.returncode == 2 and "no answer: the input ended, so nothing was started" in result.stderr and "DRY RUN" not in result.stdout


def test_the_script_has_no_flag_for_a_size_limit_and_refuses_one():
    """Design choices entry 91: there is no profile limit to raise for a session, so the flags that did are gone (the script refuses them as any unknown option)."""
    for flag in ("--max-position-pct", "--max-exposure-pct", "--max-loss-pct"):
        result = run("", "--session", *BOB, "--prompt", OIL, flag, "0.5")
        assert result.returncode == 2 and "unknown option" in result.stderr, flag
