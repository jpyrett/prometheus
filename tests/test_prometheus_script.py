"""Tests for ./test_prometheus.sh: the prompts for symbol, term, source and from_days_ago, and the command it builds for the desk.

Run:  pytest tests/test_prometheus_script.py -q

The script is run for real with bash, with PROMETHEUS_DRY_RUN=1 so that it prints the command it WOULD run instead of running it: no network, no
key, no cost, nothing fetched or written. Answers are piped in, the way a person would type them.
"""
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


@pytest.mark.parametrize("answer", ["n", "N", "no", "No"])
def test_answering_no_to_the_fetch_question_turns_the_fetch_off(answer):
    result = run(f"msft\n\n\n7\n{answer}\n")
    assert "--no-fetch" in command(result) and "no fetch (the stored news is searched)" in result.stdout


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


def test_a_company_name_with_spaces_given_as_a_flag_is_passed_as_one_argument():
    result = run("", "--symbol", "Procter & Gamble", "--from-days-ago", "7", "--yes")
    assert result.returncode == 0 and command(result)[command(result).index("--symbol") + 1] == "Procter & Gamble"


def test_a_ticker_in_lower_case_is_passed_as_typed_and_is_said_to_be_looked_up():
    result = run("", "--symbol", "$aapl", "--yes")
    assert command(result)[command(result).index("--symbol") + 1] == "aapl"
    assert "FETCH fresh news for 'aapl' (its ticker is looked up first)" in result.stdout


def test_a_ticker_in_capitals_is_not_said_to_be_looked_up():
    result = run("", "--symbol", "BRK.B", "--yes")
    assert "FETCH fresh news for BRK.B (paid APIs; writes news/BRK.B.txt)" in result.stdout and "looked up" not in result.stdout


@pytest.mark.parametrize("typed", ["Dillard's", "O'Reilly Automotive", "Coca-Cola", "AT&T", "Berkshire Hathaway Inc."])
def test_ordinary_company_names_are_accepted(typed):
    cmd = command(run("", "--symbol", typed, "--yes"))
    assert cmd[cmd.index("--symbol") + 1] == typed


@pytest.mark.parametrize("typed", ["a;b", "x|y", "../etc/passwd", "$(whoami)", "`id`", "a" * 61, "&amp", "'quoted"])
def test_text_that_is_neither_a_ticker_nor_a_company_name_is_refused(typed):
    result = run("", "--symbol", typed, "--yes")
    assert result.returncode == 2 and "is not a valid ticker symbol or company name." in result.stderr and "DRY RUN" not in result.stdout


def test_parameters_given_as_flags_are_not_asked_for():
    result = run("", "--symbol", "MSFT", "--term", "copilot", "--source", "news", "--from-days-ago", "14", "--yes")
    assert result.returncode == 0 and "Symbol  (" not in result.stderr and "Term    (" not in result.stderr
    assert command(result)[3:] == ["--symbol", "MSFT", "--term", "copilot", "--source", "news", "--from-days-ago", "14"]


def test_a_symbol_and_a_term_together_are_both_passed_on():
    cmd = command(run("SNDK\nlong-term contracts\n\n7\nn\n"))
    assert cmd[cmd.index("--symbol") + 1] == "SNDK" and cmd[cmd.index("--term") + 1] == "long-term contracts"


def test_with_nothing_given_it_asks_again_and_then_gives_up_when_the_input_ends():
    result = run("\n\n")  # a blank symbol and a blank term, then the input ends
    assert result.returncode == 2 and "Give a symbol, a term, or both." in result.stderr and "DRY RUN" not in result.stdout


def test_with_no_input_at_all_it_stops_instead_of_looping():
    result = run("")
    assert result.returncode == 2 and "Give a symbol, a term, or both." in result.stderr


def test_a_blank_answer_the_first_time_and_a_good_one_the_second_time_is_accepted():
    result = run("\n\n\nflying car\n\n\n")  # blank, blank (asked again), then the term
    assert result.returncode == 0 and result.stderr.count("Give a symbol, a term, or both.") == 1
    assert command(result)[command(result).index("--term") + 1] == "flying car"


def test_an_invalid_symbol_is_asked_for_again():
    result = run("../etc\nmsft\n\n\n\nn\n")
    assert "'../etc' is not a valid ticker symbol or company name." in result.stderr and result.returncode == 0
    assert command(result)[command(result).index("--symbol") + 1] == "msft"  # passed as typed: the desk looks a ticker in lower case up


def test_an_invalid_symbol_given_as_a_flag_is_refused_without_asking_again():
    result = run("", "--symbol", "../etc")
    assert result.returncode == 2 and "not a valid ticker symbol or company name" in result.stderr


@pytest.mark.parametrize("answers, expected", [("x\n91\nabc\n0\n10\n", "10"), ("\n", "7")])
def test_days_outside_one_to_ninety_are_asked_for_again_and_blank_is_seven(answers, expected):
    result = run("\nflying car\n\n" + answers)
    assert command(result)[command(result).index("--from-days-ago") + 1] == expected
    if expected == "10":
        assert result.stderr.count("is not a whole number of days from 1 to 90") == 4


@pytest.mark.parametrize("days", ["0", "91", "-1", "seven", "7.5", ""])
def test_days_that_are_not_valid_as_a_flag_are_refused(days):
    result = run("", "--term", "flying car", "--from-days-ago", days)
    if days == "":
        assert result.returncode == 0 and command(result)[command(result).index("--from-days-ago") + 1] == "7"  # blank means the default, which is 7
    else:
        assert result.returncode == 2 and "is not a whole number of days from 1 to 90" in result.stderr


def test_a_source_that_is_not_wired_is_asked_for_again_and_news_is_the_default():
    result = run("\nflying car\nsocial\nnews\n\n")  # 'filings' was the example until entry 35 wired it
    assert "Source 'social' is not wired yet (only: news, press_releases, filings)." in result.stderr and command(result)[command(result).index("--source") + 1] == "news"
    assert run("", "--term", "flying car", "--source", "social").returncode == 2


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


def test_the_company_prompt_is_only_asked_for_press_releases():
    assert "Company name" not in run("MSFT\n\n\n7\nn\n").stderr
    assert "Company name" in run("AAPL\n\npress_releases\n\n\nn\n").stderr


def test_a_company_typed_at_the_prompt_is_passed_on_with_the_symbol():
    result = run("AAPL\n\npress_releases\nApple Inc\n\nn\n")
    cmd = command(result)
    assert cmd[cmd.index("--symbol") + 1] == "AAPL" and cmd[cmd.index("--company") + 1] == "Apple Inc" and "--no-fetch" in cmd
    assert result.returncode == 0 and cmd[cmd.index("--from-days-ago") + 1] == "365"


def test_a_company_name_alone_as_a_flag_is_a_whole_request_with_a_lookup_before_the_file_is_known():
    result = run("", "--source", "press_releases", "--company", "Apple Inc", "--yes")
    assert result.returncode == 0, result.stderr
    assert command(result) == [*DESK, "--symbol", "", "--term", "", "--source", "press_releases", "--from-days-ago", "365", "--company", "Apple Inc"]
    assert "FETCH fresh press releases for 'Apple Inc' (its ticker is looked up first) (paid APIs; writes data/press_releases/<its ticker>_PRs.txt)" in result.stdout
    assert "company: Apple Inc" in result.stdout


def test_company_alone_means_press_releases_and_a_company_name_with_spaces_is_one_argument():
    result = run("", "--company", "Procter & Gamble Co", "--yes")
    cmd = command(result)
    assert result.returncode == 0 and cmd[cmd.index("--source") + 1] == "press_releases" and cmd[cmd.index("--company") + 1] == "Procter & Gamble Co"


def test_company_with_the_news_source_is_refused_and_not_silently_ignored():
    result = run("", "--symbol", "MSFT", "--source", "news", "--company", "Microsoft", "--yes")
    assert result.returncode == 2 and "--company is only used with --source press_releases or filings (not 'news')" in result.stderr and "DRY RUN" not in result.stdout


def test_a_symbol_and_a_company_name_are_both_passed_on_and_the_confirmation_says_which_comes_first():
    result = run("", "--source", "press_releases", "--symbol", "AAPL", "--company", "Apple Inc", "--yes")
    cmd = command(result)
    assert cmd[cmd.index("--symbol") + 1] == "AAPL" and cmd[cmd.index("--company") + 1] == "Apple Inc"
    assert "FETCH fresh press releases for AAPL (if it is not found: 'Apple Inc')" in result.stdout


def test_a_ticker_in_lower_case_is_passed_as_typed_and_its_file_is_named_in_capitals():
    result = run("", "--source", "press_releases", "--symbol", "$aapl", "--yes")
    cmd = command(result)
    assert cmd[cmd.index("--symbol") + 1] == "aapl" and "writes data/press_releases/AAPL_PRs.txt" in result.stdout  # the Data Clerk looks it up and upper-cases it


def test_a_term_narrows_a_press_release_request_and_cannot_replace_the_symbol_or_the_company():
    narrowed = command(run("", "--source", "press_releases", "--symbol", "AAPL", "--term", "dividend", "--yes"))
    assert narrowed[narrowed.index("--term") + 1] == "dividend"
    result = run("", "--source", "press_releases", "--term", "dividend", "--yes")
    assert result.returncode == 2 and "A press release request needs a symbol, a company name, or both." in result.stderr and "DRY RUN" not in result.stdout


def test_a_press_release_request_with_nothing_but_blank_answers_is_refused():
    result = run("\ndividend\npress_releases\n\n\n")  # a term only, then the source, then no company
    assert result.returncode == 2 and "A press release request needs a symbol, a company name, or both." in result.stderr


@pytest.mark.parametrize("days, ok", [("1", True), ("365", True), ("730", True), ("731", False), ("0", False), ("abc", False)])
def test_press_releases_take_one_to_730_days_and_news_still_one_to_90(days, ok):
    result = run("", "--source", "press_releases", "--symbol", "AAPL", "--from-days-ago", days, "--yes")
    assert (result.returncode == 0) is ok and (ok or "is not a whole number of days from 1 to 730" in result.stderr)
    assert run("", "--symbol", "AAPL", "--from-days-ago", "91", "--yes").returncode == 2  # news: unchanged


def test_blank_days_for_press_releases_means_a_year_and_for_news_a_week():
    pr = command(run("", "--source", "press_releases", "--symbol", "AAPL", "--from-days-ago", "", "--yes"))
    news = command(run("", "--symbol", "AAPL", "--from-days-ago", "", "--yes"))
    assert pr[pr.index("--from-days-ago") + 1] == "365" and news[news.index("--from-days-ago") + 1] == "7"


@pytest.mark.parametrize("typed", ["a;b", "x|y", "../etc/passwd", "$(whoami)", "`id`", "a" * 61, "&amp"])
def test_text_that_cannot_be_a_company_name_is_refused(typed):
    result = run("", "--source", "press_releases", "--company", typed, "--yes")
    assert result.returncode == 2 and "is not a valid company name." in result.stderr and "DRY RUN" not in result.stdout


def test_an_invalid_company_typed_at_the_prompt_is_asked_for_again():
    result = run("AAPL\n\npress_releases\na;b\nApple Inc\n\nn\n")
    assert "'a;b' is not a valid company name." in result.stderr and result.returncode == 0
    assert command(result)[command(result).index("--company") + 1] == "Apple Inc"


def test_the_fetch_question_for_press_releases_follows_the_same_safety_rules_as_news():
    no = run("AAPL\n\npress_releases\n\n\nn\n")
    assert "--no-fetch" in command(no) and "no fetch (the stored press releases are searched; the symbol or name is still looked up in Finnhub)" in no.stdout
    ended = run("AAPL\n\npress_releases\n\n\n")  # the input ends at the confirmation: nothing is fetched
    assert "--no-fetch" in command(ended) and "no answer: not fetching. Pass --yes to fetch without being asked." in ended.stderr and "FETCH fresh" not in ended.stdout
    flagged = run("", "--source", "press_releases", "--symbol", "AAPL", "--no-fetch")
    assert "--no-fetch" in command(flagged) and "Fetch fresh" not in flagged.stderr  # --no-fetch: not even asked


def test_a_source_other_than_news_press_releases_and_filings_is_still_refused():
    for typed in ("social", "Filings", "Press_Releases", "press-releases"):
        result = run("", "--term", "x", "--source", typed)
        assert result.returncode == 2 and f"Source '{typed}' is not wired yet (only: news, press_releases, filings)." in result.stderr


def test_help_explains_the_press_release_source_the_company_flag_and_the_ticker_first_rule():
    text = run("", "--help").stdout
    for part in ("press_releases", "--company NAME", "default 365, from 1 to 730", "The ticker comes first", "Symbol: X is not found", "Company Name not found.",
                 "--company alone implies --source press_releases", "A typo is NOT matched", "Pepsico"):
        assert part in text, part


def test_the_other_flags_still_reach_the_desk_after_a_press_release_request():
    cmd = command(run("", "--source", "press_releases", "--symbol", "AAPL", "--no-advisor", "--no-fetch", "--model", "x/y"))
    assert cmd[-4:] == ["--no-fetch", "--no-advisor", "--model", "x/y"] and "--company" not in cmd  # no company was given, so none is passed


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


def test_the_company_prompt_for_press_releases_keeps_its_words():
    assert "Company name  (press_releases only, used when the symbol is not given or not found; blank for none)" in run("AAPL\n\npress_releases\n\n\nn\n").stderr
    assert "Company name" not in run("MSFT\n\n\n7\nn\n").stderr  # and news has none


def test_a_company_typed_at_the_filings_prompt_is_passed_on_with_the_symbol():
    result = run("AAPL\n\nfilings\nApple Inc\n\nn\n")
    cmd = command(result)
    assert cmd[cmd.index("--symbol") + 1] == "AAPL" and cmd[cmd.index("--company") + 1] == "Apple Inc" and "--no-fetch" in cmd
    assert result.returncode == 0 and cmd[cmd.index("--source") + 1] == "filings" and cmd[cmd.index("--from-days-ago") + 1] == "365"


def test_a_company_name_alone_with_the_filings_source_is_a_whole_request_with_a_lookup_before_the_file_is_known():
    result = run("", "--source", "filings", "--company", "Apple Inc", "--yes")
    assert result.returncode == 0, result.stderr
    assert command(result) == [*DESK, "--symbol", "", "--term", "", "--source", "filings", "--from-days-ago", "365", "--company", "Apple Inc"]
    assert "FETCH fresh filings for 'Apple Inc' (its ticker is looked up first) (paid APIs; writes data/filings/<its ticker>.txt)" in result.stdout
    assert "company: Apple Inc" in result.stdout


def test_company_alone_still_means_press_releases_and_filings_must_be_asked_for_by_name():
    cmd = command(run("", "--company", "Procter & Gamble Co", "--yes"))
    assert cmd[cmd.index("--source") + 1] == "press_releases"  # unchanged by filings
    cmd = command(run("", "--company", "Procter & Gamble Co", "--source", "filings", "--yes"))
    assert cmd[cmd.index("--source") + 1] == "filings" and cmd[cmd.index("--company") + 1] == "Procter & Gamble Co"


def test_a_symbol_and_a_company_name_are_both_passed_on_for_filings_and_the_confirmation_says_which_comes_first():
    result = run("", "--source", "filings", "--symbol", "AAPL", "--company", "Apple Inc", "--yes")
    cmd = command(result)
    assert cmd[cmd.index("--symbol") + 1] == "AAPL" and cmd[cmd.index("--company") + 1] == "Apple Inc"
    assert "FETCH fresh filings for AAPL (if it is not found: 'Apple Inc')" in result.stdout


def test_a_ticker_in_lower_case_is_passed_as_typed_and_its_filings_file_is_named_in_capitals():
    result = run("", "--source", "filings", "--symbol", "$aapl", "--yes")
    cmd = command(result)
    assert cmd[cmd.index("--symbol") + 1] == "aapl" and "writes data/filings/AAPL.txt" in result.stdout


def test_a_filings_request_with_nothing_to_look_up_is_refused_in_the_words_of_filings():
    result = run("", "--source", "filings", "--term", "risk factors", "--yes")
    assert result.returncode == 2 and "A filings request needs a symbol, a company name, or both." in result.stderr and "DRY RUN" not in result.stdout
    assert run("\nrisk factors\nfilings\n\n\n").returncode == 2  # a term only, then the source, then no company
    assert "A press release request needs a symbol, a company name, or both." in run("", "--source", "press_releases", "--term", "x", "--yes").stderr  # unchanged


@pytest.mark.parametrize("days, ok", [("1", True), ("365", True), ("730", True), ("731", False), ("0", False), ("abc", False)])
def test_filings_take_one_to_730_days(days, ok):
    result = run("", "--source", "filings", "--symbol", "AAPL", "--from-days-ago", days, "--yes")
    assert (result.returncode == 0) is ok and (ok or "is not a whole number of days from 1 to 730" in result.stderr)


def test_blank_days_for_filings_means_a_year():
    cmd = command(run("", "--source", "filings", "--symbol", "AAPL", "--from-days-ago", "", "--yes"))
    assert cmd[cmd.index("--from-days-ago") + 1] == "365"


@pytest.mark.parametrize("typed", ["a;b", "x|y", "../etc/passwd", "$(whoami)", "`id`", "a" * 61, "&amp"])
def test_text_that_cannot_be_a_company_name_is_refused_for_filings_too(typed):
    result = run("", "--source", "filings", "--company", typed, "--yes")
    assert result.returncode == 2 and "is not a valid company name." in result.stderr and "DRY RUN" not in result.stdout


def test_the_fetch_question_for_filings_follows_the_same_safety_rules_as_news():
    no = run("AAPL\n\nfilings\n\n\nn\n")
    assert "--no-fetch" in command(no) and "no fetch (the stored filings are searched; the symbol or name is still looked up in Finnhub)" in no.stdout
    ended = run("AAPL\n\nfilings\n\n\n")  # the input ends at the confirmation: nothing is fetched
    assert "--no-fetch" in command(ended) and "no answer: not fetching. Pass --yes to fetch without being asked." in ended.stderr and "FETCH fresh" not in ended.stdout
    flagged = run("", "--source", "filings", "--symbol", "AAPL", "--no-fetch")
    assert "--no-fetch" in command(flagged) and "Fetch fresh" not in flagged.stderr  # --no-fetch: not even asked


def test_help_explains_the_filings_source_and_what_has_no_text():
    text = " ".join(run("", "--help").stdout.split())
    for part in ("--source filings", "news|press_releases|filings", "data/filings/<TICKER>.txt", "only 10-K and 8-K come with text", "press_releases and filings only",
                 "default 365, from 1 to 730 for press_releases and filings"):
        assert part in text, part


def test_the_other_flags_still_reach_the_desk_after_a_filings_request():
    cmd = command(run("", "--source", "filings", "--symbol", "AAPL", "--no-advisor", "--no-fetch", "--model", "x/y"))
    assert cmd[-4:] == ["--no-fetch", "--no-advisor", "--model", "x/y"] and "--company" not in cmd
