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
    assert "[news]" in prompts and "[7]" in prompts and "only news is wired" in prompts
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
    result = run("\nflying car\nfilings\nnews\n\n")
    assert "Source 'filings' is not wired yet (only: news)." in result.stderr and command(result)[command(result).index("--source") + 1] == "news"
    assert run("", "--term", "flying car", "--source", "filings").returncode == 2


def test_the_other_flags_reach_the_desk():
    cmd = command(run("", "--term", "flying car", "--no-advisor", "--no-fetch", "--model", "x/y"))
    assert cmd[-4:] == ["--no-fetch", "--no-advisor", "--model", "x/y"]


def test_the_old_demo_is_still_there_behind_legacy():
    result = run("", "--legacy")
    assert result.returncode == 0 and "DRY RUN: python main.py" in result.stdout and "Symbol  (" not in result.stderr


def test_help_explains_the_four_parameters_and_unknown_options_are_refused():
    help_text = run("", "--help").stdout
    for part in ("symbol", "term", "source", "from_days_ago", "Default 7", "only \"news\" is wired", "--legacy", "--yes"):
        assert part in help_text, part
    bad = run("", "--nope")
    assert bad.returncode == 2 and "unknown option: --nope" in bad.stderr
    assert run("", "--symbol").returncode == 2  # a flag without its value
