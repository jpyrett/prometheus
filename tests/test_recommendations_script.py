"""Tests for ./test_recommendations.sh: what it asks, the kinds of request it accepts and the command it builds. Run with bash and PROMETHEUS_DRY_RUN=1: it prints the command it WOULD run, so
there is no network, no key, no cost, and nothing is fetched or written. Answers are piped in, the way a person would type them."""
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "test_recommendations.sh"
MODULE = ["python", "-m", "prometheus.application.recommendations"]
OSS = ["--model", "openai/gpt-oss-120b"]   # the script's default Advisor model


def run(stdin: str = "", *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], input=stdin, capture_output=True, text=True, cwd=REPO, env={**os.environ, "PROMETHEUS_DRY_RUN": "1"}, timeout=30)


def command(result: subprocess.CompletedProcess) -> list[str]:
    line = next(line for line in result.stdout.splitlines() if line.startswith("DRY RUN:"))
    return shlex.split(line.removeprefix("DRY RUN:"))


def test_the_script_is_executable_and_valid_bash():
    assert os.access(SCRIPT, os.X_OK) and subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0


@pytest.mark.parametrize("request_text", ["Pizza", "DPZ", "DPZ PZZA YUM", "DPZ,PZZA", "Domino's Pizza", "weight loss drugs", "$DPZ"])
def test_a_symbol_a_list_a_company_name_or_a_phrase_is_passed_on_as_typed(request_text):
    result = run("", request_text, "--yes", "--find", "5")
    assert result.returncode == 0, result.stderr
    assert command(result) == [*MODULE, request_text, "--find", "5", *OSS, "--fetch"]


def test_with_no_request_it_asks_what_to_look_at_how_many_and_whether_to_fetch_and_the_defaults_are_five_and_yes():
    result = run("Pizza\n\n\n")
    assert result.returncode == 0, result.stderr
    prompts = result.stderr
    order = [prompts.index(p) for p in ("What to look at", "How many top recommendations", "Refresh the data")]
    assert order == sorted(order) and "[5]" in prompts and "a search phrase such as Pizza" in prompts and "PAID APIs" in prompts and "[Y/n]" in prompts
    assert command(result) == [*MODULE, "Pizza", "--find", "5", *OSS, "--fetch"]


def test_answering_n_to_the_paid_fetch_leaves_the_flag_out():
    result = run("DPZ\n3\nn\n")
    assert command(result) == [*MODULE, "DPZ", "--find", "3", *OSS] and "no fetch (the stored data is used)" in result.stdout


def test_a_script_nobody_is_watching_never_spends_the_paid_apis_by_default():
    result = run("", "Pizza", "--find", "5")                       # the input ends at the question
    assert "no answer: not fetching" in result.stderr and "--fetch" not in command(result)
    assert "--fetch" not in command(run("", "Pizza", "--find", "5", "--no-fetch", "--yes")) and "--fetch" in command(run("", "Pizza", "--find", "5", "--yes"))   # --no-fetch wins


def test_a_blank_request_is_asked_again_and_when_the_input_ends_it_stops():
    result = run("\n\nDPZ\n5\nn\n")
    assert result.returncode == 0 and result.stderr.count("What to look at") == 3 and "Give a symbol, a list of symbols, a company name or a search phrase." in result.stderr
    ended = run("")
    assert ended.returncode == 2 and "Give a symbol" in ended.stderr


@pytest.mark.parametrize("bad", ["Pizza; rm -rf /", "../etc/passwd", "a|b", "`id`", "x" * 121, "$(id)"])
def test_a_request_that_could_be_a_command_or_a_path_is_refused(bad):
    result = run("", bad, "--yes")
    assert result.returncode == 2 and "not a valid request" in result.stderr and "DRY RUN" not in result.stdout


def test_the_kind_can_be_forced_with_a_flag_and_the_flag_replaces_the_text():
    assert command(run("", "--phrase", "AI memory shortage", "--yes", "--find", "4")) == [*MODULE, "--phrase", "AI memory shortage", "--find", "4", *OSS, "--fetch"]
    assert command(run("", "--symbols", "DPZ PZZA", "--no-fetch", "--find", "2")) == [*MODULE, "--symbols", "DPZ PZZA", "--find", "2", *OSS]
    assert command(run("", "--symbol", "DPZ", "--no-fetch", "--find", "1"))[3:5] == ["--symbols", "DPZ"]                      # --symbol is the same as --symbols
    assert command(run("", "--company", "Domino's Pizza", "--no-fetch", "--find", "1"))[3:5] == ["--company", "Domino's Pizza"]
    assert command(run("", "--term", "Pizza", "--no-fetch", "--find", "1"))[3:5] == ["--phrase", "Pizza"]                       # --term, as in test_prometheus.sh
    both = run("", "Pizza", "--phrase", "Pizza", "--yes")
    assert both.returncode == 2 and "give the request once" in both.stderr
    two = run("", "Pizza", "Pasta", "--yes")
    assert two.returncode == 2 and "give the request once, in quotes" in two.stderr


def test_the_other_options_reach_the_command():
    result = run("", "Pizza", "--yes", "--find", "3", "--case", "theme", "--sector", "xly", "--cash", "1000", "--from-days-ago", "30", "--today", "2026-11-05", "--model", "some/model", "--effort", "medium", "--providers", "groq", "--save", "out.json")
    assert command(result) == [*MODULE, "Pizza", "--find", "3", "--case", "theme", "--sector", "XLY", "--cash", "1000", "--from-days-ago", "30", "--today", "2026-11-05", "--model", "some/model", "--effort", "medium", "--providers", "groq", "--save", "out.json", "--fetch"]
    assert command(run("", "Pizza", "--yes", "--find", "3", "--portfolio", "configs/portfolios/jacob_pyrett_sndk_mu.json"))[6:8] == ["--portfolio", "configs/portfolios/jacob_pyrett_sndk_mu.json"]


@pytest.mark.parametrize("args, message", [
    (("--find", "0"), "not a whole number from 1 to 20"), (("--find", "21"), "not a whole number from 1 to 20"), (("--find", "x"), "not a whole number from 1 to 20"),
    (("--find", "5", "--case", "market"), "--case is funds, industry or theme"), (("--find", "5", "--from-days-ago", "91"), "--from-days-ago is a whole number from 1 to 90"),
    (("--find", "5", "--sector", "XLY;ls"), "--sector is a fund ticker"), (("--find", "5", "--cash", "abc"), "--cash is a number"), (("--find", "5", "--cash", "10", "--portfolio", "p.json"), "not both"),
    (("--find", "5", "--effort", "max"), "--effort is low, medium or high"), (("--find", "5", "--providers", "a;b"), "--providers is a comma separated list"), (("--find", "5", "--model", "x y"), "--model is an OpenRouter model id"),
    (("--find", "5", "--bogus"), "unknown option: --bogus"), (("--find",), "--find needs a value"),
])
def test_bad_options_are_refused_with_the_reason(args, message):
    result = run("", "Pizza", "--yes", *args)
    assert result.returncode == 2 and message in result.stderr and "DRY RUN" not in result.stdout


def test_help_documents_every_kind_of_request_and_the_cost():
    text = run("", "--help").stdout
    for needle in ("a SEARCH PHRASE", "ONE SYMBOL", "a LIST of symbols", "a COMPANY NAME", "NOT SCORED", "ChromaDB", "WHAT IT COSTS", "paid APIs", "--find N"):
        assert needle in text, needle


def test_the_advisor_model_is_gpt_oss_120b_by_default_and_advisor_means_the_env_model():
    assert command(run("", "Pizza", "--yes", "--find", "5"))[6:8] == OSS and "Advisor model: openai/gpt-oss-120b" in run("", "Pizza", "--yes", "--find", "5").stdout
    env = run("", "Pizza", "--yes", "--find", "5", "--model", "advisor")
    assert "--model" not in command(env) and "ADVISOR_MODEL of .env" in env.stdout


def test_a_list_to_compare_with_commas_and_quotes_is_passed_on_and_can_be_forced():
    text = 'MCD, YUM, "Burger King"'
    assert command(run("", text, "--yes", "--find", "3")) == [*MODULE, text, "--find", "3", *OSS, "--fetch"]
    assert command(run("", "--list", "MCD, YUM, Burger King", "--no-fetch", "--find", "3")) == [*MODULE, "--list", "MCD, YUM, Burger King", "--find", "3", *OSS]
    assert "give the request once" in run("", text, "--list", "A, B", "--yes").stderr
    assert "a COMPANY NAME" in run("", "--help").stdout and "list      items separated by commas" in run("", "--help").stdout


def test_no_ledger_is_passed_on_and_is_off_by_default():
    assert "--no-ledger" not in command(run("", "Pizza", "--yes", "--find", "3"))
    assert command(run("", "Pizza", "--yes", "--find", "3", "--no-ledger"))[-2:] == ["--no-ledger", "--fetch"]


# ---------------------------------------------------------------- the Critic's flags (Critic v1, entries 56 to 59)

def test_the_critic_runs_by_default_and_its_flags_reach_the_command():
    assert "--no-critic" not in command(run("", "PG", "--yes", "--no-fetch"))
    result = run("", "PG", "--no-fetch", "--mode", "auto", "--now", "2026-11-05 10:30", "--profile", "configs/profiles/jacob_pyrett.json", "--critic-model", "z-ai/glm-5.3", "--critic-effort", "max", "--no-ask")
    assert command(result) == [*MODULE, "PG", "--find", "5", *OSS, "--profile", "configs/profiles/jacob_pyrett.json", "--mode", "auto", "--now", "2026-11-05 10:30", "--critic-model", "z-ai/glm-5.3",
                               "--critic-effort", "max", "--no-ask"]
    assert "Critic: z-ai/glm-5.3, mode auto" in result.stdout and "the Critic approves or denies each" in result.stdout


def test_no_critic_leaves_it_out_and_says_so():
    result = run("", "PG", "--no-fetch", "--no-critic")
    assert command(result)[-1] == "--no-critic" and "no Critic" in result.stdout


@pytest.mark.parametrize("args, message", [
    (("--mode", "manual"), "--mode is ask or auto"), (("--now", "tomorrow"), "--now is a US Eastern moment"), (("--now", "2026-11-05T10:30"), "--now is a US Eastern moment"),
    (("--critic-model", "x y"), "--critic-model is an OpenRouter model id"), (("--critic-effort", "extreme"), "--critic-effort is low, high or max"), (("--profile", "p;rm -rf"), "--profile is the path"),
    (("--no-critic", "--mode", "auto"), "go with the Critic: drop --no-critic"), (("--no-critic", "--no-ask"), "go with the Critic: drop --no-critic"), (("--no-critic", "--critic-effort", "low"), "go with the Critic")])
def test_bad_critic_options_are_refused_with_the_reason(args, message):
    result = run("", "PG", "--yes", "--no-fetch", *args)
    assert result.returncode == 2 and message in result.stderr and "DRY RUN" not in result.stdout


def test_help_documents_the_critic_and_what_it_costs():
    out = run("", "--help").stdout
    assert "THE CRITIC" in out and "z-ai/glm-5.3" in out and "--no-critic" in out and "NOTHING IS BOUGHT OR SOLD" in out and "ASK mode" in out
