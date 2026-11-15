"""What the user session and the desk's request flow do that needs no agent, no model and no data (design choices entry 78): the flags, the messages when the account book or a key is missing, the cap on how
many companies are scored, and the shape of the graph. Everything with agents is in `tests/test_end_to_end_live.py` (real agents, real data, opt-in); there is no stand-in agent and no invented price here.
"""
import re

import pytest

from prometheus.application import recommendations as rec
from prometheus.application import session as ss
from prometheus.langgraph import desk

PERSON = ["--name", "Ada Lovelace", "--age", "41", "--risk", "moderate", "--cash", "15000", "--mode", "auto", "--yes"]

# Real oil and gas tickers and companies, in the order a theme search can find them (the first live session of 2026-11-14 found 18 listed companies for "oil")
OIL_NAMES = {"XOM": "Exxon Mobil Corp", "CVX": "Chevron Corp", "COP": "ConocoPhillips", "EOG": "EOG Resources Inc", "OXY": "Occidental Petroleum Corp", "VLO": "Valero Energy Corp",
             "PSX": "Phillips 66", "MPC": "Marathon Petroleum Corp", "SLB": "Schlumberger NV", "HAL": "Halliburton Co", "DVN": "Devon Energy Corp", "FANG": "Diamondback Energy Inc",
             "APA": "APA Corp", "CVE": "Cenovus Energy Inc", "CNQ": "Canadian Natural Resources Ltd", "BP": "BP plc", "HES": "Hess Corp", "MRO": "Marathon Oil Corp"}


def found(symbols, private=()):
    rows = [{"brand": s, "parent": None, "symbol": s, "company": OIL_NAMES[s], "public": True, "how": "company name", "note": None} for s in symbols]
    rows += [{"brand": p, "parent": None, "symbol": None, "company": None, "public": False, "how": None, "note": "privately held: not scored"} for p in private]
    return rows


def result_with(companies):
    result = rec.Result(rec.Request("phrase", "oil", phrase="oil"), "industry", rec.date(2026, 11, 13), 3)
    result.companies = companies
    return result


# ---------------------------------------------------------------- how many companies are scored

def test_the_cap_is_the_advisors_and_the_data_clerks_own_limit():
    from prometheus.agents.advisor.config import load_config
    from prometheus.agents.data_clerk.data_clerk_agent import MAX_REFRESH_SYMBOLS
    assert rec.MAX_SCORED == MAX_REFRESH_SYMBOLS == load_config().limits.max_candidates == 15


def test_a_theme_that_finds_fewer_than_the_cap_scores_them_all_in_the_order_found_and_leaves_out_what_is_not_listed():
    result = result_with(found(["XOM", "CVX", "COP"], private=["Some Private Refiner"]))
    assert rec.scored_symbols(result) == ["XOM", "CVX", "COP"] and result.notes == []


def test_a_broad_theme_scores_the_first_fifteen_and_says_which_were_left_out():
    """The first live session found 18 listed companies for "oil"; the paid refresh accepts 15 and the Advisor scores 15: it crashed the desk before this."""
    result = result_with(found(list(OIL_NAMES)))
    assert len(result.companies) == 18 and rec.scored_symbols(result) == list(OIL_NAMES)[:15]
    assert result.notes == ["18 listed companies were found: the first 15 are scored (the Advisor's limit); not scored: BP, HES, MRO"]


def test_exactly_fifteen_is_not_a_cut():
    result = result_with(found(list(OIL_NAMES)[:15]))
    assert rec.scored_symbols(result) == list(OIL_NAMES)[:15] and result.notes == []


# ---------------------------------------------------------------- the flags, refused before anything is created

@pytest.mark.parametrize("argv, message", [
    (["--prompt", "x"], "a new user needs --name, --age, --risk, --cash, --mode"),
    (["--prompt", "x", "--name", "A", "--age", "17", "--risk", "moderate", "--cash", "100", "--mode", "auto"], "age"),
    (["--prompt", "x", "--name", "A", "--age", "40", "--risk", "moderate", "--cash", "abc", "--mode", "auto"], "amount"),
    (["--prompt", "x", "--name", "A", "--age", "40", "--risk", "moderate", "--cash", "-5", "--mode", "auto"], "amount"),
    (["--prompt", "x", "--name", "A", "--age", "40", "--risk", "low", "--cash", "100", "--mode", "auto"], "invalid choice"),
    (["--prompt", "x", "--name", "A", "--age", "40", "--risk", "moderate", "--cash", "100", "--mode", "manual"], "invalid choice"),
    (["--prompt", "x", "--account", "5", "--name", "A"], "do not give --name"),
    (["--prompt", "x", "--account", "5", "--max-position-pct", "2"], "--max-position-pct must be above 0 and at most 1"),
    (["--prompt", "x", "--account", "5", "--auto-min-score", "101"], "--auto-min-score must be from 0 to 100"),
    (["--prompt", "x", "--account", "5", "--default-count", "11"], "--default-count must be from 1 to 10"),
    ([], "--prompt"),
])
def test_a_bad_flag_is_refused_before_anything_is_created(argv, message, capsys):
    with pytest.raises(SystemExit) as stop:
        ss.main(argv, parts=object())                                  # the flags are refused before the parts are touched
    assert stop.value.code == 2 and message in capsys.readouterr().err


def test_without_the_account_book_set_up_the_command_says_how(monkeypatch):
    def missing(args):
        raise FileNotFoundError("/x/secrets/postgres_local.txt does not exist (run scripts/setup_postgres.sh)")
    monkeypatch.setattr(ss, "build_parts", missing)
    lines = []
    assert ss.main(["--prompt", "x", *PERSON], out=lines.append) == 2
    assert "bash scripts/setup_postgres.sh" in "\n".join(lines)


def test_a_missing_key_is_a_message_that_names_the_setting(monkeypatch):
    def nokey(args):
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env; see .env.example)")
    monkeypatch.setattr(ss, "build_parts", nokey)
    lines = []
    assert ss.main(["--prompt", "x", *PERSON], out=lines.append) == 2 and "OPENROUTER_API_KEY is not set" in lines[0]


# ---------------------------------------------------------------- the session is the real thing, and knows nothing

def test_the_real_session_is_built_from_the_real_agents_and_nothing_from_tests():
    """`build_parts` is the only place a session gets its agents: it must name the real classes and nothing from tests/."""
    import inspect
    source = inspect.getsource(ss.build_parts)
    for real in ("DataClerk(", "Advisor(", "Critic(", "Trader(", "AccountManager(", "open_account_store()"):
        assert real in source, real
    assert "tests" not in source.lower() and "fake" not in source.lower() and "scripted" not in source.lower() and "double" not in source.lower()


def test_the_session_starts_the_langgraph_desk_it_does_not_run_the_procedural_runner():
    import inspect
    body = inspect.getsource(ss.run_session)
    assert "build_desk(" in body and "graph.stream(" in body and "recommendations.run" not in body and "import run" not in body


def test_the_session_gives_the_account_manager_the_users_screen_because_the_user_talks_to_it_and_to_nobody_else():
    """Entry 82: a trade that failed is shown to the user by the Account Manager, at once, through the session's own `out`: the desk is given a manager factory that sets `screen`."""
    import inspect
    body = inspect.getsource(ss.run_session)
    assert "manager.screen = out" in body and "funds_factory=new_manager" in body and "funds_factory=parts.new_manager" not in body


def test_nothing_in_the_session_knows_a_sector_a_company_or_a_ticker():
    import inspect
    source = inspect.getsource(ss).lower()
    for word in ("oil", "xom", "chevron", "exxon", "energy", "bob jones", "aapl", "msft", "dpz"):
        assert not re.search(rf"\b{word}\b", source), word


# ---------------------------------------------------------------- the graph is the same graph

def test_the_graph_has_no_node_the_other_requests_do_not_have():
    nodes = lambda **o: set(desk.build_desk(clerk_factory=lambda order: None, advisor_factory=lambda order: None, **o).get_graph().nodes)
    assert nodes(with_critic=True, with_trader=True) == {"__start__", "account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader", "__end__"}
    assert nodes() == {"__start__", "account_manager", "data_clerk", "advisor", "__end__"}


def test_a_request_whose_account_cannot_be_opened_stops_the_desk_at_its_first_node_with_the_reason():
    """The profile file is missing: the Account Manager node reports it before any model is asked or any data is read."""
    from prometheus.agents.account_manager.account_manager import AccountManager
    from prometheus.common.portfolio import Portfolio
    state = desk.run_desk({"ask": {"text": "I want to buy oil stocks"}, "account": {"profile": "/nonexistent/profile.json", "portfolio": Portfolio(user="Jacob Pyrett", cash_avail=1.0)}},
                          clerk_factory=lambda order: None, advisor_factory=lambda order: None, with_critic=True, funds_factory=lambda: AccountManager(model=object(), model_id="account-book"))
    assert state["order"] is None and state["plan"] is None and state["path"] == ["account_manager"] and "the profile file does not exist" in state["errors"][0]
