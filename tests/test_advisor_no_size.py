"""The Advisor recommends no amount and does not use the user's money (design choices entry 80). REAL DATA: the three captures of `tests/fixtures/advisor_real` (SNDK, PG, PG_open: the real work orders, the real
metrics, the real replies of the real Advisor model) run through the real Advisor code, and the real session capture `tests/fixtures/session_real/bob_jones_oil.json`.

    STRUCTURE   the Advisor's source (the package, its two prompt files and its settings) names no cash, no equity, no number of shares, no risk budget and no position percentage, imports nothing of the user's
                profile, the Account Manager or the account book, and IS GIVEN NO PORTFOLIO (entry 105: it imports no `Portfolio`, takes none, and names no holding): a test, like the project's import tests, so it
                fails when someone puts one back
    BEHAVIOUR   a record has no size field and no holding field, the real thesis replies of the real model pass the program's checks on a record that has no size (the digest the model reads has none, and the
                numbers the check allows have none), and no word of what the model is told is about a holding or a sale
"""
import ast
import copy
import json
from datetime import date

import pytest
from advisor_helpers import MemoryLedger, Scripted
from pydantic import ValidationError
from test_advisor_sndk_pg import FIXTURES, RecordedClerk, recorded_plan_config

from prometheus.agents.advisor import thesis as th
from prometheus.agents.advisor.advisor import Advisor
from prometheus.agents.advisor.calls import parse_list
from prometheus.agents.advisor.config import AdvisorConfig
from prometheus.agents.advisor.record import Concentration, TradePlan
from prometheus.common import work_order as wo
from prometheus.rag.config import REPO_ROOT

ADVISOR_DIR = REPO_ROOT / "prometheus" / "agents" / "advisor"
ADVISOR_FILES = [*sorted(ADVISOR_DIR.glob("*.py")), REPO_ROOT / "prometheus" / "prompts" / "advisor_v1.py", REPO_ROOT / "prometheus" / "prompts" / "advisor.py"]
SETTINGS = REPO_ROOT / "configs" / "advisor_v1.json"
# Names that mean the user's money or a size. Whole identifiers, anywhere in the source, comments and docstrings included: the words of the design are written around them ("no risk budget").
BANNED_ANYWHERE = ("cash_avail", "suggested_shares", "risk_budget", "position_pct", "equity_basis", "equity_of", "funds_feasible", "funds_note", "funds_check", "size_note", "oversized", "risk_pct")
SIZE_FIELDS = {"equity", "equity_basis", "risk_budget", "suggested_shares", "position_pct", "oversized", "size_note", "funds_feasible", "funds_note"}
NOT_THE_ADVISORS = ("prometheus.common.user_profile", "prometheus.agents.account_manager", "prometheus.persistence", "prometheus.tools.trading_math", "prometheus.agents.critic", "prometheus.agents.trader")


def docstring_nodes(tree: ast.AST) -> set[int]:
    """The ids of the string nodes that are docstrings (prose about the code, not code)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            out.add(id(node.body[0].value))
    return out


def identifiers(tree: ast.AST) -> list[str]:
    """Every name the code uses: variables, attributes, arguments, keywords, definitions, imports, and the string constants that are not docstrings (a dict key, a field name)."""
    skip, out = docstring_nodes(tree), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
        elif isinstance(node, ast.arg):
            out.append(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            out.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(node.name)
        elif isinstance(node, ast.alias):
            out.append(node.asname or node.name.split(".")[-1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            out.append(node.value)
    return out


# ---------------------------------------------------------------- structure

def test_the_advisors_source_names_no_cash_no_equity_no_share_count_no_budget_and_no_position_percentage():
    assert len(list(ADVISOR_DIR.glob("*.py"))) >= 14 and len(ADVISOR_FILES) >= 16 and all(p.exists() for p in ADVISOR_FILES)           # the whole package, trade_plan.py and record.py included, and the two prompt files
    problems = []
    for path in ADVISOR_FILES:
        text = path.read_text(encoding="utf-8")
        problems += [f"{path.name}: {word}" for word in BANNED_ANYWHERE if word in text]
        names = identifiers(ast.parse(text))
        problems += [f"{path.name}: the name {n!r}" for n in names if n == "equity"]                      # `negative_equity` and `debt_to_equity` are a company's balance sheet, not the user's money (and `cash` in families.py is the company's cash balance: the user's is `cash_avail`, above)
    assert problems == []


def test_the_advisor_is_given_no_portfolio_it_imports_none_takes_none_and_names_no_holding():
    """AMENDED (entry 105). This test said the Advisor reads nothing of the portfolio but its `positions`. It is given no portfolio at all: no import of `Portfolio` or of `prometheus.common.portfolio`, no name `portfolio`
    or `positions` anywhere in the code (an argument, a variable, an attribute), none of the names the holdings pass used (`holding_action`, `add_floor`), and no string constant that is an action of a holding
    (`reduce`, `hold`, `avoid`). The retired KEYS of a version 2 record (`held`, `held_shares`, `cost_basis`, `unrealized_pct`, `portfolio_fit`) are named in `record.py` alone: its readers of version 2 drop them
    (`add` is read in `ranking.cooldown_of` too: an OLD ledger line is about what was recommended before, not about what is held)."""
    banned_names = {"portfolio", "Portfolio", "PortfolioFit", "load_portfolio", "positions", "holding_action", "add_floor"}
    retired = {"held", "held_shares", "cost_basis", "unrealized_pct", "portfolio_fit"}
    problems = []
    for path in ADVISOR_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = identifiers(tree)
        problems += [f"{path.name}: the name {n!r}" for n in names if n in banned_names]
        problems += [f"{path.name}: the retired key {n!r}" for n in names if n in retired and path.name != "record.py"]
        skip = docstring_nodes(tree)
        strings = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip]
        problems += [f"{path.name}: the string {n!r}" for n in strings if n in {"reduce", "hold", "avoid"}]
        problems += [f"{path.name}: the string 'add'" for n in strings if n == "add" and path.name != "ranking.py"]
        for node in ast.walk(tree):
            names_in = [node.module or "", *(f"{node.module}.{a.name}" for a in node.names)] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            problems += [f"{path.name}: imports {n}" for n in names_in if "portfolio" in n.lower()]
    assert problems == []


def test_the_advisor_imports_nothing_of_the_users_profile_the_account_manager_or_the_account_book():
    for path in ADVISOR_FILES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [node.module or "", *(f"{node.module}.{a.name}" for a in node.names)] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            assert not [n for n in names if n.startswith(NOT_THE_ADVISORS)], (path.name, names)


def test_the_advisors_settings_hold_no_size_and_the_loader_expects_none():
    def keys(node):
        return {k for k, v in node.items()} | {x for v in node.values() if isinstance(v, dict) for x in keys(v)} if isinstance(node, dict) else set()

    found = keys(json.loads(SETTINGS.read_text(encoding="utf-8")))
    assert not found & {"risk_pct", "risk_pct_stress", "max_position_pct", "add_max_position_pct", "oversized_pct"}, found
    assert set(AdvisorConfig.model_fields["trade"].annotation.model_fields) == {"entry_atr", "stop_atr", "target_edge_multiple", "min_reward_to_risk", "trailing_families", "trend_limit_atr", "event_limit_reaction_atr"}      # the two between are Advisor config version 3's exit conventions (a target pays 2 for 1 risked; the trend family trails): prices, never a size
    assert set(AdvisorConfig.model_fields["ranking"].annotation.model_fields) == {"diversity_within"}


def test_the_record_has_no_field_for_a_size_a_budget_or_the_users_money():
    assert not SIZE_FIELDS & set(TradePlan.model_fields) and set(Concentration.model_fields) == {"note"}        # entry 105: `portfolio_fit` and its two numbers that were never computed are gone
    for field, value in (("suggested_shares", 5), ("equity", 1000.0), ("position_pct", 0.1), ("risk_budget", 10.0), ("funds_feasible", True)):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TradePlan(entry_basis="last_close", **{field: value})


# ---------------------------------------------------------------- the real captures through the real Advisor

def capture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def run(name: str, plan_reply=None, changes=None, config_path=None):
    """The Advisor on the captured inputs, as `test_advisor_sndk_pg.replay` does it (no portfolio is given), and, for a test that needs it, ONE real metric changed. `config_path`: the Advisor's settings (when left out, the plan the recorded replies were written for: `recorded_plan_config`)."""
    fx = capture(name)
    metrics = copy.deepcopy(fx["metrics"])
    for (symbol, metric), value in (changes or {}).items():
        metrics[symbol][metric]["value"] = value
    replies = fx["replies"] if plan_reply is None else [plan_reply, *(r for r in fx["replies"] if '"theses"' in r)]
    order = wo.WorkOrder.model_validate(fx["order"])
    Advisor(model=Scripted(*replies), clerk=RecordedClerk(metrics), ledger=MemoryLedger(), config_path=config_path or recorded_plan_config()).generate_recommendations(order, "funds", date.fromisoformat(fx["today"]))
    return order, fx


def test_a_red_flag_on_the_real_captured_company_stops_it_as_a_candidate_there_is_no_reduction():
    """AMENDED (entry 105). This test changed ONE real number of the held GM (its cash flow is negative while it shows a profit: a red flag) and the holdings pass wrote a `reduce` with an entry zone and no amount. The Advisor
    knows no holding: the same change to the company the capture is about (SNDK) stops it at the red_flag gate, it is a candidate with the reason, and there is no record, so nothing is recommended to buy or to sell."""
    order, _ = run("SNDK", changes={("SNDK", "cfo_negative_with_profit"): True})
    assert order.recommendations == [] and [(c["symbol"], c["gate"]) for c in order.candidates] == [("SNDK", "red_flag")]
    assert "held" not in order.candidates[0] and order.candidates[0]["reason"].startswith("gate red_flag failed")


# ---------------------------------------------------------------- the thesis

def real_thesis_items(name: str) -> list[dict]:
    """The items of the real model's thesis reply in the capture (the last call that wrote the theses: the one the Advisor accepted, after a retry when the first was refused), as the Advisor parsed them."""
    reply = next(r for r in reversed(capture(name)["replies"]) if '"theses"' in r)
    return parse_list(reply, "theses")


@pytest.mark.parametrize("name", ["SNDK", "PG_open"])
def test_the_real_models_thesis_passes_the_checks_on_a_record_with_no_size(name):
    order, _ = run(name)
    symbol = "PG" if name.startswith("PG") else name
    record = next(r for r in order.recommendations if r["symbol"] == symbol)
    assert record["thesis"]["thesis_error"] is None and record["thesis"]["thesis"]                                  # the Advisor accepted it on the new record
    item = next(i for i in real_thesis_items(name) if i["symbol"] == symbol)
    dumped = {k: v for k, v in record.items()}
    assert th.check(item, dumped) == []                                                                              # and so does the check, called on its own
    digest = th.digest(dumped)
    assert "suggested" not in digest and "oversized" not in digest and "of equity" not in digest and "position " not in digest and "shares" not in digest.replace("share_count_change", "")
    assert "Trade: strategy" in digest and "stop " in digest and "time stop " in digest and "order " in digest


def test_the_thesis_prompt_tells_the_model_there_is_no_amount_and_the_system_text_names_no_size():
    from prometheus.prompts.advisor_v1 import THESIS_SYSTEM

    assert "no amount to buy either" in THESIS_SYSTEM and "do not say how many shares or how much money" in THESIS_SYSTEM
    assert "suggested" not in THESIS_SYSTEM.lower() and "position size" not in THESIS_SYSTEM.lower()


# ---------------------------------------------------------------- the report

def test_the_runners_report_prints_no_size_for_a_recommendation():
    """The report of `application/recommendations.py` has no `size    :` line any more: its source does not read a size from the record, so no record can make it print one."""
    source = (REPO_ROOT / "prometheus" / "application" / "recommendations.py").read_text(encoding="utf-8")
    assert not [w for w in ("suggested_shares", "position_pct", "size_note", "funds_feasible", "of equity", "size    :") if w in source]
    assert "plan    : entry" in source and "time stop" in source                                                      # the plan lines stay
