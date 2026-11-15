"""The Advisor recommends no amount and does not use the user's money (design choices entry 80). REAL DATA: the three captures of `tests/fixtures/advisor_real` (SNDK, PG, PG_open: the real work orders, the real
metrics, the real replies of the real Advisor model) run through the real Advisor code, and the real session capture `tests/fixtures/session_real/bob_jones_oil.json`.

    STRUCTURE   the Advisor's source (the package, its two prompt files and its settings) names no cash, no equity, no number of shares, no risk budget and no position percentage, imports nothing of the user's
                profile, the Account Manager or the account book, and reads of the portfolio nothing but the positions: a test, like the project's import tests, so it fails when someone puts one back
    BEHAVIOUR   the same records come out whatever the user holds in cash, a record has no size field, a reduction carries no amount, and the real thesis replies of the real model pass the program's checks
                on a record that has no size (the digest the model reads has none, and the numbers the check allows have none)
"""
import ast
import copy
import json
from datetime import date

import pytest
from advisor_helpers import MemoryLedger, Scripted
from pydantic import ValidationError
from test_advisor_sndk_pg import FIXTURES, RecordedClerk, scrub

from prometheus.agents.advisor import thesis as th
from prometheus.agents.advisor.advisor import Advisor
from prometheus.agents.advisor.calls import parse_list
from prometheus.agents.advisor.config import AdvisorConfig
from prometheus.agents.advisor.record import PortfolioFit, Recommendation, TradePlan
from prometheus.common import work_order as wo
from prometheus.common.portfolio import load_portfolio
from prometheus.rag.config import REPO_ROOT

ADVISOR_DIR = REPO_ROOT / "prometheus" / "agents" / "advisor"
ADVISOR_FILES = [*sorted(ADVISOR_DIR.glob("*.py")), REPO_ROOT / "prometheus" / "prompts" / "advisor_v1.py", REPO_ROOT / "prometheus" / "prompts" / "advisor.py"]
SETTINGS = REPO_ROOT / "configs" / "advisor_v1.json"
# Names that mean the user's money or a size. Whole identifiers, anywhere in the source, comments and docstrings included: the words of the design are written around them ("no risk budget").
BANNED_ANYWHERE = ("cash_avail", "suggested_shares", "risk_budget", "position_pct", "equity_basis", "equity_of", "funds_feasible", "funds_note", "funds_check", "size_note", "oversized", "risk_pct", "max_adv_pct")
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


def test_the_advisor_reads_nothing_of_the_portfolio_but_its_positions():
    seen = set()
    for path in ADVISOR_FILES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "portfolio":
                seen.add(node.attr)
    assert seen == {"positions"}, seen


def test_the_advisor_imports_nothing_of_the_users_profile_the_account_manager_or_the_account_book():
    for path in ADVISOR_FILES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [node.module or "", *(f"{node.module}.{a.name}" for a in node.names)] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            assert not [n for n in names if n.startswith(NOT_THE_ADVISORS)], (path.name, names)


def test_the_advisors_settings_hold_no_size_and_the_loader_expects_none():
    def keys(node):
        return {k for k, v in node.items()} | {x for v in node.values() if isinstance(v, dict) for x in keys(v)} if isinstance(node, dict) else set()

    found = keys(json.loads(SETTINGS.read_text(encoding="utf-8")))
    assert not found & {"risk_pct", "risk_pct_stress", "max_position_pct", "max_adv_pct", "add_max_position_pct", "oversized_pct"}, found
    assert set(AdvisorConfig.model_fields["trade"].annotation.model_fields) == {"entry_atr", "stop_atr", "target_edge_multiple", "limit_spread_bps", "trend_limit_atr", "event_limit_reaction_atr"}
    assert set(AdvisorConfig.model_fields["ranking"].annotation.model_fields) == {"diversity_within"}


def test_the_record_has_no_field_for_a_size_a_budget_or_the_users_money():
    assert not SIZE_FIELDS & set(TradePlan.model_fields) and not {"position_pct_after", "funds_feasible"} & set(PortfolioFit.model_fields)
    assert set(PortfolioFit.model_fields) == {"sector_weight_after", "correlation_to_portfolio_63d", "note"}
    for field, value in (("suggested_shares", 5), ("equity", 1000.0), ("position_pct", 0.1), ("risk_budget", 10.0), ("funds_feasible", True)):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TradePlan(entry_basis="last_close", **{field: value})


# ---------------------------------------------------------------- the real captures through the real Advisor

def capture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def run(name: str, cash=None, plan_reply=None, changes=None):
    """The Advisor on the captured inputs, as `test_advisor_sndk_pg.replay` does it, with the user's cash set to `cash` (the capture's own portfolio when None) and, for a test that needs it, ONE real metric changed."""
    fx = capture(name)
    metrics = copy.deepcopy(fx["metrics"])
    for (symbol, metric), value in (changes or {}).items():
        metrics[symbol][metric]["value"] = value
    portfolio = load_portfolio(REPO_ROOT / fx["portfolio"])
    if cash is not None:
        portfolio = portfolio.model_copy(update={"cash_avail": cash})
    replies = fx["replies"] if plan_reply is None else [plan_reply, *(r for r in fx["replies"] if '"theses"' in r)]
    order = wo.WorkOrder.model_validate(fx["order"])
    Advisor(model=Scripted(*replies), clerk=RecordedClerk(metrics), ledger=MemoryLedger()).generate_recommendations(order, portfolio, "funds", date.fromisoformat(fx["today"]))
    return order, fx


@pytest.mark.parametrize("name", ["SNDK", "PG", "PG_open"])
def test_no_real_record_holds_a_size_and_each_validates_as_a_record(name):
    order, _ = run(name)
    assert order.status == "recommended" and order.recommendations
    for record in order.recommendations:
        Recommendation.model_validate(record)
        assert set(record["trade"]) == set(TradePlan.model_fields) and not SIZE_FIELDS & set(record["trade"])
        assert set(record["portfolio_fit"]) == set(PortfolioFit.model_fields)
        assert "suggested_shares" not in json.dumps(record) and "position_pct" not in json.dumps(record) and "risk_budget" not in json.dumps(record)       # not in any section, not even as text


@pytest.mark.parametrize("name, plan_reply", [("SNDK", None), ("PG_open", None), ("SNDK", json.dumps({"companies": []}))], ids=["SNDK", "PG_open", "SNDK-no-model-evidence"])
def test_the_advisor_gives_the_same_records_whatever_the_user_has_in_cash(name, plan_reply):
    """The cash is never read: no cash, the sample's $51.23 and a billion dollars give the very same recommendations and candidates, record for record (the last case has a held name recommended as an add)."""
    results = []
    for cash in (0.0, None, 1_000_000_000.0):
        order, _ = run(name, cash=cash, plan_reply=plan_reply)
        results.append(([scrub(r) for r in order.recommendations], order.candidates))
    assert results[0] == results[1] == results[2] and results[0][0]


def test_a_reduction_has_an_entry_zone_and_no_amount_on_the_real_capture():
    """The real held GM with ONE real number changed (its cash flow is negative while it shows a profit: a red flag): the holdings pass reduces it. The record is a reduction with a price zone and nothing about how much."""
    order, fx = run("SNDK", changes={("GM", "cfo_negative_with_profit"): True})
    gm = next(r for r in order.recommendations if r["symbol"] == "GM")
    assert gm["action"]["action"] == "reduce" and "red_flag gate failed" in gm["action"]["reason"] and gm["action"]["held_shares"] == 60 and order.recommendations[0]["symbol"] == "GM"   # a reduction first
    Recommendation.model_validate(gm)
    t = gm["trade"]
    assert t["entry_zone"] and t["entry_mid"] and t["family"] is None and t["stop_price"] is None and t["stop_distance"] is None and t["target_price"] is None and t["invalidation"] == []
    assert not SIZE_FIELDS & set(t) and "held_shares" in gm["action"] and set(t) == set(TradePlan.model_fields)
    assert all(not SIZE_FIELDS & set(r["trade"]) for r in order.recommendations)


def test_an_add_is_a_plan_of_prices_and_the_amount_is_left_to_the_account_manager():
    order, _ = run("SNDK", plan_reply=json.dumps({"companies": []}))
    gm = next(r for r in order.recommendations if r["symbol"] == "GM")
    assert gm["action"]["action"] == "add" and gm["trade"]["stop_price"] and gm["trade"]["entry_zone"] and not SIZE_FIELDS & set(gm["trade"])


# ---------------------------------------------------------------- the thesis

def real_thesis_items(name: str) -> list[dict]:
    """The items of the real model's thesis reply in the capture (the call that wrote the theses), as the Advisor parsed them."""
    reply = next(r for r in capture(name)["replies"] if '"theses"' in r)
    return parse_list(reply, "theses")


@pytest.mark.parametrize("name", ["SNDK", "PG_open"])
def test_the_real_models_thesis_passes_the_checks_on_a_record_with_no_size(name):
    order, _ = run(name)
    record = order.recommendations[0]
    assert record["thesis"]["thesis_error"] is None and record["thesis"]["thesis"]                                  # the Advisor accepted it on the new record
    item = real_thesis_items(name)[0]
    dumped = {k: v for k, v in record.items()}
    assert th.check(item, dumped) == []                                                                              # and so does the check, called on its own
    digest = th.digest(dumped)
    assert "suggested" not in digest and "oversized" not in digest and "of equity" not in digest and "position " not in digest and "shares" not in digest.replace("share_count_change", "")
    assert "Trade: strategy" in digest and "stop " in digest and "time stop " in digest and "order " in digest


def test_the_thesis_may_not_name_a_number_the_record_does_not_have_so_it_cannot_propose_an_amount():
    order, _ = run("PG_open")
    record = order.recommendations[0]
    allowed = th.allowed_numbers(record)
    amount = next(n for n in range(2, 5000) if float(n) not in allowed and not any(abs(round(a, 0) - n) < 1e-9 for a in allowed))
    item = {**real_thesis_items("PG_open")[0], "thesis": f"Buy {amount} shares of PG on its capital return."}
    reasons = th.check(item, record)
    assert any(f"{amount}" in reason and "not in the record" in reason for reason in reasons)


def test_the_thesis_prompt_tells_the_model_there_is_no_amount_and_the_system_text_names_no_size():
    from prometheus.prompts.advisor_v1 import THESIS_SYSTEM

    assert "no amount to buy or sell" in THESIS_SYSTEM and "do not say how many shares or how much money" in THESIS_SYSTEM
    assert "suggested" not in THESIS_SYSTEM.lower() and "position size" not in THESIS_SYSTEM.lower()


# ---------------------------------------------------------------- the report

def test_the_runners_report_prints_no_size_for_a_recommendation():
    """The report of `application/recommendations.py` has no `size    :` line any more: its source does not read a size from the record, so no record can make it print one."""
    source = (REPO_ROOT / "prometheus" / "application" / "recommendations.py").read_text(encoding="utf-8")
    assert not [w for w in ("suggested_shares", "position_pct", "size_note", "funds_feasible", "of equity", "size    :") if w in source]
    assert "plan    : entry" in source and "time stop" in source                                                      # the plan lines stay
