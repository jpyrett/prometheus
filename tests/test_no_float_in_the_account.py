"""A test that reads the source (design choices entry 112; design/decimal_money_spec.md section 6): there is no float in the account's code.

Run:  pytest tests/test_no_float_in_the_account.py -q

It reads every module on the ACCOUNT'S side (spec section 3: the common money code, the account book, the Account Manager, the Trader, RaiseFunds, the desk, the session, the page's bridge, the prompts that print money) and fails when it finds, OUTSIDE
the allow-list below, a use of the name `float` (a call `float(...)`, an annotation `: float` or `-> float`, `type=float`), a call of `round(` (money and prices are rounded by `cents`, `price_scale`, `trade_amount`, counts by `shares`), or a float literal.
Each exception is named by its file and the function (or `<module>`) it lives in, with the reason: it is what keeps "no float in the account's code" true after this change. The MEASUREMENTS (the Advisor, the Critic, the Data Clerk's analytics, `rag`,
the ledger, `labs`) are not read: converting them is later work (Jacob, 2026-11-16).
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_SIDE = [
    "prometheus/common/money.py", "prometheus/common/portfolio.py", "prometheus/common/shares.py", "prometheus/common/user_account.py", "prometheus/common/work_order.py",
    "prometheus/persistence/accounts.py",
    "prometheus/tools/trader.py", "prometheus/tools/trading_math.py", "prometheus/reasoning/raise_funds.py",
    "prometheus/langgraph/desk.py", "prometheus/application/session.py", "prometheus/application/recommendations.py",
    *sorted(str(p.relative_to(ROOT)) for pattern in ("prometheus/agents/account_manager/*.py", "prometheus/agents/trader/*.py", "prometheus/application/web/*.py", "prometheus/prompts/account_manager*.py", "prometheus/prompts/trader.py")
            for p in ROOT.glob(pattern) if p.name != "__init__.py"),
]

# (file, enclosing function, or "<module>" for a module-level line) -> why a float, a round or a float literal may stay there. A class method is named by its function name.
TIMING = "a timing or a wait (seconds), not money"
SCORE = "the evaluator's score of a plan (0 to 100) and its components (fractions between 0 and 1): a measurement, never added to money (spec section 3, last paragraph); a ratio of two Decimal amounts is worked out from them at the very end (`_ratio`)"
RETURN = "a return: a fraction worked out from Decimal prices at the very end (spec section 3, last paragraph)"
DOOR = "a door that takes the int or float `json` gave and turns it into a Decimal by its written digits (spec D4), or the JSON number type of the encoder"
TEMPERATURE = "a model's temperature: a setting of the model, not money"
ADVISOR = "the Advisor's or the Critic's own measurement (a score, a weight, a relevance): not the account's money (spec D6)"
ALLOWED = {
    "prometheus/common/money.py": {"_number": DOOR, "exact": DOOR, "json_number": "THE JSON encoder: `float(d)`, checked by `Decimal(repr(f)) == d`, is the one place a Decimal becomes a float (spec section 4)", "<module>": "the pydantic type of a JSON number: `int | float`"},
    "prometheus/common/shares.py": {"_decimal": DOOR, "<module>": "the pydantic type of a JSON number: `int | float`"},
    "prometheus/common/portfolio.py": {"<module>": "the pydantic type of a JSON number: `int | float`"},
    "prometheus/common/user_account.py": {"UserAccount": "the holder's investment horizon in YEARS (not money)", "_plain_decimal": "a door that REFUSES a float"},
    "prometheus/persistence/accounts.py": {"load_profile": "the profile's horizon in years and its minimum price: the Advisor's and the Critic's inputs (spec D6), not the account's money", "_user": "the holder's horizon in years"},
    "prometheus/agents/account_manager/account_manager.py": {"top_recommendation": ADVISOR, "__init__": TIMING + " (the RaiseFunds search deadline)", "main": TIMING + " (`--deadline`)"},
    "prometheus/agents/account_manager/clarify.py": {"<module>": "the pydantic type of a JSON number: `int | float`"},
    "prometheus/agents/account_manager/intent.py": {"parse_intent": DOOR + ": a number in the model's reply is checked before it is a Decimal"},
    "prometheus/agents/account_manager/neutral.py": {"neutral_candidates": ADVISOR + " (the sort key: the Advisor's score)"},
    "prometheus/agents/trader/config.py": {"Limits": TIMING + " (`model_timeout_seconds`)"},
    "prometheus/agents/trader/execute.py": {"execute": TEMPERATURE},
    "prometheus/agents/trader/record.py": {"Outcome": RETURN, "Run": TEMPERATURE},
    "prometheus/agents/trader/report.py": {"_pct": RETURN},
    "prometheus/agents/trader/simulator.py": {"_fraction": RETURN, "_benchmark": RETURN, "_return_difference": RETURN},
    "prometheus/agents/trader/trader.py": {"<module>": TEMPERATURE},
    "prometheus/tools/trader.py": {"loggable": "the model's own tool call, recorded as it made it", "place_order": "the model's tool argument: the number it wrote is a float here and a Decimal at the first line (the price door)"},
    "prometheus/reasoning/raise_funds.py": {"<module>": SCORE, "_ratio": SCORE, "Node": SCORE, "root": SCORE, "score_plan": SCORE, "green_component": SCORE, "progress": SCORE, "priority": SCORE,
                                             "build_graph": TIMING + " (the search's deadline and the seconds of a level)", "evaluate_and_prune": TIMING + " (the seconds of a level)", "search": TIMING, "answer": TIMING},
    "prometheus/langgraph/desk.py": {"node": TIMING + " (the seconds a node took)", "report": TIMING},
    "prometheus/application/recommendations.py": {"<module>": "the research part of the runner: the retriever's relevance bar (a measurement)", "phrase_relevance": "the retriever's relevance (a measurement)", "Result": TIMING + " and the retriever's relevance (a measurement)",
                                                  "timed": TIMING},
    "prometheus/application/web/__main__.py": {"build_parser": TIMING + " (the replay's pace, the question timeout)"},
    "prometheus/application/web/app.py": {"create_app": TIMING + " (the question timeout)", "stream": TIMING + " (the event stream's wait)"},
    "prometheus/application/web/desk_run.py": {"<module>": TIMING, "callables": TIMING, "run_live": TIMING, "_finish": TIMING},
    "prometheus/application/web/emitter.py": {"node_line": TIMING, "finished": TIMING, "stages": TIMING},
    "prometheus/application/web/replay.py": {"<module>": TIMING + " (the replay's own pacing)", "__init__": TIMING, "play": TIMING, "_pause": TIMING},
    "prometheus/application/web/runs.py": {"<module>": TIMING, "emit": TIMING + " (the event's timestamp)", "read_from": TIMING, "ask": TIMING},
    "prometheus/application/web/views.py": {"stage_states": TIMING + " (the seconds a stage took)", "_score_build": ADVISOR + " (the weight of a dimension of the score, as a percentage)"},
}


def findings(path: str) -> list[tuple[str, int, str, str]]:
    """(file, line, the function it is in, what) for every use of `float`, every call of `round(` and every float literal in the file."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    out: list[tuple[str, int, str, str]] = []

    def walk(node, scope: str):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else scope
            if isinstance(child, ast.Name) and child.id == "float":
                out.append((path, child.lineno, scope, "the name `float`"))
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "round":
                out.append((path, child.lineno, scope, "a call of `round(`"))
            elif isinstance(child, ast.Constant) and isinstance(child.value, float):
                out.append((path, child.lineno, scope, f"the float literal {child.value!r}"))
            walk(child, inner)

    walk(tree, "<module>")
    return out


def allowed(file: str, scope: str) -> bool:
    here = ALLOWED.get(file, {})
    return scope in here or ("<module>" in here and scope == "<module>")


def test_the_modules_of_the_accounts_side_are_all_read():
    """The list is the spec's section 3: a module that is not on it is not read, so a new module of the account's side must be added to it."""
    missing = [p for p in ACCOUNT_SIDE if not (ROOT / p).is_file()]
    assert not missing, f"not found: {missing}"
    for folder in ("prometheus/agents/account_manager", "prometheus/agents/trader", "prometheus/application/web"):
        assert all(f"{folder}/{p.name}" in ACCOUNT_SIDE for p in (ROOT / folder).glob("*.py") if p.name != "__init__.py"), f"a module of {folder} is not read"
    assert len(ACCOUNT_SIDE) == len(set(ACCOUNT_SIDE)) and len(ACCOUNT_SIDE) >= 40


@pytest.mark.parametrize("file", ACCOUNT_SIDE)
def test_no_float_round_or_float_literal_outside_the_allow_list(file):
    found = [f for f in findings(file) if not allowed(file, f[2])]
    assert not found, "\n".join(f"{p}:{n} in {scope}: {what}" for p, n, scope, what in found) + "\n(money, prices and share counts are Decimal: convert at the door, once; or add the exception to ALLOWED with its reason)"


def test_every_exception_of_the_allow_list_is_used():
    """An exception that nothing needs any more is removed: the list says what is left, and only that."""
    used = {(f[0], f[2]) for file in ACCOUNT_SIDE for f in findings(file)}
    stale = [(file, scope) for file, scopes in ALLOWED.items() for scope in scopes if (file, scope) not in used]
    assert not stale, f"no float, round or float literal is in these any more: {stale}"
    assert all(file in ACCOUNT_SIDE for file in ALLOWED)
    assert all(reason.strip() for scopes in ALLOWED.values() for reason in scopes.values())


def test_the_measurements_are_not_read_here():
    """Spec section 3: the Advisor, the Critic, the Data Clerk's analytics, `rag` and the ledger stay floats and are not part of this change."""
    assert not [p for p in ACCOUNT_SIDE if p.startswith(("prometheus/agents/advisor", "prometheus/agents/critic", "prometheus/agents/data_clerk", "prometheus/rag", "prometheus/memory", "labs"))]
