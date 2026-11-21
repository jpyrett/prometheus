"""Capture the REAL replies of the Account Manager's model to the questions RaiseFunds asks it for the plans the tests of design choices entry 142 need (design/direct_raise_funds_spec.md), so that a test can replay them offline.

    python scripts/capture_raise_funds_plans.py                 # says what it would do and stops
    python scripts/capture_raise_funds_plans.py --yes           # does it

WHAT IT CALLS (so it asks for --yes): the Account Manager's model (openai/gpt-oss-120b on OpenRouter, the account_manager role) for each question the thought generator of RaiseFunds asks in four scenarios, a few cents in all; it reads the
recorded quotes of account 6553 (`tests/fixtures/raise_funds/account_6553.json`) and the stored closes of Friday 2026-11-13, calls no Finnhub, fetches nothing, and reads Postgres only for THROWAWAY accounts it creates (the copies of 6553's cash and
holdings that the tests make) and deletes at the end: account 6553 itself, the three samples and the kept paper accounts are never touched. It writes ONLY `tests/fixtures/raise_funds/plans.json`: {the sha256 of the question (the messages, with
their roles): the model's reply}, unedited. The four scenarios are the ones of `tests/test_raise_funds_direct.py` and `tests/test_raise_funds_recommendation.py`: Jacob's "Buy 2 shares of ANT and raise funds if needed", the same with a second stock,
the same in dollars, and the shares typed with a real recommendation (the real review of 2026-11-14, the sample portfolio). A question that is already recorded is not asked again.
"""
import argparse
import json
import os
import sys
from contextlib import suppress
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import raise_funds_helpers as H  # noqa: E402

from prometheus.rag.config import REPO_ROOT  # noqa: E402

SCENARIOS = (("Buy 2 shares of ANT and raise funds if needed", None), ("Buy 2 shares of ANT and 1 share of GOOGL and raise funds if needed", None), ("Buy $200 of ANT and raise funds if needed", None),
             # entry 143: the user's own sale in the same request (S2: RaiseFunds works from the account once it is filled), and an account holding $8.00 whose typed purchase is $347.64 short (B1: the sales may fall short)
             ("Sell 1 share of MCD and buy 5 shares of ANT and raise funds if needed", None), ("Buy 4 shares of ANT and raise funds if needed", Decimal("8.00")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--yes", action="store_true", help="make the calls (the Account Manager's model on OpenRouter) and write the fixture")
    args = parser.parse_args()
    have = json.loads(H.PLANS_FILE.read_text(encoding="utf-8")) if H.PLANS_FILE.exists() else {"replies": {}}
    print(f"{len(have['replies'])} questions already recorded -> {H.PLANS_FILE.relative_to(REPO_ROOT)}")
    if not args.yes:
        print("(this asks the Account Manager's model the questions that are not recorded: a few cents; run with --yes)")
        return
    for name, value in (("ALLOW_NEUTRAL_RECOMMENDATIONS", "false"), ("ALWAYS_RAISE_FUNDS", "false"), ("IGNORE_MARKET_HOURS", "false"), ("PAPER_TRADING", "true"), ("USE_LIVE_TRADING", "true"), ("LANGSMITH_TRACING", "false")):
        os.environ[name] = value      # what the tests' conftest gives every test: nothing here may depend on Jacob's .env
    from prometheus.agents.account_manager.account_manager import AccountManager
    from prometheus.persistence.accounts import open_account_store

    live = AccountManager().model      # the Account Manager's own model, as the desk builds it
    store, made, recorded = open_account_store(), [], dict(have["replies"])
    try:
        for typed, cash in SCENARIOS:
            account_id = H.make_account(store, made, f"Capture {typed[:20]}", cash=cash)
            model = H.Model(typed, recorded=recorded, live=live)
            state = H.run_desk(store, account_id, typed, approver=H.Cards(funding=True, approve=True), model=model)
            order = state["order"]
            print(f"  {typed!r}: path {state['path']}, trades {[(t['trade_id'], t['symbol'], t['side'], t['status']) for t in order.trades]}, {model.calls} model calls")
        waiting_funded(store, made, recorded, live)
        waiting_funded(store, made, recorded, live, cash=Decimal("0.00"))      # an account with no free cash at all: the whole price is raised. Both before `recommendation_plan`, which switches the clerk and the environment to the stored closes of the recommendation path
        recommendation_plan(recorded, live)
    finally:
        for account_id in made:
            with suppress(ValueError):
                store.delete_account(account_id)
    H.PLANS_FILE.parent.mkdir(parents=True, exist_ok=True)
    note = have.get("_note") or ("The REAL replies of the Account Manager's model (openai/gpt-oss-120b on OpenRouter) to the questions the thought generator of RaiseFunds asked in the scenarios of design choices entry 142, recorded "
                                 + date.today().isoformat() + " by scripts/capture_raise_funds_plans.py: {the sha256 of the conversation (every message with its role): the model's reply}. Nothing here is edited.")
    H.PLANS_FILE.write_text(json.dumps({"_note": note, "replies": recorded}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written: {H.PLANS_FILE.relative_to(REPO_ROOT)}; {len(recorded)} questions")


def waiting_funded(store, made: list, recorded: dict, live, cash: Decimal | None = None) -> None:
    """Entry 143, W-F: "Buy 2 shares of ANT at $85 or less and raise funds if needed" waits (ANT is $88.91), then the watcher meets ANT at its recorded day low ($83.05) on the Monday and fills it: RaiseFunds runs for the shortfall at THAT price."""
    import tempfile
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    from prometheus.agents.account_manager.orders import Watcher

    account_id = H.make_account(store, made, "Capture waiting funded", mode="auto", cash=cash)
    model = H.Model(H.WAITING, recorded=recorded, live=live)
    state = H.run_desk(store, account_id, H.WAITING, approver=H.Cards(), model=model)
    with tempfile.TemporaryDirectory() as audit:
        watcher = Watcher(store, clerk=H.clerk(H.FALLEN), trader_factory=lambda: H.trader(), model=model, accounts=[account_id], audit_path=Path(audit) / "audit.jsonl")
        events = watcher.tick(datetime(2026, 11, 23, 11, 0, tzinfo=ZoneInfo("America/New_York")))
    print(f"  {H.WAITING!r}: decisions {[(d['will_trade'], d.get('open_order_id')) for d in state['order'].trade_decisions]}, events {[e['type'] for e in events]}, {model.calls} model calls")


def recommendation_plan(recorded: dict, live) -> None:
    """R10: the shares typed with the real review of 2026-11-14 (XOM and EOG approved), the sample portfolio ($51.23 and MSFT, TSLA, GM), the stored closes of Friday 2026-11-13: ONE plan raises the whole shortfall."""
    import handshake_helpers as hh

    os.environ["USE_LIVE_TRADING"] = "false"
    from prometheus.agents.data_clerk import data_clerk_agent

    data_clerk_agent.market_day = lambda *a, **k: hh.FRIDAY      # what the tests' `friday_close` fixture does
    model = H.Model(None, recorded=recorded, live=live)
    am, order, portfolio = H.sample_manager(model), H.reviewed_order(), hh.jacob_as_bob()
    am.decide_trades(order, portfolio, None, hh.MOMENT, "auto", shares_of={"XOM": 8, "EOG": 8}, raise_funds=True)
    funds = am.fund_decisions(order, portfolio, hh.MOMENT)
    print(f"  the recommendation path: {funds['status']}, {[(x['symbol'], x['shares']) for x in (funds['plan'] or {}).get('sells', [])]}, {model.calls} model calls")
    capped, portfolio = H.reviewed_order(), hh.jacob_as_bob()      # the same request with a budget under what the typed shares cost: the pot is the budget, so less is bought and less is raised
    am.decide_trades(capped, portfolio, None, hh.MOMENT, "auto", budget=H.BUDGET, shares_of={"XOM": 8, "EOG": 8}, raise_funds=True)
    funds = am.fund_decisions(capped, portfolio, hh.MOMENT)
    print(f"  with a budget of {H.BUDGET}: {funds['status']}, {[(x['symbol'], x['shares']) for x in (funds['plan'] or {}).get('sells', [])]}, {model.calls} model calls in all")
    mixed, portfolio = H.reviewed_order(), hh.jacob_as_bob()      # entry 143, P1: only XOM has typed shares (EOG, approved, has none): the typed shares are paid for by ONE plan and EOG gets nothing
    am.decide_trades(mixed, portfolio, None, hh.MOMENT, "auto", shares_of={"XOM": 8}, raise_funds=True)
    funds = am.fund_decisions(mixed, portfolio, hh.MOMENT)
    print(f"  only XOM typed: {funds['status']}, {[(x['symbol'], x['shares']) for x in (funds['plan'] or {}).get('sells', [])]}, {model.calls} model calls in all")


if __name__ == "__main__":
    main()
