"""Replay a saved work order through the Trader (Trader v1 spec section 16).

    python -m prometheus.agents.trader.replay --order data/processed/desk_PG.json --mode backtest --no-model        # FREE: stored bars, no model, no paid call, nothing written

    --order FILE          a saved work order: the desk's saved JSON (it has an "order" key) or a bare order
    --mode paper|backtest (default backtest)
    --portfolio FILE | --cash N      --profile FILE
    --now "YYYY-MM-DD HH:MM"         US Eastern; default: the latest decided_utc of the permitted decisions (the moment the permission was given)
    --no-model            the sweep only: no model at all
    --yes-funding         answer yes to the funding question (a replay has nobody to ask)
    --json                --save FILE
    --fresh               ignore the trades the order already carries and start from its decisions (a replay of an order a desk run already traded)
    --no-ledger           paper mode only: give the Trader a throwaway ledger instead of data/ledger/trades.jsonl

THE ONE COMMAND THAT IS FREE: `--mode backtest --no-model` reads the stored daily bars (`data/historical`, read only) and prints. It calls no model and no paid API, and writes nothing (a backtest has no ledger and
never books). Everything else costs: a run with the model calls OpenRouter, and paper mode calls Finnhub for the quotes (and appends to the trades ledger unless `--no-ledger`).

THIS IS WIRING, so it is the one Trader module that may import the Data Clerk (rule T5 and its test cover every other). It builds the Account Manager with `clerk=DataClerk(model=object())`: the bars and the quotes are
code, that Data Clerk's own model is never called and no key is needed for it. The Account Manager's own model is `object()` as well: the Trader only uses its portfolio, price and notify methods. The Trader reaches the
Data Clerk through the Account Manager, as it does in the desk.

MONEY IS `decimal.Decimal` (design choices entry 112): `--cash` is the dollars you typed read into a Decimal to the cent once, at the edge (more than two decimals is refused), the order read from a saved file keeps the numbers `json` gave (floats; every reader goes
through a door), and `--json` and `--save` write the one encoder's numbers.
"""
import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

from prometheus.agents.account_manager.account_manager import DEFAULT_PORTFOLIO, DEFAULT_PROFILE, AccountManager
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.agents.trader.report import trade_section
from prometheus.agents.trader.trader import NullLedger, Trader
from prometheus.common.money import dumps, money
from prometheus.common.portfolio import Portfolio, load_portfolio
from prometheus.common.user_profile import load_profile
from prometheus.common.work_order import TRADING, WorkOrder, parse_utc

EASTERN = ZoneInfo("America/New_York")


def load_order(path: str | Path) -> tuple[WorkOrder, dict | None]:
    """(the order, the RaiseFunds answer the desk saved with it or None) from a saved JSON file: the desk's (an "order" key beside "funds") or a bare order. Raises ValueError, with the reason, when it cannot be read."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path}: there is no such file") from None
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: cannot be read as JSON ({type(error).__name__})") from None
    saved = isinstance(data, dict) and isinstance(data.get("order"), dict)
    try:
        order = WorkOrder.model_validate(data["order"] if saved else data)
    except Exception as error:   # pydantic's ValidationError: its text names the fields
        raise ValueError(f"{path}: not a work order ({str(error).splitlines()[0] if str(error) else type(error).__name__})") from None
    funds = data.get("funds") if saved and isinstance(data.get("funds"), dict) else None
    return order, funds


def moment_of(text: str) -> datetime:
    """"2026-11-05 10:30" (US Eastern) as an aware datetime. Raises ValueError for anything else."""
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=EASTERN)
    except ValueError:
        raise ValueError(f'--now must be written "YYYY-MM-DD HH:MM" (US Eastern), not {text!r}') from None


def default_now(order: WorkOrder) -> datetime:
    """The moment the permission was given: the latest `decided_utc` of the decisions that permit a trade. Raises ValueError when no decision permits one (there is nothing to replay)."""
    moments = [m for d in order.trade_decisions if d.get("will_trade") in TRADING and (m := parse_utc(d.get("decided_utc"))) is not None]
    if not moments:
        raise ValueError("no decision of this order permits a trade (yes_auto or yes_user) with a decided_utc: give --now, or there is nothing to replay")
    return max(moments)


def start_fresh(order: WorkOrder) -> WorkOrder:
    """`--fresh`: forget the trades and the runs the order carries, so that the Trader starts from the decisions. A `traded` order is `decided` again; nothing else of the order is touched."""
    order.trades, order.trader_runs = [], []
    if order.status == "traded":
        order.status = "decided"
    return order


def _dollars(text: str) -> Decimal:
    """argparse: the amount the user typed read into a Decimal of dollars ONCE, at the edge (`money.money`: at most two decimals, never a float)."""
    try:
        return money(Decimal(text.strip().replace(",", "")))
    except (InvalidOperation, ValueError):
        raise argparse.ArgumentTypeError(f"{text!r} is not an amount of dollars (a number with at most two decimals)") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m prometheus.agents.trader.replay", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--order", required=True, help='a saved work order: the desk\'s saved JSON (it has an "order" key) or a bare order')
    parser.add_argument("--mode", choices=("paper", "backtest"), default="backtest", help="backtest (default): the stored daily bars, free. paper: the current quotes, which COSTS (Finnhub)")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--portfolio", help=f"the portfolio JSON file (default: {DEFAULT_PORTFOLIO.name} of configs/portfolios)")
    where.add_argument("--cash", type=_dollars, help="a portfolio with no holdings and this much cash (dollars, at most two decimals)")
    parser.add_argument("--profile", help=f"the user's profile JSON file (default: {DEFAULT_PROFILE.name} of configs/profiles)")
    parser.add_argument("--now", help='the moment of the run, US Eastern: "YYYY-MM-DD HH:MM" (default: the latest decided_utc of the permitted decisions)')
    parser.add_argument("--no-model", action="store_true", help="no model at all: the program places the default orders (free in backtest mode)")
    parser.add_argument("--yes-funding", action="store_true", help="answer yes to the funding question: a replay has nobody to ask")
    parser.add_argument("--json", action="store_true", help="print the order as JSON instead of the report")
    parser.add_argument("--save", metavar="FILE", help='write the order and the funding answer as JSON ({"order": ..., "funds": ...}, the desk\'s own shape) to this file')
    parser.add_argument("--fresh", action="store_true", help="ignore the trades the order already carries and start from its decisions")
    parser.add_argument("--no-ledger", action="store_true", help="paper mode: do not append to data/ledger/trades.jsonl")
    return parser


def main(argv: list[str] | None = None, clerk=None, trader=None, out=None) -> int:
    """Replay. `clerk` is the Data Clerk the Account Manager asks for prices (the bars-only one, `DataClerk(model=object())`, when left out; a fake in tests), `trader` a Trader to use instead of the one the flags
    describe (tests), `out` a writer (print when left out). Returns 0 when the Trader ran (the order is `traded`, `decided` or `reviewed`), 1 when it refused to, 2 for an argument or a file that is not usable."""
    say = out or print
    args = build_parser().parse_args(argv)
    try:
        order, funding = load_order(args.order)
        profile = load_profile(args.profile) if args.profile else None
        manager = AccountManager(model=object(), clerk=clerk or DataClerk(model=object()), profile=profile or DEFAULT_PROFILE)
        portfolio = load_portfolio(args.portfolio or DEFAULT_PORTFOLIO) if args.cash is None else Portfolio(user=manager.profile.user, cash_avail=args.cash)
        now = moment_of(args.now) if args.now else default_now(order)
    except (ValueError, OSError) as error:
        print(f"replay: {error}", file=sys.stderr)
        return 2
    if args.fresh:
        start_fresh(order)
    trader = trader or Trader(mode=args.mode, use_model=not args.no_model, ledger=NullLedger() if args.no_ledger else None)
    approver = (lambda card: True) if args.yes_funding else None
    order = manager.perform_trades(order, portfolio, trader, now, funding=funding, approver=approver, book=False)
    if args.save:
        Path(args.save).write_text(dumps({"order": order.model_dump(mode="json"), "funds": funding}, indent=1), encoding="utf-8")
    if args.json:
        say(dumps({"order": order.model_dump(mode="json"), "funds": funding}, indent=1))
    else:
        say("\n".join(trade_section(order)))
    run = order.trader_runs[-1] if order.trader_runs else None
    return 0 if run is not None and run.get("portfolio_before") is not None and order.status in ("traded", "decided", "reviewed") else 1


if __name__ == "__main__":
    sys.exit(main())
