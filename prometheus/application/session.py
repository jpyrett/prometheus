"""A user session, end to end (design choices entry 78): a person, an account with money, a request in their own words, and what the desk did with it.

    python -m prometheus.application.session --name "Ada Lovelace" --age 41 --risk moderate --cash 10000 --mode auto \\
        --prompt "I want to buy <anything>. Buy the most shares from the 2 most recommended companies and split the money evenly."
    ./test_prometheus.sh --session          # the same, with prompts for each answer

What happens, in the order a real user would meet it:

  1. ACCOUNT     a new user and profile are created in the Postgres account book (name, age, risk tolerance, Ask or Auto) and money is added to the account. (`--account N` uses an existing one.)
  2. REQUEST     the Account Manager's model reads what was typed into a form (`AccountManager.understand_request`: a theme, tickers or company names; how many to buy from; an even split; a budget), code checks
                 every field against the user's own words, and the person is shown what was understood before anything runs.
  3. THE DESK    the LangGraph workflow of `prometheus/langgraph/desk.py`, started with the user's words: Account Manager -> Data Clerk -> Advisor -> Critic -> Account Manager's decision -> Trader, on ONE
                 work order. The Account Manager node reads the request; the Data Clerk node finds the companies and (with `--fetch`: PAID) refreshes their data; the Advisor node ranks them; the Critic reviews each
                 recommendation against the profile; the decision node trades what the Critic approves above the mark (Auto) or asks (Ask) and spreads the money evenly when asked; the Trader executes against a
                 SIMULATED broker. Every fill is booked in the account. This module only starts the graph, shows each node as it finishes, and prints.
  4. REPORTS     the status of the recommendations, the Trader's results, and the account's portfolio before and after.

Nothing here knows a sector, a company or a ticker: it is the same for any request the form can hold. Nothing is sent to a broker. A session that was meant to trade but left the portfolio as it was is a FAILED
check (exit 1) with the reasons, because a sanity check that cannot trade has found something; `--allow-no-trade` accepts that.

EXIT CODES: 0 the session did what was asked (the portfolio changed, or it was only a look); 1 it could not trade (the reasons are printed); 2 a bad flag, or the account book, the keys or the request could not be used.
"""
import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from prometheus.common.portfolio import Portfolio
from prometheus.common.trading_clock import clock_note
from prometheus.common.user_account import clean_age, clean_display_name, clean_risk_level, clean_trade_mode, parse_money
from prometheus.common.user_profile import TOLERANCES
from prometheus.langgraph.desk import build_desk

DEFAULT_COUNT = 3
RULE = "=" * 118


@dataclass
class Parts:
    """The pieces of a session: the real ones (`build_parts`), or fakes in a test."""
    store: object
    clerk: object
    advisor: object
    critic: object
    trader: object
    new_manager: Callable               # () -> an AccountManager that is not connected to an account: the desk connects it, and the session connects its own to create the account
    approver: Callable | None = None    # asked about a trade that needs the user (Ask mode, an Auto approval under the mark, the sells of a funding plan); None: the session's own, which asks at the terminal (`approver_for`)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A user session end to end: create a user and an account, add money, read a request in the user's own words, run the desk, and report the recommendations, the trades and the portfolio.")
    who = parser.add_argument_group("the user and the account (a new one is created, unless --account)")
    who.add_argument("--name", help="the user's name")
    who.add_argument("--age", type=int, help="the user's age (18 to 120): older users are held to a lower risk by the Critic")
    who.add_argument("--risk", choices=TOLERANCES, help="the risk tolerance: conservative, moderate or aggressive")
    who.add_argument("--cash", help="dollars to add to the new account, for example 15000")
    who.add_argument("--mode", choices=("ask", "auto"), help="ask: the user approves each trade. auto: the desk trades what the Critic approves with a high score")
    who.add_argument("--account", help="use an EXISTING account (an account number or a sample key) instead of creating one: no user is created and no money is added")
    parser.add_argument("--prompt", required=True, help="what the user wants, in their own words")
    desk = parser.add_argument_group("the desk")
    desk.add_argument("--fetch", action="store_true", help="REFRESH the companies' data first with the PAID APIs (bars, news, press releases, filings: writes files); without it the stored data is used")
    desk.add_argument("--default-count", type=int, default=DEFAULT_COUNT, help=f"how many of the top companies to buy from when the request does not say (default {DEFAULT_COUNT})")
    desk.add_argument("--model", default="openai/gpt-oss-120b", help="the Advisor's OpenRouter model id (default openai/gpt-oss-120b)")
    desk.add_argument("--no-trader-model", action="store_true", help="the Trader without a model: the program places the default orders")
    desk.add_argument("--real-ledger", action="store_true", help="use the real ledgers (data/ledger/): a company recommended before is then hidden by its cooldown. The default is throwaway ledgers, so a session can be repeated")
    limits = parser.add_argument_group("overrides of the profile's limits, for this session only (the account book keeps its own)")
    limits.add_argument("--auto-min-score", type=int, help="Auto mode: the lowest Critic score that trades without asking (the profile's own, else the Critic's 80)")
    limits.add_argument("--max-position-pct", type=float, help="the most one stock may be of the account's equity, 0 to 1 (the profile's 0.10)")
    limits.add_argument("--max-exposure-pct", type=float, help="the most all the purchases of one review may add, of equity, 0 to 1 (the profile's 0.30)")
    limits.add_argument("--max-loss-pct", type=float, help="the most a stop-out of one trade may lose, of equity, 0 to 1 (the profile's 0.01)")
    run = parser.add_argument_group("the session")
    run.add_argument("--yes", "-y", action="store_true", help="do not ask to confirm what was understood (in Ask mode you are STILL asked to approve each trade: --yes never approves a trade)")
    run.add_argument("--allow-no-trade", action="store_true", help="exit 0 even when the portfolio did not change")
    run.add_argument("--save", metavar="FILE", help="write what the desk made (the request, what was understood, and the work order without its data: recommendations, reviews, decisions, trades) to this JSON file: a real capture")
    run.add_argument("--cleanup", action="store_true", help="delete the new account at the end (a sample or an --account account is never deleted)")
    return parser


def check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict:
    """The person, cleaned. Calls `parser.error` (exit 2) for what makes no sense."""
    try:
        if args.account:
            if any(v is not None for v in (args.name, args.age, args.risk, args.cash, args.mode)):
                parser.error("--account uses an existing account: do not give --name, --age, --risk, --cash or --mode with it")
            person = {}
        else:
            missing = [f"--{k}" for k in ("name", "age", "risk", "cash", "mode") if getattr(args, k) is None]
            if missing:
                parser.error(f"a new user needs {', '.join(missing)} (or use --account for an existing one)")
            person = {"name": clean_display_name(args.name), "age": clean_age(args.age), "risk": clean_risk_level(args.risk), "cash": parse_money(args.cash), "mode": clean_trade_mode(args.mode)}
        if not 1 <= args.default_count <= 10:
            parser.error("--default-count must be from 1 to 10")
        for name in ("max_position_pct", "max_exposure_pct", "max_loss_pct"):
            value = getattr(args, name)
            if value is not None and not 0 < value <= 1:
                parser.error(f"--{name.replace('_', '-')} must be above 0 and at most 1")
        if args.auto_min_score is not None and not 0 <= args.auto_min_score <= 100:
            parser.error("--auto-min-score must be from 0 to 100")
    except ValueError as error:
        parser.error(str(error))
    return person


# ---------------------------------------------------------------- the pieces

def build_parts(args: argparse.Namespace) -> Parts:
    """The real thing: the Postgres account book, the Data Clerk, the Advisor, the Critic and the Trader. Raises FileNotFoundError, OSError, RuntimeError or ValueError, with a message that names the setting and never a value."""
    from prometheus.agents.account_manager.account_manager import AccountManager
    from prometheus.agents.advisor.advisor import Advisor
    from prometheus.agents.critic.critic import Critic
    from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
    from prometheus.agents.trader.trader import NullLedger, Trader
    from prometheus.application.recommendations import throwaway_ledger
    from prometheus.persistence.accounts import open_account_store

    store = open_account_store()
    clerk = DataClerk(allow_fetch=args.fetch)
    advisor = Advisor(model_id=args.model, clerk=clerk, ledger=None if args.real_ledger else throwaway_ledger(), effort="auto", providers="auto")
    critic = Critic(clerk=clerk, ledger=None if args.real_ledger else throwaway_ledger(), history=advisor.ledger)
    trader = Trader(mode="paper", use_model=not args.no_trader_model, ledger=None if args.real_ledger else NullLedger())
    return Parts(store=store, clerk=clerk, advisor=advisor, critic=critic, trader=trader, new_manager=lambda: AccountManager(clerk=clerk))      # the approver is made by `run_session`, from its own ask and out


def open_account(parts: Parts, args: argparse.Namespace, person: dict) -> int:
    """The account number: a new user and profile with the money added, or the existing account asked for."""
    if args.account:
        return int(args.account) if args.account.isdigit() else parts.store.account_id_for_seed(args.account)
    account_id = parts.store.create_account(person["name"], person["risk"], trade_mode=person["mode"], age=person["age"])
    banker = parts.new_manager()
    banker.connect(parts.store, account_id)
    banker.add_funds(Decimal(person["cash"]))
    return account_id


def with_limits(profile, args: argparse.Namespace):
    """The profile with this session's overrides (the account book is not changed)."""
    changes = {k: v for k, v in (("auto_min_score", args.auto_min_score), ("max_position_pct", args.max_position_pct), ("max_new_exposure_pct", args.max_exposure_pct), ("max_loss_per_trade_pct", args.max_loss_pct)) if v is not None}
    return profile.model_copy(update=changes) if changes else profile


# ---------------------------------------------------------------- the reports

def money(value) -> str:
    return "n/a" if value is None else f"${float(value):,.2f}"


def portfolio_lines(portfolio: Portfolio) -> list[str]:
    lines = [f"  cash      {money(portfolio.cash_avail):>14}"]
    if not portfolio.positions:
        lines.append("  holdings  none")
    for p in portfolio.positions:
        lines.append(f"  {p.symbol:<9} {p.shares:>6} shares at an average cost of {money(p.cost_basis):>10}  = {money(p.shares * p.cost_basis):>12}" + (f"   (latest purchase {p.acquired})" if p.acquired else ""))
    return lines


def holdings_of(portfolio: Portfolio) -> dict[str, int]:
    return {p.symbol: p.shares for p in portfolio.positions}


def changed(before: Portfolio, after: Portfolio) -> bool:
    return round(before.cash_avail, 2) != round(after.cash_avail, 2) or holdings_of(before) != holdings_of(after)


def account_block(title: str, holder, portfolio: Portfolio, profile) -> list[str]:
    risk = f"{holder.risk_tolerance}" + (f", age {holder.age}" if holder.age else "")
    return [RULE, title, RULE, f"  account {holder.account_number}: {holder.display_name} | {risk} | trade mode {holder.trade_mode.upper()} | "
            f"limits: {profile.max_position_pct:.0%} of equity in one stock, {profile.max_new_exposure_pct:.0%} added by one review", *portfolio_lines(portfolio)]


def status_lines(result, intent, find: int) -> list[str]:
    """A brief status of the recommendations: each company the Advisor recommended with the Critic's verdict, the Account Manager's decision and what was to be bought; then the companies that were scored and not recommended."""
    from prometheus.agents.critic.report import DECISION_WORDS
    from prometheus.application.recommendations import ranking_rows

    order = result.order
    listed = [c for c in result.companies if c.get("symbol")]
    names = {c["symbol"]: c.get("company") for c in listed}
    lines = [RULE, f"RECOMMENDATIONS  asked for the top {find} of {intent.what()}; {len(result.companies)} compan{'y' if len(result.companies) == 1 else 'ies'} found, {len(listed)} listed", RULE]
    if order is None:
        return lines + ["  no work order was made: " + ("; ".join(result.notes) or "no companies")]
    reviews = {r["record_id"]: r for r in order.reviews}
    decisions = {d["record_id"]: d for d in order.trade_decisions}
    recs = sorted(order.recommendations, key=lambda r: ((r.get("ranking") or {}).get("rank") is None, (r.get("ranking") or {}).get("rank") or 0))
    if order.status in ("new", "data_failed"):
        lines.append(f"  the Advisor did not run: the work order is {order.status}" + (": " + "; ".join(order.errors)[:300] if order.errors else ""))
    elif not recs:
        lines.append("  the Advisor recommended none of them")
    for r in recs:
        record_id, review, decision = r["identity"]["record_id"], reviews.get(r["identity"]["record_id"]), decisions.get(r["identity"]["record_id"])
        head = f"  #{(r.get('ranking') or {}).get('rank') or '-'}  {r['symbol']:<6} {(r.get('company') or names.get(r['symbol']) or '')[:26]:<26} Advisor {r.get('score') if r.get('score') is not None else '-':>3} {r.get('stance') or '':<8}"
        if review is None:
            lines.append(head + "  not reviewed")
            continue
        verdict = f"Critic {'APPROVE' if review['approved'] else 'DENY   '} {review['score']:>3}"
        if not review["approved"]:
            why = ", ".join(b["code"] for b in review["blockers"]) or ", ".join(review["denial_reasons"]) or review["summary"]
            lines.append(f"{head}  {verdict}  ({why})")
        else:
            words = DECISION_WORDS.get(decision["will_trade"], decision["will_trade"]) if decision else "no decision"
            what = f"{decision['shares']} shares, about {money(decision['est_cost'])}" if decision and decision["shares"] else ""
            lines.append(f"{head}  {verdict}  decision: {words}" + (f"  {what}" if what else ""))
            if decision and decision["will_trade"] in ("no_funds", "no_expired", "pending_user"):
                lines.append(f"{'':>8}{decision['reason']}")
    balanced = [d for d in order.trade_decisions if d.get("balance")]
    if balanced:
        lines.append(f"  balanced evenly: about {money(balanced[0]['balance']['target_dollars'])} for each of {balanced[0]['balance']['of']} purchases (never more than a company's own limits allow)")
    others = [row for row in ranking_rows(result) if row["status"] != "RECOMMENDED"]
    if others:
        lines.append("  scored, not recommended: " + "; ".join(f"{row['symbol']} ({(row['reason'] or '')[:48]})" for row in others[:6]) + ("; ..." if len(others) > 6 else ""))
    return lines


LIMIT_WORDS = {"position": ("the position limit ({position:.0%} of equity in one stock)", "--max-position-pct"),
               "loss": ("the loss limit (a stop-out may lose {loss:.1%} of equity)", "--max-loss-pct"),
               "exposure": ("the exposure limit ({exposure:.0%} of equity added by one review)", "--max-exposure-pct"),
               "order": ("the order cap", None),
               "liquidity": ("the liquidity cap (a share of the stock's average daily dollar volume)", None)}
POOL_WORDS = {"money": "the money: the cash had no room for another share", "budget": "the amount you asked to spend: it had no room for another share",
              "even_split": "the even split: the same dollars for every company, and another company's limit set the amount"}


def use_of_money_lines(order, before: Portfolio, after: Portfolio, profile) -> list[str]:
    """How much of the money was used, and what held each purchase back. The Account Manager sized every purchase (`decision["size"]`: the shares, every cap, and `binding`, what decided the number), so each line says
    which limit of the profile stopped it, and which flag raises that limit for the next session; the other things that can decide it (the cash, the amount asked for, the even split) are said in words. In Auto mode
    the funds were spent round-robin, a share of each stock in turn, and the report says so. Plain text from the records: nothing is guessed, and the Critic and the Advisor are not credited with a size."""
    spent = before.cash_avail - after.cash_avail
    lines = [RULE, f"HOW MUCH WAS BOUGHT  spent {money(spent)} of {money(before.cash_avail)} ({spent / before.cash_avail:.0%} of the money)" if before.cash_avail else "HOW MUCH WAS BOUGHT", RULE]
    words = {"position": profile.max_position_pct, "loss": profile.max_loss_per_trade_pct, "exposure": profile.max_new_exposure_pct}
    binding, sized_n, ways = set(), 0, set()
    for review in order.reviews:
        decision = next((d for d in order.trade_decisions if d["record_id"] == review["record_id"]), None)
        size = (decision or {}).get("size")
        if not review["approved"] or size is None:
            continue
        sized_n += 1
        ways.add(size["allocation"])
        if not decision["shares"]:
            lines.append(f"  {review['symbol']:<6} no share was bought: {decision['reason']}")
            continue
        text = f"  {review['symbol']:<6} {decision['shares']} share{'s' if decision['shares'] != 1 else ''} ({money(decision['est_cost'])}) of the {money(before.cash_avail)}"
        name = size["binding"]
        if name in LIMIT_WORDS:                                             # a limit of the profile (or of the Account Manager) is what stopped it
            binding.add(name)
            text += ": held back by " + LIMIT_WORDS[name][0].format(**words)
        elif name == "asked":                                               # nothing held it back: the user asked for this much, and the limits and the money would have allowed more
            allowed = min((n for n in size["caps"].values() if n is not None), default=None)
            text += ": the amount you asked for" + (f" (your limits would have allowed up to {allowed} shares)" if allowed is not None and allowed > decision["shares"] else "")
        elif name in POOL_WORDS:
            text += ": held back by " + POOL_WORDS[name]
        if decision["will_trade"] not in ("yes_auto", "yes_user"):
            text += f"; not traded ({decision['will_trade']})"
        lines.append(text)
    if "round_robin" in ways:
        lines.append("  Auto mode spent the funds round-robin: one share of the first stock, one of the next, and so on in rank order, until the money (or every stock's limits) ran out.")
    if "even" in ways:
        lines.append("  The money was split evenly: the same dollars for every company, in whole shares.")
    flags = [LIMIT_WORDS[n][1] for n in ("position", "exposure", "loss") if n in binding and LIMIT_WORDS[n][1]]
    if flags and sized_n:
        share = min(1.0, round(1 / sized_n, 2))
        lines.append("  To buy more, raise the limits for the next session (the account book keeps its own): " + " ".join(f"{f} {1.0 if f == '--max-exposure-pct' else share if f == '--max-position-pct' else min(1.0, profile.max_loss_per_trade_pct * 2):g}" for f in flags))
    elif spent and sized_n:
        lines.append("  No limit of the profile held these purchases back; the money, the amount asked for and whole shares did.")
    return lines


def trade_lines(order) -> list[str]:
    from prometheus.agents.trader.report import trade_section

    lines = [RULE, "TRADER  results", RULE]
    if order is None or not order.trader_runs:
        return lines + ["  the Trader did not run"]
    lines += trade_section(order)
    run = order.trader_runs[-1]
    if run.get("errors"):
        lines += [f"  trader note: {e}" for e in run["errors"][:5]]
    return lines


def why_not(result, order) -> list[str]:
    """Why the portfolio did not change, from what the run recorded."""
    reasons = [n for n in result.notes if "did not run" not in n][:3]
    if order is None:
        return reasons or ["no work order was made"]
    if order.status in ("new", "data_failed"):
        reasons.append(f"the Data Clerk's work order is {order.status}" + (": " + "; ".join(order.errors)[:300] if order.errors else ""))
    elif not order.recommendations:
        reasons.append("the Advisor recommended none of the companies (each failed a gate or a strategy: see the ranking)")
    denied = [r for r in order.reviews if not r["approved"]]
    if denied:
        reasons.append(f"the Critic denied {len(denied)} of {len(order.reviews)}: " + "; ".join(f"{r['symbol']} ({', '.join(b['code'] for b in r['blockers']) or 'score under the mark'})" for r in denied[:5]))
    for d in order.trade_decisions:
        if d["will_trade"] == "pending_user":
            reasons.append(f"{d['symbol']} waits for the user's answer ({d['reason']})")
        elif d["will_trade"] in ("no_funds", "no_expired", "no_user_denied"):
            reasons.append(f"{d['symbol']}: {d['reason']}")
    for t in order.trades:
        if t["status"] == "rejected":
            reasons.append(f"{t['symbol']} was rejected by the Trader ({t['reject_code']}): {t['reason']}")
        elif t["status"] == "active":
            reasons.append(f"{t['symbol']} is waiting (a resting order)")
    return reasons or ["nothing was approved to trade"]


# ---------------------------------------------------------------- the session

NODE_WORDS = {"account_manager": "the Account Manager read the request and made the work order", "data_clerk": "the Data Clerk found the companies and answered the work order",
              "advisor": "the Advisor ranked the companies", "critic": "the Critic reviewed each recommendation against the profile", "account_manager_decision": "the Account Manager decided what may trade",
              "trader": "the Trader executed the permitted trades (simulated: nothing is sent to a broker)"}


def approver_for(parts_approver: Callable | None, ask=input, out=print) -> Callable:
    """The person who approves trades in Ask mode: the parts' own approver, else one that asks at the terminal THROUGH THE SESSION'S ask AND out (so what a test or a caller feeds is what answers). With the real
    `input` it asks only when stdin is a terminal (a script nobody watches never trades in Ask mode); a `ask` that was given is taken to be a person."""
    from prometheus.agents.account_manager.decisions import terminal_approver

    return parts_approver or terminal_approver(ask=ask, show=out, interactive=None if ask is input else True)


def clarifier_for(ask=input, out=print) -> Callable:
    """The Account Manager's channel to the user for its questions about how much to buy (design choices entry 85), THROUGH THE SESSION'S ask AND out, as `approver_for` does for the approvals: with the real `input` it
    asks only when stdin is a terminal (nobody there: nothing is asked and the loop is not run); an `ask` that was given is taken to be a person."""
    from prometheus.agents.account_manager.clarify import terminal_clarifier

    return terminal_clarifier(ask=ask, show=out, interactive=None if ask is input else True)


def decision_lines(order, mode: str) -> list[str]:
    """What the Account Manager decided, right after it did: each purchase with its final size and who said yes, and whether anything goes on to the Trader. In Ask mode this is the answer to "what happens now?"."""
    from prometheus.agents.critic.report import DECISION_WORDS

    lines, going = [], 0
    for d in order.trade_decisions:
        if d["action"] != "reduce" or d["will_trade"] in ("yes_auto", "yes_user"):
            what = f"{d['shares']} shares, about {money(d['est_cost'])}" if d["shares"] and d["est_cost"] else f"the whole holding ({d['shares']} shares)" if d["action"] == "reduce" and d["shares"] else ""
            lines.append(f"      {d['symbol']:<6} {DECISION_WORDS.get(d['will_trade'], d['will_trade'])}" + (f"  {what}" if what else ""))
        going += d["will_trade"] in ("yes_auto", "yes_user")
    waiting = [d["symbol"] for d in order.trade_decisions if d["will_trade"] == "pending_user"]
    if going:
        lines.append(f"      -> {going} approved trade{'s' if going != 1 else ''} go on to the Trader.")
    else:
        lines.append("      -> nothing was approved to trade: the Trader will not run.")
    if waiting:
        lines.append(f"      {', '.join(waiting)} still wait for your answer: " + ("nobody was at the terminal to ask (Ask mode needs a person at the keyboard: run it in a terminal, or use --mode auto)." if mode == "ask" else "nobody answered."))
    return lines


def run_session(args: argparse.Namespace, person: dict, parts: Parts, ask=input, out=print) -> int:
    say = lambda *lines: [out(line) for line in lines]
    try:
        account_id = open_account(parts, args, person)
        manager = parts.new_manager()
        manager.connect(parts.store, account_id)
        holder = manager.users.get()
        profile = with_limits(manager.profile, args)               # the overrides are for this session only: the account book keeps its own
        before = manager.list_portfolio()
    except ValueError as error:   # our own checks and the account book's refusals: the message is for the user
        say(f"ERROR: the account could not be opened: {error}")
        return 2
    except Exception as error:   # the database, the network, a missing key: the kind only, never connection details (an OSError carries the host and the user)
        say(f"ERROR: the account book could not be used ({type(error).__name__})")
        return 2
    created = not args.account
    say("", *account_block("ACCOUNT  " + ("created" if created else "in use") + (f", {money(person['cash'])} added" if created else ""), holder, before, profile))
    if note := clock_note():
        say("", note)
    say("", RULE, "REQUEST", RULE, f"  \"{' '.join(args.prompt.split())}\"")

    def confirm(intent) -> bool | None:
        """The desk shows what the Account Manager understood and asks for a yes; the desk does nothing until it gets one."""
        say(f"  understood: {intent.describe()}")
        if args.yes:
            return True
        try:
            answer = ask("  Go ahead with this? [Y/n]: ")
        except EOFError:
            say("  (no answer: not going ahead. Pass --yes to run without being asked.)")
            return None
        if answer.strip().lower().startswith("n"):
            say("  Stopped. Nothing was bought or sold.")
            return False
        return True

    # ---- the LangGraph desk, started with the user's own words
    def new_manager():
        manager = parts.new_manager()
        manager.screen = out         # the user talks to the Account Manager and to nobody else: what it has to tell them while it trades (a trade that failed, and the account as it stands) goes to this same screen
        return manager

    graph = build_desk(clerk_factory=lambda order: parts.clerk, advisor_factory=lambda order: parts.advisor, funds_factory=new_manager, with_critic=True, critic_factory=lambda order: parts.critic,
                       with_trader=True, trader_factory=lambda order: parts.trader, approver=approver_for(parts.approver, ask, out), store_factory=lambda: parts.store, confirm=confirm, clarify=clarifier_for(ask, out))
    request = {"ask": {"text": args.prompt, "default_count": args.default_count, "fetch": args.fetch}, "account": {"account": account_id, "profile": profile}, "trade": {"mode": "paper", "book": True, "model": not args.no_trader_model}}
    say("", RULE, "THE DESK  (LangGraph: Account Manager -> Data Clerk -> Advisor -> Critic -> Account Manager's decision -> Trader)", RULE)
    final, shown = None, 0
    for state in graph.stream({"request": request, "order": None, "errors": [], "resolution": None, "funds": None, "path": [], "seconds": {}, "with_trader": True, "plan": None}, stream_mode="values"):
        final = state
        for name in state["path"][shown:]:
            say(f"  .. {NODE_WORDS.get(name, name)} ({state['seconds'].get(name, 0):.1f} s)")
            if name == "account_manager_decision" and state.get("order") is not None and state["order"].trade_decisions:
                say(*decision_lines(state["order"], profile.trade_mode))
        shown = len(state["path"])
    order, plan = final["order"], final.get("plan")
    if order is None or plan is None:   # the request could not be read, was not confirmed, or the account could not be used: the desk stopped at its first node
        say("", "  " + ("; ".join(final["errors"]) or "the desk did not start"))
        return 2
    intent, find, result = plan["intent"], plan["find"], plan["result"]
    result.order = order
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps({"request": " ".join(args.prompt.split()), "intent": intent.model_dump(), "path": final["path"], "order": json.loads(order.model_dump_json(exclude={"data"}))}, indent=1), encoding="utf-8")
        say(f"  (what the desk made is saved in {args.save})")
    buying = intent.action == "buy"
    say("", *status_lines(result, intent, find))
    if buying:
        say("", *trade_lines(order))
    after = manager.list_portfolio()
    if buying and order.reviews:
        say("", *use_of_money_lines(order, before, after, profile))
    say("", *account_block("ACCOUNT  after the session", holder, after, profile))

    # ---- the verdict
    did = changed(before, after)
    if not buying:
        say("", RULE, "SESSION  a look only: nothing was to be bought. The portfolio is as it was.", RULE)
        ok = True
    elif did:
        bought = [f"{shares - holdings_of(before).get(symbol, 0):+d} {symbol}" for symbol, shares in holdings_of(after).items() if shares != holdings_of(before).get(symbol, 0)]
        sold = [f"{shares - holdings_of(after).get(symbol, 0):+d} {symbol}" for symbol, shares in holdings_of(before).items() if symbol not in holdings_of(after)]
        say("", RULE, f"SESSION  PASSED: the portfolio changed ({', '.join(bought + sold)}); cash {money(before.cash_avail)} -> {money(after.cash_avail)}", RULE)
        ok = True
    else:
        say("", RULE, "SESSION  the portfolio did NOT change. Why:", *[f"  - {reason}" for reason in why_not(result, order)], RULE)
        ok = args.allow_no_trade
    if args.cleanup and created:
        try:
            parts.store.delete_account(account_id)
            say(f"  (account {account_id} was deleted: --cleanup)")
        except Exception as error:   # never hide the result for a clean-up that failed
            say(f"  (account {account_id} could not be deleted: {type(error).__name__})")
    elif created:
        say(f"  account {account_id} is kept: python -m prometheus.agents.account_manager.account_manager --account {account_id}")
    return 0 if ok else 1


def main(argv: list[str] | None = None, parts: Parts | None = None, ask=input, out=print) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    person = check_args(parser, args)
    if parts is None:
        try:
            parts = build_parts(args)
        except FileNotFoundError as error:
            out(f"ERROR: {error}")
            out("       the account book is not set up: run  bash scripts/setup_postgres.sh")
            return 2
        except (OSError, RuntimeError, ValueError) as error:   # a missing key: the message names the setting, never its value
            out(f"ERROR: {error}")
            return 2
    return run_session(args, person, parts, ask=ask, out=out)


if __name__ == "__main__":
    sys.exit(main())
