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
                 recommendation against the profile; the decision node trades what the Critic approves (Auto: nothing is asked; Ask: you approve each trade), returns what the Critic denied to you with its reason, and spreads the money evenly when asked; the Trader executes against a
                 SIMULATED broker. Every fill is booked in the account. This module only starts the graph, shows each node as it finishes, and prints.
  4. REPORTS     the status of the recommendations, the Trader's results, and the account's portfolio before and after.

A request that NAMES THE TRADE ("Buy 50 shares of <a ticker>", design choices entry 87) is a DIRECT ORDER: the desk skips the research, the Advisor and the Critic (Account Manager -> Account Manager's direct decision -> Trader), nothing
is recommended and the user is not asked a second time; the report shows the order, the Trader's result and the portfolio before and after. A stock the Data Clerk did not find (design choices entries 101 and 102: "Google" is found as Alphabet
Inc., "Chick-fil-a" is not listed) is left out without a question in Auto mode and asked about in Ask mode (the question is the approver's, so `--yes` does not answer it); when the order cannot be made (nothing found, a lookup that failed, a no,
no answer) the session says what was missing and why and asks the user to clarify the request and send it again (`direct.CLARIFY`), and exits 1.

Nothing here knows a sector, a company or a ticker: it is the same for any request the form can hold. Nothing is sent to a broker. A session that was meant to trade but left the portfolio as it was is a FAILED
check (exit 1) with the reasons, because a sanity check that cannot trade has found something; `--allow-no-trade` accepts that.

IGNORE CRITIC (design choices entry 148): `--ignore-critic` skips the Critic's step for the session. The desk goes Advisor -> the Account Manager's decision, and every Advisor recommendation counts as approved (Auto trades it, Ask asks you and says that the Critic did
not review it); the report says so before the decisions ("IGNORE CRITIC: the Critic did not review these recommendations (the user's setting)"). Without the flag nothing is different.

A REQUEST THAT ONLY ASKS FOR THE NEWS ("Show the news for Tesla", "What's TSLA doing the last week?"; design choices entry 149, `design/news_view_spec.md`) is answered by the Data Clerk alone (Account Manager -> Data Clerk): nothing is asked, in either mode, nothing is
recommended, decided or traded, and the session prints the stocks, the window, the summary and the numbered list of the articles (headline, publisher, date, link), and nothing about decisions, trades or money. The Data Clerk ALWAYS fetches the latest news of each stock first
(the news only); `--refuse-news-fetch` is the TEST-ONLY switch that takes that right away for the run (a fetch writes `news/SYMBOL.txt`; the news is then searched as it is stored and the report says so). `--save` writes the work order WITH its `data` (the news is the result). A
news run is not saved for the Track Record: it has no recommendation to score.
A news request may name a PAST DAY or span (entry 151, `design/news_dated_window_spec.md`: "What was the news on TSLA on November 20?"): the headline, the "understood" line and the list say "on 2026-11-20" or "from 2026-11-20 to 2026-11-22" in place of "for the last 7 days".

A REQUEST THAT ONLY ASKS FOR THE CURRENT PRICE ("What's the current price of TSLA?", "Price of Tesla", "Quote for TSLA and NVDA"; design choices entry 150, `design/quote_request_spec.md`) is answered the same way, by the Data Clerk alone (Account Manager -> Data Clerk): nothing is asked,
recommended, decided or traded, and the session prints one line for each stock (symbol, company, price, the change and its percent, the time or the simulated clock's market close) and the problems, and nothing about decisions, trades or money. The quote is the live one
(Finnhub's `/quote`) or, with USE_LIVE_TRADING=false, the stored close of the simulated day; nothing else is fetched and no file is written. `--save` writes the work order WITH its `data`; the capture goes in `tests/fixtures/quote_real/`. A price run is not saved for the Track Record.

MONEY IS `decimal.Decimal` (design choices entry 112): the cash a user starts with is `parse_money`'s Decimal, the report's amounts are exact (`spent / cash` is a Decimal ratio, printed as a percentage), and the saved session (`--save`) is written by the one encoder
(`money.dumps`): a Decimal is a JSON number, a whole value an integer, never a string; a capture made before this change loads as it is.

EXIT CODES: 0 the session did what was asked (the portfolio changed, or it was only a look); 1 it could not trade (the reasons are printed); 2 a bad flag, or the account book, the keys or the request could not be used.
"""
import argparse
import functools
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from prometheus.agents.account_manager import ignore_critic as ignoring
from prometheus.agents.account_manager.direct import CLARIFY, data_clerk_lines, direct_section
from prometheus.agents.account_manager.neutral import neutral_lines
from prometheus.common import shares as sh
from prometheus.common.money import dumps
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.common.trading_clock import clock_note
from prometheus.common.user_account import clean_age, clean_display_name, clean_risk_level, clean_trade_mode, parse_money
from prometheus.common.user_profile import TOLERANCES
from prometheus.application.web import views
from prometheus.langgraph.desk import build_desk
from prometheus.common import langsmith_tracing
from prometheus.proving import sessions as proving_sessions
from prometheus.proving.sessions import screen_of

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
    approver: Callable | None = None    # asked about a trade that needs the user (Ask mode, the sells of a funding plan); never in Auto mode, which asks nothing; None: the session's own, which asks at the terminal (`approver_for`)
    guard: object | None = None         # the security layer's doors (entry 116, `prometheus.security.guard.Guard`): the real parts build `Guard.with_scorer()` (rules + the LLM Risk Scorer + audit); None: the desk's own `Guard.code_only()`


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A user session end to end: create a user and an account, add money, read a request in the user's own words, run the desk, and report the recommendations, the trades and the portfolio.")
    who = parser.add_argument_group("the user and the account (a new one is created, unless --account)")
    who.add_argument("--name", help="the user's name")
    who.add_argument("--age", type=int, help="the user's age (18 to 120): older users are held to a lower risk by the Critic")
    who.add_argument("--risk", choices=TOLERANCES, help="the risk tolerance: conservative, moderate or aggressive")
    who.add_argument("--cash", help="dollars to add to the new account, for example 15000")
    who.add_argument("--mode", choices=("ask", "auto"), help="ask: the user approves each trade. auto: the desk trades what the Critic approves and asks nothing")
    who.add_argument("--account", help="use an EXISTING account (an account number or a sample key) instead of creating one: no user is created and no money is added")
    parser.add_argument("--prompt", required=True, help="what the user wants, in their own words")
    desk = parser.add_argument_group("the desk")
    desk.add_argument("--fetch", action="store_true", help="REFRESH the companies' data first with the PAID APIs (bars, news, press releases, filings: writes files); without it the stored data is used")
    desk.add_argument("--default-count", type=int, default=DEFAULT_COUNT, help=f"how many of the top companies to buy from when the request does not say (default {DEFAULT_COUNT})")
    desk.add_argument("--model", default="openai/gpt-oss-120b", help="the Advisor's OpenRouter model id (default openai/gpt-oss-120b)")
    desk.add_argument("--no-trader-model", action="store_true", help="the Trader without a model: the program places the default orders")
    desk.add_argument("--always-raise-funds", action="store_true", default=None, help="for this session ALWAYS_RAISE_FUNDS is true: the same as saying \"raise funds if needed\" in the request (entry 142, R12): when the cash does not cover a purchase you typed, holdings are sold to pay for it "
                      "(Ask mode still shows you the plan and asks; Auto mode sells and buys; \"don't sell anything\" in your words wins). Without the flag the setting is what the environment and .env say (false when unset)")
    desk.add_argument("--ignore-critic", action="store_true", help="skip the Critic's step for THIS session (design choices entry 148): the desk goes Advisor -> the Account Manager's decision, and every Advisor recommendation counts as approved (Auto trades it; Ask asks you, "
                      "and says the Critic did not review it). Without the flag the Critic reviews, and a denial is final")
    desk.add_argument("--refuse-news-fetch", action="store_true", help="TEST ONLY (entry 149): a request that only asks for the news always fetches the latest news first (a paid call that writes news/SYMBOL.txt); this takes that right away for the session, so the news "
                      "is searched as it is stored and the report says that the fetch was refused. Never a setting of the page")
    desk.add_argument("--real-ledger", action="store_true", help="use the real ledgers (data/ledger/): a company recommended before is then hidden by its cooldown. The default is throwaway ledgers, so a session can be repeated")
    run = parser.add_argument_group("the session")
    run.add_argument("--yes", "-y", action="store_true", help="do not ask to confirm what was understood (in Ask mode you are STILL asked to approve each trade: --yes never approves a trade)")
    run.add_argument("--allow-no-trade", action="store_true", help="exit 0 even when the portfolio did not change")
    run.add_argument("--save", metavar="FILE", help="write what the desk made (the request, what was understood, and the work order without its data: recommendations, reviews, decisions, trades; a direct order keeps its data: the company names and symbols the Data Clerk found) to this JSON file: a real capture")
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
    from prometheus.security.guard import Guard

    store = open_account_store()
    guard = Guard.with_scorer()         # entry 116: the request, the documents and the messages are screened; the model of the `guard` role is built when the first request is screened
    clerk = DataClerk(allow_fetch=args.fetch, guard=guard)
    advisor = Advisor(model_id=args.model, clerk=clerk, ledger=None if args.real_ledger else throwaway_ledger(), effort="auto", providers="auto")
    guard.ledger = advisor.ledger       # the Trust Score's history signal reads the same ledger the Advisor writes
    critic = Critic(clerk=clerk, ledger=None if args.real_ledger else throwaway_ledger(), history=advisor.ledger)
    trader = Trader(mode="paper", use_model=not args.no_trader_model, ledger=None if args.real_ledger else NullLedger())
    return Parts(store=store, clerk=clerk, advisor=advisor, critic=critic, trader=trader, new_manager=lambda: AccountManager(clerk=clerk, guard=guard), guard=guard)      # the approver is made by `run_session`, from its own ask and out


def open_account(parts: Parts, args: argparse.Namespace, person: dict) -> int:
    """The account number: a new user and profile with the money added, or the existing account asked for."""
    if args.account:
        return int(args.account) if args.account.isdigit() else parts.store.account_id_for_seed(args.account)
    account_id = parts.store.create_account(person["name"], person["risk"], trade_mode=person["mode"], age=person["age"])
    banker = parts.new_manager()
    banker.connect(parts.store, account_id)
    banker.add_funds(person["cash"])      # `parse_money` already made it a Decimal to the cent
    return account_id


# ---------------------------------------------------------------- the reports

def money(value) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def portfolio_lines(portfolio: Portfolio) -> list[str]:
    lines = [f"  cash      {money(portfolio.cash_avail):>14}"] if not portfolio.reserved_cash else [f"  cash      {money(portfolio.cash_total):>14}", f"  set aside {money(portfolio.reserved_cash):>14}   (for waiting limit orders)", f"  available {money(portfolio.cash_avail):>14}"]
    if not portfolio.positions:
        lines.append("  holdings  none")
    width = max(6, *(len(sh.text(p.shares)) for p in portfolio.positions)) if portfolio.positions else 6                    # a quantity with a fraction is wider (entry 110)
    for p in portfolio.positions:
        lines.append(f"  {p.symbol:<9} {sh.text(p.shares):>{width}} shares at an average cost of {money(p.cost_basis):>10}  = {money(trade_amount(p.shares, p.cost_basis)):>12}" + (f"   (latest purchase {p.acquired})" if p.acquired else "")
                     + (f"   ({sh.text(portfolio.reserved_shares[p.symbol])} set aside for a waiting sale)" if portfolio.reserved_shares.get(p.symbol) else ""))
    return lines


def holdings_of(portfolio: Portfolio) -> dict[str, Decimal]:
    return {p.symbol: p.shares for p in portfolio.positions}


def changed(before: Portfolio, after: Portfolio) -> bool:
    return before.cash_total != after.cash_total or holdings_of(before) != holdings_of(after)      # the WHOLE cash: cash a waiting order sets aside is not a change of the account (entry 140)


def account_block(title: str, holder, portfolio: Portfolio, profile) -> list[str]:
    risk = f"{holder.risk_tolerance}" + (f", age {holder.age}" if holder.age else "")
    return [RULE, title, RULE, f"  account {holder.account_number}: {holder.display_name} | {risk} | trade mode {holder.trade_mode.upper()}", *portfolio_lines(portfolio)]


def status_lines(result, intent, find: int) -> list[str]:
    """A brief status of the recommendations: the top N the request asked for as ONE list in the order of the Advisor's overall score, numbered 1 to N (entry 145): each recommendation with the Critic's verdict, the Account Manager's decision and what was to be bought, and the best scored companies that were not recommended in the places left; the others are counted."""
    from prometheus.agents.critic.report import DECISION_WORDS
    from prometheus.application.recommendations import ranking_rows

    order = result.order
    listed = [c for c in result.companies if c.get("symbol")]
    names = {c["symbol"]: c.get("company") for c in listed}
    lines = [RULE, f"RECOMMENDATIONS  asked for the top {find} of {intent.what()}; the Data Clerk found {len(listed)} listed compan{'y' if len(listed) == 1 else 'ies'} to score"
                   + (f" ({len(result.companies) - len(listed)} not listed)" if len(result.companies) > len(listed) else ""), RULE]
    if order is None:
        return lines + ["  no work order was made: " + ("; ".join(result.notes) or "no companies")]
    if ignoring.skipped(order):      # entry 148: said once, at the head of the list
        lines.append("  " + ignoring.SKIPPED_LINE)
    reviews = {r["record_id"]: r for r in order.reviews}
    decisions = {d["record_id"]: d for d in order.trade_decisions}
    recs = sorted(order.recommendations, key=lambda r: ((r.get("ranking") or {}).get("rank") is None, (r.get("ranking") or {}).get("rank") or 0))
    if order.status in ("new", "data_failed"):
        lines.append(f"  the Advisor did not run: the work order is {order.status}" + (": " + "; ".join(order.errors)[:300] if order.errors else ""))
    elif not recs:
        lines.append("  the Advisor recommended none of them")
    if recs and len(recs) < find:
        lines.append(f"  {len(recs)} of the top {find} passed a strategy: the Advisor recommends only what did")
    others = [row for row in ranking_rows(result) if row["status"] != "RECOMMENDED"]
    room = max(0, find - len(recs))                 # the list stops at the number asked for (entry 144): the best scored companies fill the places the recommendations left
    overall = lambda r: (r.get("ranking") or {}).get("universe_rank") if (r.get("ranking") or {}).get("universe_rank") is not None else (r.get("ranking") or {}).get("rank")
    shown = sorted([("rec", overall(r), r) for r in recs] + [("scored", row["rank"], row) for row in others[:room]], key=lambda item: (item[1] is None, item[1] or 0))      # ONE list in the order of the overall score, numbered 1 to N (entry 145)
    for n, (kind, _, r) in enumerate(shown, 1):
        if kind == "scored":
            lines.append(f"  #{n:<2} {r['symbol']:<6} {(r.get('company') or names.get(r['symbol']) or '')[:26]:<26} Advisor {r.get('score') if r.get('score') is not None else '-':>3} {r.get('stance') or '':<8}  not recommended ({(r['reason'] or '')[:60]})")
            continue
        record_id, review, decision = r["identity"]["record_id"], reviews.get(r["identity"]["record_id"]), decisions.get(r["identity"]["record_id"])
        head = f"  #{n:<2} {r['symbol']:<6} {(r.get('company') or names.get(r['symbol']) or '')[:26]:<26} Advisor {r.get('score') if r.get('score') is not None else '-':>3} {r.get('stance') or '':<8}"
        if review is None and order.ignore_critic and decision and decision.get("origin") == ignoring.ADVISOR:      # the user chose to ignore the Critic (entry 148): it did not run, and the recommendation counts as approved
            words = DECISION_WORDS.get(decision["will_trade"], decision["will_trade"])
            what = (f"{money(decision['dollars'])}, about {sh.text(decision['shares'])} shares" + (f" at {money(decision['price'])}" if decision.get("price") else "") if decision.get("dollars") else
                    f"{sh.text(decision['shares'])} shares, about {money(decision['est_cost'])}" if decision["shares"] else "")
            lines.append(f"{head}  Critic SKIPPED (Ignore Critic)  decision: {words}" + (f"  {what}" if what else ""))
            if decision["will_trade"] in ("no_funds", "no_expired", "pending_user", "no_system_denied"):
                lines.append(f"{'':>8}{decision['reason']}")
            continue
        if review is None:
            lines.append(head + "  not reviewed")
            continue
        verdict = f"Critic {'APPROVE' if review['approved'] else 'DENY   '} {review['score']:>3}"
        if not review["approved"]:
            why = ", ".join(review["denial_reasons"]) or ", ".join(b["code"] for b in review["blockers"]) or review["summary"]      # the blockers, the score under the mark, and a person who must look (`needs_human_review`)
            lines.append(f"{head}  {verdict}  ({why})")
        else:
            words = DECISION_WORDS.get(decision["will_trade"], decision["will_trade"]) if decision else "no decision"
            what = (f"{money(decision['dollars'])}, about {sh.text(decision['shares'])} shares" + (f" at {money(decision['price'])}" if decision.get("price") else "") if decision and decision.get("dollars") else
                    f"{sh.text(decision['shares'])} shares, about {money(decision['est_cost'])}" if decision and decision["shares"] else "")      # an order in dollars says its dollars first (entry 111, M2)
            lines.append(f"{head}  {verdict}  decision: {words}" + (f"  {what}" if what else ""))
            if decision and decision["will_trade"] in ("no_funds", "no_expired", "pending_user"):
                lines.append(f"{'':>8}{decision['reason']}")
    balanced = [d for d in order.trade_decisions if d.get("balance")]
    if balanced:
        lines.append(f"  balanced evenly: about {money(balanced[0]['balance']['target_dollars'])} for each of {balanced[0]['balance']['of']} purchases")
    if others[room:]:
        lines.append(f"  {len(others[room:])} more scored below your top {find}, not shown")
    return lines


POOL_WORDS = {"money": "the cash: it had no room for another share", "budget": "the amount you asked to spend: it had no room for another share",
              "even_split": "the even split: the same dollars for every company, and another company's smaller amount set it"}
FRACTION_WORDS = {"money": "the cash: its equal share was spent to the cent", "budget": "the amount you asked to spend: its equal share was spent to the cent"}      # a fractional even split (entry 110) has no "next share" that did not fit


def sized_purchases(order) -> list[tuple[str, dict]]:
    """The purchases of the recommendation path that the Account Manager sized, in the order they were decided: each recommendation the Critic APPROVED, then each NEUTRAL recommendation counted as approved (entry 91, `origin:
    "neutral"`; a run of neutral companies alone has no review at all, entry 123), and each Advisor recommendation the user chose not to have the Critic review (`origin: "advisor"`, entry 148). [(symbol, decision)]; a purchase with no size (not decided as a trade) is left out."""
    approved = {r["record_id"]: r["symbol"] for r in order.reviews if r["approved"]}
    out = []
    for decision in order.trade_decisions:
        symbol = approved.get(decision.get("record_id")) or (decision.get("symbol") if decision.get("origin") in ("neutral", ignoring.ADVISOR) else None)      # an Advisor recommendation the user chose not to have reviewed (entry 148) has no review either
        if symbol and decision.get("size") is not None:
            out.append((symbol, decision))
    return out


def use_of_money_lines(order, before: Portfolio, after: Portfolio) -> list[str]:
    """How much of the money was used, and what stopped each purchase. The Account Manager sized every purchase (`decision["size"]`: the shares, the caps and `binding`, what decided the number). THE ONLY CHECK IS THAT THE
    FUNDS COVER THE ORDER (design choices entry 91): no limit of the profile exists, so what ends a purchase is the cash, the amount the user asked to spend or buy, or the even split. In Auto mode the funds were spent
    round-robin, a share of each stock in turn, and the report says so. Plain text from the records: nothing is guessed, and the Critic and the Advisor are not credited with a size."""
    spent = before.cash_avail - after.cash_avail
    lines = [RULE, f"HOW MUCH WAS BOUGHT  spent {money(spent)} of {money(before.cash_avail)} ({spent / before.cash_avail:.0%} of the money)" if before.cash_avail else "HOW MUCH WAS BOUGHT", RULE]
    ways = set()
    for symbol, decision in sized_purchases(order):
        size = decision["size"]
        ways.add(size["allocation"])
        if not decision["shares"]:
            lines.append(f"  {symbol:<6} no share was bought: {decision['reason']}")
            continue
        if decision.get("dollars"):      # an order in dollars (entry 111, M2): the dollars first, and the shares as the estimate at the price it was sized at
            text = f"  {symbol:<6} {money(decision['dollars'])} (about {sh.text(decision['shares'])} {sh.noun(decision['shares'])} at {money(decision['price'])}) of the {money(before.cash_avail)}" if decision.get("price") else \
                f"  {symbol:<6} {money(decision['dollars'])} (about {sh.text(decision['shares'])} {sh.noun(decision['shares'])}) of the {money(before.cash_avail)}"
        else:
            text = f"  {symbol:<6} {sh.text(decision['shares'])} {sh.noun(decision['shares'])} ({money(decision['est_cost'])}) of the {money(before.cash_avail)}"
        name = size["binding"]
        if name == "asked":                                                 # nothing held it back: the user asked for this much
            text += ": the amount you asked for"
        elif name in FRACTION_WORDS and size.get("fractional"):
            text += ": stopped by " + FRACTION_WORDS[name]
        elif name in POOL_WORDS:
            text += ": stopped by " + POOL_WORDS[name]
        if decision["will_trade"] not in ("yes_auto", "yes_user"):
            text += f"; not traded ({decision['will_trade']})"
        lines.append(text)
    if "round_robin" in ways and any((d.get("size") or {}).get("binding") != "asked" for _, d in sized_purchases(order)):      # every purchase the amount the user typed: nothing was spent round-robin (entry 146)
        lines.append("  Auto mode spent the funds round-robin: one share of the first stock, one of the next, and so on in rank order, until the money ran out.")
    if "even" in ways:
        if any((d.get("size") or {}).get("fractional") for d in order.trade_decisions):      # which split it was (entry 110): in fractions it is equal to the cent
            lines.append("  The money was split evenly: the same dollars for every company, equal to the cent, by buying a fraction of a share where it takes one (the cents the equal amounts leave over go one at a time to the first companies in rank order).")
        else:
            lines.append("  The money was split evenly: the same dollars for every company as far as whole shares allow (what the equal amounts leave over buys more shares, the smallest amount first).")
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


def waiting_lines_of(order) -> list[str]:
    """The sentence of each direct order that was placed to WAIT for its limit price (a decision with an `open_order_id`): "Waiting: buy 10 MU at $90.00 or less (now $95.12), $900.00 set aside, until 4:00 pm ET today." (design choices entry 140)."""
    return [d["reason"] for d in order.trade_decisions if isinstance(d, dict) and d.get("open_order_id") is not None and isinstance(d.get("reason"), str)]


def watcher_note(parts) -> list[str]:
    """A waiting order is watched only while the page's server runs (`./run_ui.sh`): this session says so, and says when NO server is running its watcher now (the account book's advisory lock is free), so the user knows nothing will fill or expire the order until one is started."""
    try:
        running = bool(parts.store.watcher_running())
    except Exception:
        return ["  A waiting order is watched only while the page's server runs (./run_ui.sh)."]
    return ["  A waiting order is watched only while the page's server runs (./run_ui.sh)." + ("" if running else " NO SERVER IS RUNNING a watcher now: start one (./run_ui.sh), or cancel the order with `python -m prometheus.agents.account_manager.orders --account N --cancel ID`; its cash or shares stay set aside until a watcher expires it after its session's close, or you cancel it.")]


def why_not(result, order) -> list[str]:
    """Why the portfolio did not change, from what the run recorded."""
    reasons = [n for n in result.notes if "did not run" not in n][:3] if result is not None else []      # no `result` for a direct order: nothing was searched
    if order is None:
        return reasons or ["no work order was made"]
    if order.status in ("new", "data_failed"):
        reasons.append(f"the Data Clerk's work order is {order.status}" + (": " + "; ".join(order.errors)[:300] if order.errors else ""))
    elif not order.recommendations and not order.direct_orders:
        reasons.append("the Advisor recommended none of the companies (each failed a gate or a strategy: see the ranking)")
    denied = [r for r in order.reviews if not r["approved"]]
    if denied:
        everything = len(denied) == len(order.reviews)
        listed = "; ".join(f"{r['symbol']} ({', '.join(r['denial_reasons']) or r['summary']})" for r in denied[:5])
        text = (f"the Critic denied {'the recommendation' if len(denied) == 1 else f'all {len(denied)} recommendations'} and approved none: {listed}" if everything
                else f"the Critic denied {len(denied)} of {len(order.reviews)}: {listed}")
        reasons.insert(0, text) if everything else reasons.append(text)       # a run where everything was denied says so first
    for d in order.trade_decisions:
        if d["will_trade"] == "pending_user":
            reasons.append(f"{d['symbol']} waits for the user's answer ({d['reason']})")
        elif d["will_trade"] in ("no_funds", "no_expired", "no_user_denied") or (d.get("origin") == "direct" and d["will_trade"] == "no_system_denied"):
            reasons.append(f"{d['symbol']}: {d['reason']}")
    for t in order.trades:
        if t["status"] == "rejected":
            reasons.append(f"{t['symbol']} was rejected by the Trader ({t['reject_code']}): {t['reason']}")
        elif t["status"] == "active":
            reasons.append(f"{t['symbol']} is waiting (a resting order)")
    return reasons or ["nothing was approved to trade"]


# ---------------------------------------------------------------- the session

def news_lines(view: dict, intent=None) -> list[str]:
    """What the session prints for a request that only asked for the news (entry 149, N10): what was understood, the stocks and the window, how the news was fetched (or why it was not), the summary, the numbered list of the articles (the cited ones
    first) and what went wrong. The ONE shape is `views.news_view`; nothing here is about decisions, trades or money."""
    lines = [RULE, "NEWS  " + views.news_headline(view), RULE]
    if intent is not None:
        lines.append(f"  understood: {intent.describe()}")
    if view["stocks"]:
        lines.append("  stocks: " + "; ".join(f"{s['symbol']}" + (f" ({s['company']})" if s.get("company") else "") for s in view["stocks"]))
    lines += [f"  not found: {m['typed']}: {m['reason']}" for m in view["not_found"]]
    lines.append("  news: " + ("the latest news was fetched just now, then searched" if view["fetched"] else view["fetch_note"] if view.get("fetch_note") else "searched in the stored news (a theme is never fetched for)"))
    lines += ["", "SUMMARY"] + [f"  {line}" for line in (view["summary"] or "(the Data Clerk gave no summary)").splitlines()]
    if view["items"]:
        lines += ["", f"ARTICLES  ({len(view['items'])}; the ones the summary cites first)"]
        for item in view["items"]:
            lines.append(f"  [{', '.join(str(n) for n in item['numbers'])}] {item['headline'] or '(no headline)'}  ({item['publisher'] or 'unknown publisher'}, {item['date'] or 'no date'}){'  cited' if item['cited'] else ''}")
            if item.get("url"):
                lines.append(f"      {item['url']}")
    elif view["status"] == "data_ready":
        lines += ["", "ARTICLES  none: the stored news has nothing about this " + (views.news_window_text(view) if view.get("window") else "in the window")]      # a past day is named (entry 151): "... nothing about this on 2026-11-03"
    lines += [f"  problem: {p}" for p in view["problems"]] + [f"  note: {n}" for n in view["notes"]]
    return lines


def quote_lines(view: dict, intent=None) -> list[str]:
    """What the session prints for a request that only asked for the current price (entry 150, Q8): what was understood, then one line for each stock (symbol, company, price, change, percent, when), what was not found and what went wrong. The ONE shape is
    `views.quote_view` and its words are `views.quote_lines`; nothing here is about decisions, trades or money."""
    lines = [RULE, "QUOTE  " + (views.quote_headline(view) or "no price"), RULE]
    if intent is not None:
        lines.append(f"  understood: {intent.describe()}")
    return lines + [f"  {line}" for line in views.quote_lines(view)] + [f"  note: {n}" for n in view["notes"]]


NODE_WORDS = {"account_manager": "the Account Manager read the request and made the work order", "data_clerk": "the Data Clerk found the companies and answered the work order",
              "advisor": "the Advisor ranked the companies", "critic": "the Critic reviewed each recommendation against the profile", "account_manager_decision": "the Account Manager decided what may trade",
              "account_manager_direct": "the Account Manager priced the trades you named and decided them (no recommendation, no Critic review)",
              "account_manager_funds": "the Account Manager raised the money by selling holdings (RaiseFunds), because the cash did not cover the purchase and you permitted it",
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
    """What the Account Manager decided, right after it did: each purchase with its final size and who said yes, the recommendations the Critic DENIED (returned to the user with the Critic's reason, design choices entry
    104: `decisions.denied_lines`), and whether anything goes on to the Trader. In Ask mode this is the answer to "what happens now?"."""
    from prometheus.agents.account_manager.decisions import denied_lines
    from prometheus.agents.critic.report import DECISION_WORDS

    lines, going = ([f"      {ignoring.SKIPPED_LINE}"] if ignoring.skipped(order) else []), 0      # the user chose to ignore the Critic (entry 148): said before the decisions
    for d in order.trade_decisions:
        what = (f"{money(d['dollars'])}, about {sh.text(d['shares'])} shares" + (f" at {money(d['price'])}" if d.get("price") else "") if d.get("dollars") else
                f"{sh.text(d['shares'])} shares, about {money(d['est_cost'])}" if d["shares"] and d["est_cost"] else "")
        lines.append(f"      {d['symbol']:<6} {DECISION_WORDS.get(d['will_trade'], d['will_trade'])}" + (f"  {what}" if what else ""))
        going += d["will_trade"] in ("yes_auto", "yes_user")
    lines += denied_lines(order)
    waiting = [d["symbol"] for d in order.trade_decisions if d["will_trade"] == "pending_user"]
    if going:
        lines.append(f"      -> {going} approved trade{'s' if going != 1 else ''} go on to the Trader.")
    else:
        lines.append("      -> nothing was approved to trade: the Trader will not run.")
    if waiting:
        lines.append(f"      {', '.join(waiting)} still wait for your answer: " + ("nobody was at the terminal to ask (Ask mode needs a person at the keyboard: run it in a terminal, or use --mode auto)." if mode == "ask" else "nobody answered."))
    return lines


def clean_up(args: argparse.Namespace, parts: Parts, account_id, created: bool, say) -> None:
    """--cleanup deletes the account this session created (a sample or an --account account is never deleted); without it the account is kept and the user is told how to see it."""
    if args.cleanup and created:
        try:
            parts.store.delete_account(account_id)
            say(f"  (account {account_id} was deleted: --cleanup)")
        except Exception as error:   # never hide the result for a clean-up that failed
            say(f"  (account {account_id} could not be deleted: {type(error).__name__})")
    elif created:
        say(f"  account {account_id} is kept: python -m prometheus.agents.account_manager.account_manager --account {account_id}")


@contextmanager
def always_raise_funds_for(flag: bool | None):
    """`--always-raise-funds` sets ALWAYS_RAISE_FUNDS to true for the session and puts it back as it was; without the flag (None) nothing is touched: the setting is what the environment and `.env` say."""
    if not flag:
        yield
        return
    had, before = "ALWAYS_RAISE_FUNDS" in os.environ, os.environ.get("ALWAYS_RAISE_FUNDS")
    os.environ["ALWAYS_RAISE_FUNDS"] = "true"
    try:
        yield
    finally:
        if had:
            os.environ["ALWAYS_RAISE_FUNDS"] = before
        else:
            os.environ.pop("ALWAYS_RAISE_FUNDS", None)


def with_always_raise_funds(function):
    """Run the session with its `--always-raise-funds` applied (the setting for the length of the session, put back after); the function keeps its own source and name (`functools.wraps`)."""
    @functools.wraps(function)
    def wrapper(args, *rest, **options):
        with always_raise_funds_for(getattr(args, "always_raise_funds", None)):
            return function(args, *rest, **options)
    return wrapper


@with_always_raise_funds
def run_session(args: argparse.Namespace, person: dict, parts: Parts, ask=input, out=print) -> int:
    say = lambda *lines: [out(line) for line in lines]
    try:
        account_id = open_account(parts, args, person)
        manager = parts.new_manager()
        manager.connect(parts.store, account_id)
        holder = manager.users.get()
        profile = manager.profile
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
                       with_trader=True, trader_factory=lambda order: parts.trader, approver=approver_for(parts.approver, ask, out), store_factory=lambda: parts.store, confirm=confirm, clarify=clarifier_for(ask, out),
                       **({"guard": parts.guard} if parts.guard is not None else {}))
    request = {"ask": {"text": args.prompt, "default_count": args.default_count, "fetch": args.fetch}, "account": {"account": account_id, "profile": profile}, "trade": {"mode": "paper", "book": True, "model": not args.no_trader_model},
               **({"ignore_critic": True} if getattr(args, "ignore_critic", False) else {}),      # the user's box for this run (entry 148): the desk skips the Critic's node; with it off the request is what it always was
               **({"refuse_news_fetch": True} if getattr(args, "refuse_news_fetch", False) else {})}      # TEST ONLY (entry 149): a news request does not fetch
    say("", RULE, "THE DESK  (LangGraph: Account Manager -> Data Clerk -> Advisor -> Critic -> Account Manager's decision -> Trader; a trade you name yourself goes from the Account Manager straight to its decision and the Trader)", RULE)
    final, shown = None, 0
    for state in graph.stream({"request": request, "order": None, "errors": [], "resolution": None, "funds": None, "path": [], "seconds": {}, "with_trader": True, "plan": None}, stream_mode="values"):
        final = state
        for name in state["path"][shown:]:
            say(f"  .. {NODE_WORDS.get(name, name)} ({state['seconds'].get(name, 0):.1f} s)")
            if name == "account_manager_decision" and state.get("order") is not None and state["order"].trade_decisions:
                say(*decision_lines(state["order"], profile.trade_mode))
            if name == "account_manager" and (state.get("plan") or {}).get("direct") and state.get("order") is not None and state["order"].data is not None:
                say(*[f"      {line}" for line in data_clerk_lines(state["order"], "the Data Clerk found", "the Data Clerk did not find")])
            if name == "account_manager_direct" and state.get("order") is not None:
                say(*[f"      {line}" for line in direct_section(state["order"])])
        shown = len(state["path"])
    order, plan = final["order"], final.get("plan")
    if order is None or plan is None:   # the request could not be read, was not confirmed, or the account could not be used: the desk stopped at its first node
        say("", "  " + ("; ".join(final["errors"]) or "the desk did not start"), *([f"  {CLARIFY}"] if final.get("clarify") else []))
        clean_up(args, parts, account_id, created, say)      # a run that stopped at its first node (a request refused by the security policy, or one that could not be read) still honours --cleanup
        return 2
    if plan.get("action") == "news":      # a request that only asked for the news (entry 149, N10): printed, saved with its data if asked, never kept for the Track Record, nothing about decisions, trades or money
        view = views.news_view(order, plan, parts.guard, account_id)
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save).write_text(dumps({"request": " ".join(args.prompt.split()), "intent": plan["intent"].model_dump(), "path": final["path"], "order": json.loads(order.model_dump_json())}, indent=1), encoding="utf-8")      # the news IS the result: the order keeps its `data`
            say(f"  (what the desk made is saved in {args.save})")
        say("", *news_lines(view, plan["intent"]))
        ok = order.status == "data_ready"
        if not ok:
            say("", f"  {CLARIFY}" if final.get("clarify") else "  the Data Clerk's answer did not pass its checks (the problems are above)")
        clean_up(args, parts, account_id, created, say)
        return 0 if ok else 1
    if plan.get("action") == "quote":      # a request that only asked for the current price (entry 150, Q8): printed, saved with its data if asked, never kept for the Track Record, nothing about decisions, trades or money
        view = views.quote_view(order, plan, parts.guard, account_id)
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save).write_text(dumps({"request": " ".join(args.prompt.split()), "intent": plan["intent"].model_dump(), "path": final["path"], "order": json.loads(order.model_dump_json())}, indent=1), encoding="utf-8")      # the quote IS the result: the order is saved WITH its data
            say(f"  (what the desk made is saved in {args.save})")
        say("", *quote_lines(view, plan["intent"]))
        ok = order.status == "data_ready"
        if not ok:
            say("", f"  {CLARIFY}" if final.get("clarify") else "  the Data Clerk could not give the price (the problems are above)")
        clean_up(args, parts, account_id, created, say)
        return 0 if ok else 1
    intent, direct = plan["intent"], bool(plan.get("direct"))
    find, result = (None, None) if direct else (plan["find"], plan["result"])
    if result is not None:
        result.order = order
    proving_sessions.save_session(args.prompt, intent, final, account_id=account_id, screen=screen_of(parts.guard, account_id))      # the Track Record keeps every run that reached the Advisor (never a failed run if it cannot)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(dumps({"request": " ".join(args.prompt.split()), "intent": intent.model_dump(), "path": final["path"], "order": json.loads(order.model_dump_json(exclude=set() if plan.get("direct") else {"data"}))}, indent=1), encoding="utf-8")      # a direct order's `data` is only what the Data Clerk found (the company name and the symbol of each stock): it is kept; a research order's is large and is left out
        say(f"  (what the desk made is saved in {args.save})")
    buying = intent.action in ("buy", "order")
    stopped = direct and bool(final.get("errors"))     # a direct order whose stocks could not be made (entry 102): the desk ended at its first node, and the user is told why and asked to clarify the request
    if not direct:
        say("", *status_lines(result, intent, find))
    if neutral_lines(order):
        say("", RULE, *neutral_lines(order), RULE)
    if buying and not stopped:
        say("", *trade_lines(order))
    after = manager.list_portfolio()
    if buying and sized_purchases(order) and not direct:      # entry 123: a run of neutral companies alone has no review and still sized its purchases
        say("", *use_of_money_lines(order, before, after))
    say("", *account_block("ACCOUNT  after the session", holder, after, profile))

    # ---- the verdict
    did = changed(before, after)
    if stopped:
        say("", RULE, "SESSION  nothing was placed: the order could not be made as you typed it. Why:", *[f"  - {reason}" for reason in final["errors"]], "", f"  {CLARIFY}", RULE)
        ok = args.allow_no_trade
    elif not buying:
        say("", RULE, "SESSION  a look only: nothing was to be bought. The portfolio is as it was.", RULE)
        ok = True
    elif not did and waiting_lines_of(order):      # a direct order whose limit price is not met WAITS as a day order, its cash or shares set aside (entry 140)
        say("", RULE, "SESSION  WAITING: the order waits in the market for its limit price; nothing was traded yet.", *[f"  - {line}" for line in waiting_lines_of(order)], *watcher_note(parts), RULE)
        ok = True
    elif did:
        bought = [f"{sh.signed(sh.difference(shares, holdings_of(before).get(symbol, 0)))} {symbol}" for symbol, shares in holdings_of(after).items() if not sh.same(shares, holdings_of(before).get(symbol, 0))]      # `+d` crashed on a fraction (entry 110)
        sold = [f"{sh.signed(sh.difference(0, shares))} {symbol}" for symbol, shares in holdings_of(before).items() if symbol not in holdings_of(after)]
        say("", RULE, f"SESSION  PASSED: the portfolio changed ({', '.join(bought + sold)}); cash {money(before.cash_total)} -> {money(after.cash_total)}", *[f"  - {line}" for line in waiting_lines_of(order)], *(watcher_note(parts) if waiting_lines_of(order) else []), RULE)
        ok = True
    else:
        say("", RULE, "SESSION  the portfolio did NOT change. Why:", *[f"  - {reason}" for reason in why_not(result, order)], RULE)
        ok = args.allow_no_trade
    clean_up(args, parts, account_id, created, say)
    return 0 if ok else 1


def main(argv: list[str] | None = None, parts: Parts | None = None, ask=input, out=print) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    langsmith_tracing.start()   # LangSmith traces the run when LANGSMITH_API_KEY is set (entry 114)
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
