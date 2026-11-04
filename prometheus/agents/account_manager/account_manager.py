"""The Account Manager, as an LLM agent: RaiseFunds, a Tree of Thought for "what do I sell to buy this?" (design choices entry 42).

    am = AccountManager()                                           # the LLM of the `account_manager` role (ACCOUNT_MANAGER_MODEL, openai/gpt-oss-120b by default)
    result = am.raise_funds(load_portfolio("configs/portfolios/jacob_pyrett.json"), "MU", 2)
    print(report(result))

    python -m prometheus.agents.account_manager.account_manager --buy MU --shares 2          # LIVE: Finnhub quotes + OpenRouter calls (paid)

The pattern is the Advisor's (prometheus/agents/advisor/advisor.py): a class that takes a chat model (a scripted one in tests) or builds the role's model with `get_chat_model`, one
public method that does the job, a prompt module, a JSON reply that is parsed defensively with one retry, and CODE that checks everything the model says. Here the model is the thought
generator of a tree search and the checks, the scores, the pruning and the final choice are the program's (prometheus/reasoning/raise_funds.py).

What `raise_funds` does, in order:
  1. asks the DATA CLERK for the current price of the target and of every holding: a work order with `action_type="fetch_quote"`, answered by `DataClerk.request_quotes` (the
     Data Clerk's `get_quote` tool, one Finnhub call per symbol). The Account Manager never reads a price from anywhere else, and a missing quote ends the request.
  2. builds the problem (the holdings, what is sellable under T+1, the cost of the purchase) and runs the tree search, calling the model once per node it expands.
  3. returns the result as one dict: the chosen plan, the alternatives, the reasoning trace, the quotes it used, and what the model run cost.

This module is also the "account_manager_funds" node of the desk graph (prometheus/langgraph/desk.py, `build_desk(with_funds=True)`).
"""
import argparse
import dataclasses
import json
import math
import re
import sys
import threading
import time
from datetime import date
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prometheus.common.portfolio import Portfolio, after_purchase, load_portfolio
from prometheus.common.work_order import new_work_order
from prometheus.models.chat import get_chat_model, is_rate_limited
from prometheus.models.chat import model_name as chat_model_name
from prometheus.prompts.account_manager import RETRY, system_prompt, user_message
from prometheus.rag.loaders import check_symbol
from prometheus.reasoning import raise_funds as rf

DEFAULT_PORTFOLIO = Path(__file__).resolve().parents[3] / "configs" / "portfolios" / "jacob_pyrett.json"
REQUESTER = "account_manager"


def parse_thoughts(text: str) -> list:
    """The `thoughts` list from the model's reply. Raises ValueError, with a short reason, if there is none."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    try:
        data = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        raise ValueError("not valid JSON") from None
    if isinstance(data, dict) and isinstance(data.get("plans"), list):   # whole plans (entry 50)
        return data["plans"]
    if not isinstance(data, dict) or not isinstance(data.get("thoughts"), list):
        raise ValueError("no thoughts list")
    return data["thoughts"]


def top_recommendation(order) -> str | None:
    """The symbol to buy from a work order the Advisor has answered: the recommendation with the highest CALCULATED score, if that score is above 50 (more verified bullish signals than
    bearish ones). None when there is none. Code, not a model: the ranking of the design ("recommendations are ranked in order of the final score")."""
    ranked = [r for r in order.recommendations if r.get("supported") and isinstance(r.get("score"), (int, float)) and r["score"] > 50]
    return max(ranked, key=lambda r: r["score"])["symbol"] if ranked else None


PREFERRED_PROVIDERS = ("cerebras", "groq")   # the fast hosts of gpt-oss-120b on OpenRouter (entry 50)


class AccountManager:
    def __init__(self, model=None, model_id: str | None = None, clerk=None, effort: str | None = "auto", only_green: bool = True, never_liquidate: bool = True, green_first: bool = True,
                 explore: bool = True, lean: bool = True, whole_plans: bool = False, parallel: bool = True, up_to: bool = True, deadline: float | None = 90.0,
                 providers: tuple[str, ...] | str | None = "auto"):
        """model: a chat model (anything with `invoke`; a scripted one in tests); otherwise OpenRouter through `get_chat_model` with the model of the `account_manager` role
        (ACCOUNT_MANAGER_MODEL), or `model_id`. clerk: the Data Clerk that is asked for prices (anything with `request_quotes`; a fake in tests); built when first needed.
        effort (entry 46): "low", "medium" or "high", OpenRouter's `reasoning` effort for a model that thinks before it answers (gpt-oss-120b does): a thought is three short JSON steps and most of a
        call's time is the model thinking, so "low" shortens every call. None leaves the model's own default; "auto" (the default) is "low" for gpt-oss and None for any other model (it made
        qwen3-30b slower and worse, entry 46); higher efforts only cost time (entry 50).
        only_green and never_liquidate (entry 47): the rules of the evaluator, ON unless you turn them off. `only_green`: only a holding whose price is above its cost basis may be sold, so no sale takes
a loss. `never_liquidate`: at least one share of every holding is kept, so no position is sold out. They are the defaults of every method below; a method's own argument overrides them.

        THE DEFAULTS ARE THE WINNING SETTINGS (entry 50), all on: `green_first` (the score prefers the most profitable holdings), `explore` (the beam ranks by score and progress, and a thought of
        one kind is asked again), `lean` (no sell of more shares than needed), `parallel` (the plans of a level call the model at the same time), `up_to` (a number of shares means up to that many)
        and `deadline` (90 seconds a purchase, then the best plan so far; None for no limit). NOT a default: `whole_plans` (a thought is a whole plan, one to three rounds of the model): measured
        slower, 121 s against 33 to 55 s for the same purchase (entry 50). `AccountManager.classic()` turns all of them off: the behaviour of entries 42 to 47, which the older tests use.

        providers (entry 50): the OpenRouter providers to prefer, in order (the others are the fallback). "auto" (the default) is Cerebras then Groq for gpt-oss and OpenRouter's own choice for any other
        model; None is OpenRouter's own choice. Measured: the same gpt-oss-120b answers at about 55 tokens a second by default, 650 on Cerebras and 370 on Groq: the largest single gain in time.
        The model is the same; only where it runs changes (the provider sees the prompt: prices, holdings and quantities, no keys)."""
        is_oss = "gpt-oss" in (model_id or chat_model_name("account_manager"))
        if effort == "auto":
            effort = "low" if is_oss else None
        if providers == "auto":
            providers = PREFERRED_PROVIDERS if is_oss else None
        extra = {**({"reasoning": {"effort": effort}} if effort else {}), **({"provider": {"order": list(providers), "allow_fallbacks": True}} if providers else {})}
        self.model = model or get_chat_model("account_manager", model=model_id, **({"extra_body": extra} if extra else {}))
        self._clerk = clerk
        self.only_green, self.never_liquidate = only_green, never_liquidate
        self.green_first, self.explore, self.lean, self.whole_plans, self.parallel, self.up_to, self.deadline = green_first, explore, lean, whole_plans, parallel, up_to, deadline
        self._lock = threading.Lock()  # the plans of a level may call the model at the same time (parallel): the counters of the run are shared
        self.run: dict = {}

    @classmethod
    def classic(cls, model=None, clerk=None, **options):
        """An Account Manager with every default of entries 45 to 50 turned off (the rules of entries 42 to 44: a loss may be taken, a position may be sold out, one sell per thought, exact
        numbers of shares, no limit, one call after the other). The tests of the earlier entries use it; `options` override any argument."""
        return cls(model=model, clerk=clerk, **{"effort": None, "only_green": False, "never_liquidate": False, "green_first": False, "explore": False, "lean": False, "whole_plans": False,
                                                 "parallel": False, "up_to": False, "deadline": None, "providers": None, **options})

    @property
    def clerk(self):
        if self._clerk is None:
            from prometheus.agents.data_clerk.data_clerk_agent import DataClerk  # imported here: the Data Clerk needs the toolbox's dependencies and builds its own model
            self._clerk = DataClerk()
        return self._clerk

    # ---- the Data Clerk: prices

    def request_prices(self, symbols: list[str]):
        """Ask the Data Clerk for the current price of each symbol (the design's `DataClerk.request_data`, for the `fetch_quote` action). Returns the work order, with `data.quotes`."""
        order = new_work_order(f"Current prices of {', '.join(symbols)}", requester=REQUESTER, action_type="fetch_quote", subject=list(symbols), live=True)
        return self.clerk.request_quotes(order)

    # ---- the thought generator

    def _generate(self, problem: rf.Problem, node: rf.Node, rejected: list[str]) -> list:
        """One expansion: the model's next steps for this node. One retry if the reply is not the JSON asked for. Raises if the model call fails (the search records it)."""
        messages = [SystemMessage(content=system_prompt(problem.branching, problem.only_green, problem.fill, problem.lean, problem.explore, problem.max_depth, problem.green_first, problem.never_liquidate, problem.whole_plans)), HumanMessage(content=user_message(problem, node, rejected))]
        problem_text = ""
        for _ in range(2):
            reply = self.model.invoke(messages)
            usage = getattr(reply, "usage_metadata", None) or {}
            with self._lock:
                self.run["calls"] += 1
                self.run["tokens_in"] += usage.get("input_tokens", 0)
                self.run["tokens_out"] += usage.get("output_tokens", 0)
            text = reply.content if isinstance(reply.content, str) else ""
            try:
                return parse_thoughts(text)
            except ValueError as error:
                problem_text = str(error)
                with self._lock:
                    self.run["retries"] += 1
                messages = [*messages, AIMessage(content=text), HumanMessage(content=RETRY.format(problem=problem_text))]
        raise ValueError(f"the model's reply was not usable ({problem_text})")

    # ---- RaiseFunds

    @staticmethod
    def _check_purchase(target, shares) -> str:
        """The cleaned target symbol. Raises ValueError, with the reason, if the purchase makes no sense."""
        if not isinstance(target, str):
            raise ValueError("the symbol to buy must be text, like 'MU'")
        target = check_symbol(target.strip().lstrip("$"))
        if shares is not None and (not isinstance(shares, int) or isinstance(shares, bool) or shares < 1):
            raise ValueError("the number of shares to buy must be a whole number of at least 1 (or left out, to buy as many as possible)")
        return target

    def _resolve(self, explore, lean, green_first, up_to, whole_plans, limits: dict):
        """The arguments that were not given, from the Account Manager's own settings (the defaults of entry 50); `parallel` and `deadline` go into the limits of the search unless they are there."""
        limits = dict(limits)
        limits.setdefault("parallel", self.parallel)
        if self.deadline and "deadline" not in limits:
            limits["deadline"] = self.deadline
        pick = lambda value, own: own if value is None else value
        return pick(explore, self.explore), pick(lean, self.lean), pick(green_first, self.green_first), pick(up_to, self.up_to), pick(whole_plans, self.whole_plans), limits

    def _new_run(self) -> None:
        self.run = {"model": getattr(self.model, "model_name", None), "calls": 0, "retries": 0, "tokens_in": 0, "tokens_out": 0, "error": None, "rate_limited": False}

    def _prices(self, symbols: list[str]) -> tuple[dict, dict | None, list[str]]:
        """(the quotes by symbol, the quote work order as JSON, the reasons it failed: empty when every symbol was quoted)."""
        order = self.request_prices(list(dict.fromkeys(symbols)))
        quotes = dict(order.data.quotes) if order.data else {}
        failed = [] if order.status == "data_ready" else (order.errors or ["the Data Clerk returned no quotes"])
        return quotes, json.loads(order.model_dump_json()), failed

    def _plan(self, portfolio: Portfolio, target: str, shares: int | None, quotes: dict, today: date | None, only_green: bool, fill: str | None = None, explore: bool = False,
              lean: bool = False, green_first: bool = False, up_to: bool = False, never_liquidate: bool = False, auto_depth: bool = True, whole_plans: bool = False, **limits) -> dict:
        """The tree search for one purchase, with the quotes already in hand. Never raises for a model problem. `limits` are the size of the tree: `max_depth` and `branching` (also told to the
        model), `beam_width` and `max_calls`.

        HOW MANY SHARES (entry 45). `shares` is X. With X and `up_to` false the purchase is exactly X (as before). With `up_to` true it is as many as the portfolio can pay for, at most X. With no X
        (None) it is as many as the portfolio can pay for: `Problem.max_affordable`, the arithmetic of cash plus everything that may be sold, so the plan has to sell nearly all of it and the tree is
        given a depth of as many steps as there are holdings that can be sold. WHAT TO SELL is always decided by the tree. If not even one share can be paid for, nothing is sold."""
        limits = dict(limits)
        sizes = {k: limits.pop(k) for k in ("max_depth", "branching") if limits.get(k)}
        prices = {s: q["price"] for s, q in quotes.items()}
        probe = rf.build_problem(portfolio, target, 1, prices, today, only_green, fill, never_liquidate=never_liquidate)
        cap = probe.max_affordable
        resolved = cap if shares is None else min(shares, cap) if up_to else shares
        info = {"requested": shares, "up_to": up_to, "resolved_shares": resolved, "most_affordable": cap}
        if shares is None:  # everything usable has to go: a plan as deep as the holdings, and a budget to match
            eligible = sum(probe.eligible(h) for h in probe.holdings.values())
            sizes.setdefault("max_depth", max(eligible, 1))
            limits.setdefault("max_calls", 12 + 6 * max(eligible - 3, 0))
        if whole_plans:   # a level is a ROUND of the model, and a round can hold any number of sells: three rounds are enough (propose, then add to the best plans, then once more)
            sizes.setdefault("max_depth", 3)
            auto_depth = False
        if resolved < 1:  # not even one share can be paid for, even by selling everything that may be sold: nothing is sold, and no model is asked
            problem = rf.build_problem(portfolio, target, 1, prices, today, only_green, fill, explore, lean, green_first=green_first, never_liquidate=never_liquidate, whole_plans=whole_plans)
            return {**rf.answer(problem, [], {"trace": [], "calls": 0, "nodes": 1, "errors": [],
                                              "stop": "selling everything that may be sold cannot pay for even one share: nothing is sold"}, rf.effective_limits(problem)), **info}
        problem = rf.build_problem(portfolio, target, resolved, prices, today, only_green, fill, explore, lean, green_first=green_first, never_liquidate=never_liquidate, whole_plans=whole_plans, **sizes)
        if "max_depth" not in sizes and auto_depth:  # not asked for: the depth the purchase needs (entry 49), from arithmetic on the holdings: the fewest stocks that could pay for it, plus two
            problem = dataclasses.replace(problem, max_depth=problem.auto_depth())
            limits.setdefault("max_calls", max(rf.MAX_LLM_CALLS, math.ceil(1.6 * (1 + limits.get("beam_width", rf.BEAM_WIDTH) * (problem.max_depth - 1)))))   # a deeper tree needs the calls to reach its depth, and room for the re-asks
            info["auto_depth"] = True
        try:
            return {**rf.search(problem, lambda prob, node, rejected: self._generate(prob, node, rejected), **limits), **info}
        except Exception as error:  # the search catches a failing model call; anything else is a bug we still must not hide behind a traceback in the middle of a desk run
            self.run["error"], self.run["rate_limited"] = f"the search failed ({type(error).__name__})", is_rate_limited(error)
            return {"status": "error", "errors": [self.run["error"]], "target": target, "shares": resolved, **info}

    def raise_funds(self, portfolio: Portfolio, target: str, shares: int | None, today: date | None = None, only_green: bool | None = None, fill: str | None = None, explore: bool | None = None,
                    lean: bool | None = None, green_first: bool | None = None, up_to: bool | None = None, never_liquidate: bool | None = None, whole_plans: bool | None = None, **limits) -> dict:
        """Buy `shares` of `target`: find what to sell. Returns the answer of `reasoning.raise_funds.search` plus `quotes` (what the Data Clerk returned), `run` (the model, calls and
        tokens) and `prices_order` (the quote work order, as plain JSON). `status` is `plan`, `funded`, `infeasible`, `no_prices` (the Data Clerk could not quote every symbol: nothing is
        guessed and the model is not called) or `refused` (the request itself is wrong: the reason is in `errors`). `only_green`: only holdings that show a profit may be sold. A stock is
        never sold to pay for more of itself. `fill` (entry 44): a second stock bought with the cash LEFT after the purchase, as many shares as it affords; it is never sold, the sells are lean (no
        step sells more shares than the shortfall needs), and if selling everything cannot pay for `shares` of `target` the status is `infeasible` and nothing is sold. Never raises for a model, API
        or input problem. `shares` None: as many as the portfolio can pay for; `up_to`: as many as it can pay for, at most `shares` (see `_plan`); `green_first`: the score prefers selling the most
        profitable (highest percent gain) holdings first (entry 45). `explore` (entry 44): the generator must give both kinds of step, COVER and PARTIAL, and is asked again (once per node, within the budget) when a kind is missing."""
        try:
            target = self._check_purchase(target, shares)
            if fill is not None:
                fill = self._check_purchase(fill, 1)
                if fill == target:
                    raise ValueError("the stock to fill with must be another symbol than the stock to buy")
        except ValueError as error:
            return {"status": "refused", "errors": [str(error)], "quotes": {}, "run": {}}
        only_green = self.only_green if only_green is None else only_green
        never_liquidate = self.never_liquidate if never_liquidate is None else never_liquidate
        explore, lean, green_first, up_to, whole_plans, limits = self._resolve(explore, lean, green_first, up_to, whole_plans, limits)
        self._new_run()
        quotes, prices_order, failed = self._prices([target, *([fill] if fill else []), *portfolio.symbols])
        out = {"quotes": quotes, "prices_order": prices_order, "run": self.run}
        if failed:
            return {**out, "status": "no_prices", "errors": failed}
        result = self._plan(portfolio, target, shares, quotes, today, only_green, fill, explore, lean, green_first, up_to, never_liquidate, whole_plans=whole_plans, **limits)
        return {**out, **result}

    def raise_funds_many(self, portfolio: Portfolio, purchases: list[tuple[str, int]], today: date | None = None, only_green: bool | None = None, explore: bool | None = None, lean: bool | None = None,
                         green_first: bool | None = None, up_to: bool | None = None, never_liquidate: bool | None = None, whole_plans: bool | None = None, **limits) -> dict:
        """Several purchases, one after the other (entry 43): the Data Clerk is asked ONCE for every price (the targets and the holdings); each purchase is then planned against the portfolio
        the one before left, with the shares it bought marked as bought today (so they cannot be sold in the next plan, the T+1 rule) and its cash carried on. A purchase that cannot be paid for
        (`infeasible`, `error`) changes nothing and the next one is still attempted. Returns {status, purchases: [one result each, with `target` and `shares`], quotes, final: the portfolio at the
        end as plain JSON, run, prices_order}. `status` is `done` (every purchase was attempted), `no_prices` or `refused`. The portfolio given is not changed."""
        try:
            cleaned = [(self._check_purchase(t, n), n) for t, n in purchases]
            if not cleaned:
                raise ValueError("give at least one purchase")
            if len(cleaned) > 1 and any(n is None for _, n in cleaned):
                raise ValueError("'as many as possible' (no number of shares) works for ONE stock: with several, give the number of shares of each (use up_to to buy as many as affordable, at most that number)")
        except ValueError as error:
            return {"status": "refused", "errors": [str(error)], "purchases": [], "quotes": {}, "run": {}}
        only_green = self.only_green if only_green is None else only_green
        never_liquidate = self.never_liquidate if never_liquidate is None else never_liquidate
        explore, lean, green_first, up_to, whole_plans, limits = self._resolve(explore, lean, green_first, up_to, whole_plans, limits)
        self._new_run()
        quotes, prices_order, failed = self._prices([*(t for t, _ in cleaned), *portfolio.symbols])
        out = {"quotes": quotes, "prices_order": prices_order, "run": self.run, "purchases": [], "start": json.loads(portfolio.model_dump_json())}
        if failed:
            return {**out, "status": "no_prices", "errors": failed}
        current = portfolio
        for target, shares in cleaned:
            result = self._plan(current, target, shares, quotes, today, only_green, None, explore, lean, green_first, up_to, never_liquidate, whole_plans=whole_plans, **limits)
            out["purchases"].append({"target": target, "shares": shares, **result})
            if result["status"] in ("plan", "funded"):
                sells = [(x["symbol"], x["shares"]) for x in (result["plan"]["sells"] if result["plan"] else [])]
                current = after_purchase(current, target, result["shares"], result["price"], sells, result["after"]["cash"], today)   # the shares BOUGHT (up to X may be fewer than X), not the shares asked
        return {**out, "status": "done", "final": json.loads(current.model_dump_json())}


def report(result: dict) -> str:
    """What the user reads for a RaiseFunds request: the whole answer, or the reason there is none."""
    if result["status"] in ("refused", "no_prices", "error"):
        lines = [f"RAISE FUNDS  {result['status'].upper().replace('_', ' ')}: " + "; ".join(result.get("errors") or ["no reason given"])]
        if result.get("quotes"):
            lines.append("quotes that did come back: " + ", ".join(f"{s} {q['price']:.2f}" for s, q in result["quotes"].items()))
        return "\n".join(lines)
    run = result["run"]
    text = rf.format_answer(result, quotes=result["quotes"])
    return text + f"\nMODEL    {run['model']}: {run['calls']} call(s), {run['retries']} retr{'y' if run['retries'] == 1 else 'ies'}, {run['tokens_in']} tokens in, {run['tokens_out']} out"


def _shares_table(start: dict, final: dict) -> list[str]:
    """Before and after, per symbol (shares and average cost basis), and the cash: the whole run's effect on the portfolio."""
    before = {p["symbol"]: p for p in start["positions"]}
    after = {p["symbol"]: p for p in final["positions"]}
    rows = [[sym, str(before[sym]["shares"]) if sym in before else "0", str(after[sym]["shares"]) if sym in after else "0",
             f"${after[sym]['cost_basis']:,.2f}" if sym in after else "-"] for sym in [*before, *(s for s in after if s not in before)]]
    rows.append(["cash", f"${start['cash_avail']:,.2f}", f"${final['cash_avail']:,.2f}", ""])
    return rf._table(["symbol", "before", "after", "cost basis after"], rows, right={1, 2, 3})


def report_many(result: dict) -> str:
    """What the user reads for several purchases in a row: each purchase as `report` shows one (the quotes once, at the top), then a summary of the run."""
    if result["status"] != "done":
        return report(result)
    lines, n = [], len(result["purchases"])
    for i, p in enumerate(result["purchases"], 1):
        lines.append(f"\n{'=' * 100}\nPURCHASE {i} of {n}: buy {p['shares']} x {p['target']}\n{'=' * 100}")
        lines.append(rf.format_answer(p, quotes=result["quotes"] if i == 1 else None) if p["status"] in ("plan", "funded", "infeasible")
                     else f"{p['status'].upper()}: " + "; ".join(p.get("errors") or ["no reason given"]))
    rows = []
    for i, p in enumerate(result["purchases"], 1):
        sold = ", ".join(f"{x['symbol']} x{x['shares']}" for x in (p["plan"]["sells"] if p.get("plan") else [])) or "-"
        rows.append([str(i), f"{p['target']} x{p['shares']}", p["status"].upper(), sold, f"${p['cost']:,.2f}" if "cost" in p else "-",
                     f"${p['after']['cash']:,.2f}" if p.get("after") else "-", str(p["plan"]["score"]) if p.get("plan") else "-"])
    lines += [f"\n{'=' * 100}\nSUMMARY OF {n} PURCHASES (in order; each was planned against what the one before left)\n{'=' * 100}",
              *rf._table(["#", "buy", "result", "sold to pay for it", "cost", "cash after", "score"], rows, right={4, 5, 6}),
              "", "PORTFOLIO, start of the run and end of it", *_shares_table(result["start"], result["final"])]
    run = result["run"]
    lines.append(f"\nMODEL    {run['model']}: {run['calls']} call(s), {run['retries']} retr{'y' if run['retries'] == 1 else 'ies'}, {run['tokens_in']} tokens in, {run['tokens_out']} out")
    return "\n".join(lines)


def parse_purchase(text: str, default_shares: int | None) -> tuple[str, int | None]:
    """"MU:2" -> ("MU", 2); "MU" -> ("MU", default_shares), which is None, as many as possible, unless --shares was given. Raises ValueError, with the reason, for anything else."""
    symbol, _, count = text.partition(":")
    if count and not count.strip().isdigit():
        raise ValueError(f"{text!r}: the number of shares after the colon must be a whole number")
    return symbol.strip(), int(count) if count else default_shares


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RaiseFunds: what to sell from a portfolio to buy one or more stocks, by a Tree of Thought. LIVE: it calls Finnhub for the quotes and OpenRouter for the model.")
    parser.add_argument("--buy", required=True, action="append", help="the symbol to buy, for example MU, or MU:2 for 2 shares; repeat it for several purchases (done in order)")
    parser.add_argument("--shares", type=int, default=None, help="how many shares of a --buy that has no :N. Left out: AS MANY AS THE PORTFOLIO CAN PAY FOR (one stock only; this can sell nearly everything)")
    parser.add_argument("--exact", action="store_true", help="a number of shares is exact: if the portfolio cannot pay for it, nothing is sold. Without this, N means UP TO N: as many as it can pay for, at most N")
    flag = lambda name, text: parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=True, help=text + " (on by default; --no-" + name + " turns it off)")
    flag("green-first", "the score prefers selling the most profitable holdings first (highest percent gain) instead of realising the least taxable gain")
    flag("explore", "the beam ranks by score and progress, and a thought of one kind only is asked again")
    flag("lean", "no sell of more shares than the shortfall needs")
    parser.add_argument("--whole-plans", action="store_true", help="a thought is a whole plan (a list of sells): one to three rounds of the model for a purchase. TRIED AND SLOWER (entry 50): off by default")
    parser.add_argument("--portfolio", default=str(DEFAULT_PORTFOLIO), help="a portfolio JSON file (default: the sample portfolio of Jacob Pyrett)")
    parser.add_argument("--only-green", action="store_true", help="(on by default since entry 47) sell only holdings that show a profit: price above cost basis, so no sale takes a loss")
    parser.add_argument("--allow-loss", action="store_true", help="turn off the rule that only holdings in profit are sold: a sale may take a loss")
    parser.add_argument("--allow-liquidate", action="store_true", help="turn off the rule that one share of every holding is kept: a position may be sold out")
    parser.add_argument("--sequential", action="store_true", help="expand the plans of a level one after the other (the default is to call the model for all of them at the same time: same result, less waiting)")
    parser.add_argument("--effort", choices=("low", "medium", "high"), help="the model's reasoning effort (OpenRouter). Default: low for gpt-oss, the model's own for any other. Medium and high only cost time: measured, the same plan")
    parser.add_argument("--max-depth", type=int, help="the most levels: with --no-whole-plans the most stocks sold in a plan (default: chosen from the purchase); with whole plans the most rounds (default 3)")
    parser.add_argument("--branching", type=int, help="the most thoughts asked of the model for each plan (default 3)")
    parser.add_argument("--beam", type=int, help="how many unfinished plans are expanded per level (default 3)")
    parser.add_argument("--max-calls", type=int, help="the most model calls per purchase (default: chosen from the depth)")
    parser.add_argument("--deadline", type=float, help="seconds a purchase may take (default 90; 0 for no limit): after it no new level is started and the best plan COMPLETED so far is the answer (none yet: nothing is sold). A level in flight is waited for")
    parser.add_argument("--fill", metavar="SYMBOL", help="with ONE --buy: also buy as many shares of this stock as the cash LEFT after the purchase affords (the sells are lean: only as much as the purchase needs; "
                        "if selling everything cannot pay for the purchase, nothing is sold)")
    parser.add_argument("--providers", help="the OpenRouter providers to prefer, comma separated, others as fallback (default for gpt-oss: cerebras,groq, about ten times faster than OpenRouter's own choice; 'none': its own choice)")
    parser.add_argument("--model", help="an OpenRouter model id instead of ACCOUNT_MANAGER_MODEL")
    parser.add_argument("--json", action="store_true", help="print the whole result as JSON instead of the report")
    args = parser.parse_args(argv)
    try:
        purchases = [parse_purchase(text, args.shares) for text in args.buy]
        portfolio = load_portfolio(args.portfolio)
        agent = AccountManager(model_id=args.model, effort=args.effort or "auto", only_green=not args.allow_loss, never_liquidate=not args.allow_liquidate, green_first=args.green_first,
                               explore=args.explore, lean=args.lean, whole_plans=args.whole_plans, parallel=not args.sequential, up_to=not args.exact,
                               deadline=(None if args.deadline == 0 else args.deadline or 90.0),
                               providers="auto" if not args.providers else None if args.providers == "none" else tuple(args.providers.split(",")))
    except (ValueError, OSError, RuntimeError) as error:  # a bad purchase or portfolio file, or a missing key: the message names the setting, never a value
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    amount = lambda n: str(n) if n else "as many as possible"
    if args.fill and len(purchases) != 1:
        print("ERROR: --fill goes with exactly one --buy", file=sys.stderr)
        return 1
    print(f"RAISE FUNDS for {portfolio.user}: buy {', then '.join(f'{amount(n)} x {s.upper()}' for s, n in purchases)}{f', then as many {args.fill.upper()} as the cash left buys' if args.fill else ''}; holdings {', '.join(f'{p.symbol} x{p.shares}' for p in portfolio.positions)}; "
          f"free cash ${portfolio.cash_avail:,.2f}{'' if args.allow_loss else '; only holdings in profit may be sold (no loss)'}{'' if args.allow_liquidate else '; one share of every holding is kept'}; model {getattr(agent.model, 'model_name', '?')}\n"
          f"(live: one Finnhub quote per symbol, then at most {rf.MAX_LLM_CALLS} model calls per purchase)\n")
    limits = {k: v for k, v in (("max_depth", args.max_depth), ("branching", args.branching), ("beam_width", args.beam), ("max_calls", args.max_calls)) if v}
    started = time.perf_counter()
    if len(purchases) == 1:
        result = agent.raise_funds(portfolio, *purchases[0], fill=args.fill, **limits)
        print(rf.as_json(result) if args.json else report(result))
        print(f"TIME     {time.perf_counter() - started:.0f} seconds ({'sequential' if args.sequential else 'the plans of a level in parallel'}{', reasoning effort ' + args.effort if args.effort else ''}{'' if args.whole_plans else ', one sell per thought'})")
        return 0 if result["status"] in ("plan", "funded") else 1
    result = agent.raise_funds_many(portfolio, purchases, **limits)
    print(rf.as_json(result) if args.json else report_many(result))
    print(f"TIME     {time.perf_counter() - started:.0f} seconds ({'sequential' if args.sequential else 'the plans of a level in parallel'}{', reasoning effort ' + args.effort if args.effort else ''}{'' if args.whole_plans else ', one sell per thought'})")
    return 0 if result["status"] == "done" and all(p["status"] in ("plan", "funded") for p in result["purchases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
