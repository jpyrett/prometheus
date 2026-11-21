"""RaiseFunds: a Tree of Thought that decides what to sell to buy a stock (design choices entry 42, design/week4_tot_guide.md).

THE PROBLEM. The Account Manager is asked to buy `shares` of a target stock (say 2 of MU). The user holds other stocks and no cash, so something must be sold first. Which stocks, how
many shares of each? There is no single obvious answer: selling the loser saves tax but may need two orders; selling one stock in full is one order but closes a position; selling
a little of everything keeps every position but means three orders. The options have to be built up and compared, which is what a chain of thought cannot do.

THE TREE.
  state       the root is "nothing sold yet": the shortfall is the whole cost of the purchase.
  thought     ONE step: "sell n shares of SYMBOL" (and the idea behind it, in words). A node is the plan so far: the steps from the root down to it.
  branching   up to BRANCHING (3) different next steps for every node that is expanded, proposed by the LLM (the thought generator).
  depth       at most MAX_DEPTH (3) steps: one sell line per holding, and the portfolio has three holdings. A node whose proceeds cover the cost is COMPLETE: it is never expanded.
  search      beam search: after every level only the BEAM_WIDTH (3) best unfinished nodes are expanded; the rest are pruned and the trace says why. Completed plans are all kept.
  end         no unfinished node is left, or MAX_DEPTH steps were taken, or the call budget (MAX_LLM_CALLS) is used. Then the best completed plan is the answer; none means "infeasible".

WHO DOES WHAT. The LLM only proposes thoughts (`generate`, supplied by the Account Manager). CODE does everything a program can check (the rule of design choices entries 24 and 26):
  hard checks  the symbol is held and is not the target (a stock is never sold to pay for more of itself), it is not already in the plan, with `only_green` it shows a profit, the shares are a
               whole number within the sellable shares (T+1), and the branch can still reach the cost with what is left. A thought that fails is REJECTED, kept in the trace with its reason, and its reason is told to the LLM in its next call.
  the score    0 to 100, calculated from four components (WEIGHTS), never written by a model. See `score_plan`.
  pruning      the beam, by score. Ties go to fewer orders, then to the earlier node.
  the decision the completed plan with the highest score.
The tree is a LangGraph (`build_graph`): expand -> evaluate -> prune -> (expand again | decide). The graph state IS the tree: the frontier, the completed plans, the trace.

WHOLE SHARES ONLY (design choices entry 110). RaiseFunds plans WHOLE shares, as it always did: what the model may sell is "a whole number of at least 1", `Holding.sellable` is the WHOLE shares of what may be sold (the T+1 quantity
and, with `never_liquidate`, the holding less one share, never below 0: a holding of one share or less is not sold from), and `Problem.shares` is a whole number (the desk hands over the purchase's quantity rounded UP to the next
whole share, so that the plan raises at least enough). What changed is only that a holding it READS may have a fraction (`Holding.shares`, `held`, `fill_held`): a position is closed only when the plan sells exactly what is held.

DECIMAL MONEY (design choices entry 112; Jacob, 2026-11-16: *"Need to always use decimal.Decimal type when dealing with money and share counts"*). The cash, the prices, the cost, the proceeds, the gains and the shortfall are `Decimal`
(a price times shares is `trade_amount`, to the cent, half up; what is left is exact arithmetic on cents), and every number of shares is a whole `Decimal` quantity (the model's reply is read with a number that is not text or a bool). A float given to
`build_problem` is a TypeError: it comes in through a door (the Account Manager's `_plan` turns a quote into a `Decimal` price before it gets here). What stays a float is the EVALUATOR'S SCORE and its components, the weights and the beam's priority: they are
measurements of how good a plan is (0 to 100, fractions between 0 and 1), never added to money (design/decimal_money_spec.md section 3, last paragraph), and a ratio of two amounts is worked out from Decimals at the very end (`_ratio`), at that line.

WHAT IT DOES NOT MODEL: fees, taxes beyond "how much of the proceeds is a gain", the T+1 settlement of the SALE proceeds (the plan treats them as available for the purchase; the Trader's
settlement rules apply when the orders are placed), a limit order, or the price moving between the quote and the fill.
"""
import json
import threading
import time
import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from prometheus.common import shares as sh
from prometheus.common.money import json_default, exact
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.common.trace import traced
from prometheus.tools import trading_math as tm

BEAM_WIDTH = 3        # unfinished nodes kept (expanded) per level
BRANCHING = 3         # thoughts asked for per expanded node
MAX_DEPTH = 3         # steps in a plan
MAX_LLM_CALLS = 12    # the budget of a search: 1 at the root, then at most BEAM_WIDTH per level, plus the re-asks (entry 44; it was 8 before them)
WEIGHTS = {"gain": 0.40, "overshoot": 0.25, "keep": 0.20, "orders": 0.15}
WEIGHTS_GREEN = {"green": 0.40, "overshoot": 0.25, "keep": 0.20, "orders": 0.15}  # `green_first` (entry 45): the first component is "sell the most profitable first" instead of "realise the least gain"
KEPT_ALTERNATIVES = 3  # other completed plans shown with the answer
MAX_REJECTIONS_TOLD = 6  # the most recent rejection reasons the next call is told about
MAX_IDEA = 160


# ---------------------------------------------------------------- the problem

@dataclass(frozen=True)
class Holding:
    symbol: str
    name: str | None
    shares: Decimal        # what is held: a quantity, which may have a fraction (entry 110)
    sellable: Decimal      # WHOLE shares that may be sold today (T+1; with `never_liquidate` one share of the holding is kept too)
    price: Decimal         # the current quote, from the Data Clerk (a Decimal at four places: a quote passes the price door before it gets here)
    basis: Decimal         # average price paid per share

    @property
    def value(self) -> Decimal:
        return trade_amount(self.shares, self.price)      # money from a count and a price is `trade_amount` (decimal, half up to the cent), never a float product (entry 111)

    @property
    def green(self) -> bool:
        """The position shows a profit: the price is above the average price paid."""
        return self.price > self.basis


@dataclass(frozen=True)
class Problem:
    """What the search is about: buy `shares` of `target` at `price` with `cash`, selling from `holdings`."""
    target: str
    shares: Decimal                # a whole number of shares
    price: Decimal
    cash: Decimal
    holdings: dict[str, Holding]   # everything held EXCEPT the target: only these can be sold
    held: Decimal = Decimal(0)     # shares of the target already held (the purchase adds to them): a quantity, which may have a fraction
    only_green: bool = False       # sell only holdings that show a profit (price above cost basis): a hard check
    fill: str | None = None        # a second stock to buy with whatever cash is LEFT after the purchase: as many shares as it affords (entry 44)
    fill_price: Decimal = Decimal(0)
    fill_held: Decimal = Decimal(0)  # shares of the fill stock already held (a quantity)
    lean: bool = False             # sell no more shares than the shortfall needs (a hard check): set with `fill`, "sell only as much as the purchase needs"
    max_depth: int = MAX_DEPTH     # steps in a plan (entry 45: a parameter; the prompt says it)
    branching: int = BRANCHING     # thoughts asked for per expanded node (the prompt says it)
    never_liquidate: bool = False  # never sell a position out: at least one share of every holding is kept (entry 47); the shares that may be sold are one fewer than the shares held
    whole_plans: bool = False      # a thought is a WHOLE PLAN, a list of sells, not one sell (entry 50): a purchase is then planned in one to three rounds of the model instead of one round per stock sold
    green_first: bool = False      # score "sell the most profitable (most in the green) first" instead of "realise the least taxable gain" (entry 45)
    explore: bool = False          # make the generator give BOTH kinds of step, COVER and PARTIAL, and ask again when it does not (entry 44); off: the generator is asked once, as in entries 42 and 43
    also: tuple[tuple[str, Decimal, Decimal, Decimal], ...] = ()   # OTHER purchases of the same run (entry 142, R10): (symbol, whole shares, price, shares already held) each. The plan raises the money for the target AND these in ONE plan, and none of these stocks is ever sold

    @property
    def weights(self) -> dict:
        return WEIGHTS_GREEN if self.green_first else WEIGHTS

    def holdings_list(self) -> list[dict]:
        """The holdings that may be sold, as the plain dicts of `trading_math` (sellable shares, price, basis): the same arithmetic the tool does."""
        return [{"symbol": h.symbol, "shares": h.sellable, "price": h.price, "basis": h.basis} for h in self.holdings.values() if self.eligible(h)]

    def fill_with(self, left: Decimal) -> tuple[Decimal, Decimal]:
        """(how many shares of the fill stock `left` buys, the cash that remains). (0, left) when there is no fill stock."""
        if not self.fill or left <= 0:
            return Decimal(0), max(left, Decimal("0.00"))
        n = left // self.fill_price                              # whole shares: the exact quotient, rounded down
        return n, left - trade_amount(n, self.fill_price)

    def eligible(self, holding: "Holding") -> bool:
        """Whether a holding may be sold in this problem: it has sellable shares and, with `only_green`, it shows a profit."""
        return holding.sellable > 0 and (holding.green or not self.only_green)

    @property
    def cost(self) -> Decimal:
        """The cost of everything the run buys: the target and, with `also` (entry 142, R10), the other purchases of the run: each is money from a count and a price, `trade_amount`."""
        return trade_amount(self.shares, self.price) + sum((trade_amount(n, p) for _, n, p, _ in self.also), Decimal("0.00"))

    @property
    def bought(self) -> tuple[str, ...]:
        """The symbols the run buys: the target and the `also` purchases. A stock being bought is never sold to pay for the purchases (the hard check of `_check_one`)."""
        return (self.target, *(s for s, _, _, _ in self.also))

    def fewest_stocks(self) -> int | None:
        """The fewest holdings that could raise the money to raise, the biggest first (arithmetic, nothing is searched): so a plan sells AT LEAST this many stocks. None when even all of them cannot."""
        total = Decimal("0.00")
        for count, value in enumerate(sorted((trade_amount(h.sellable, h.price) for h in self.holdings.values() if self.eligible(h)), reverse=True), 1):
            total += value
            if total >= self.need:
                return count
        return None

    def auto_depth(self, slack: int = 2) -> int:
        """The depth a purchase needs (entry 49): the fewest stocks it needs plus `slack` for plans that spread the sale (the best plan of the complex scenario sold one stock more than the fewest), at least
        the default depth and at most the holdings that can be sold. The default depth of 3 cannot find any plan for a purchase that needs four stocks."""
        eligible = sum(self.eligible(h) for h in self.holdings.values()) or 1
        return max(MAX_DEPTH, min((self.fewest_stocks() or eligible) + slack, eligible))

    @property
    def need(self) -> Decimal:
        """The money to raise: the cost less the free settled cash (`cash_avail`) already there."""
        return max(Decimal("0.00"), self.cost - self.cash)

    @property
    def max_affordable(self) -> Decimal:
        """The most shares of the target that selling every sellable share (and spending the cash) could buy: an upper bound, calculated, for 'how many can I buy'."""
        return (self.cash + sum((trade_amount(h.sellable, h.price) for h in self.holdings.values() if self.eligible(h)), Decimal("0.00"))) // self.price


def ceil_div(amount: Decimal, price: Decimal) -> Decimal:
    """The whole shares that `amount` dollars needs at `price`, rounded UP (a Decimal ceiling of the exact quotient)."""
    return (amount / price).to_integral_value(rounding=ROUND_CEILING)


def _ratio(part: Decimal, whole: Decimal) -> float:
    """`part` as a fraction of `whole`, as a float: the evaluator's SCORES are measurements, never added to money (decimal_money_spec.md section 3, last paragraph), so a ratio of two Decimal amounts becomes a float here, at the very end."""
    return float(part / whole)


def _sellable(sellable_today: Decimal, held: Decimal, never_liquidate: bool) -> Decimal:
    """The WHOLE shares of a holding that a plan may sell: the quantity sellable today (T+1) and, with `never_liquidate`, at most the holding less one share; rounded down, never below 0. A holding of one share or less is
    not sold from (it has nothing to spare), and a fraction of a share is never planned (RaiseFunds sells whole shares only: design choices entry 110)."""
    most = min(sellable_today, sh.difference(held, 1)) if never_liquidate else sellable_today
    return max(Decimal(0), most.to_integral_value(rounding=ROUND_DOWN))


def build_problem(portfolio: Portfolio, target: str, shares: Decimal, prices: dict[str, Decimal], today: date | None = None, only_green: bool = False, fill: str | None = None,
                  explore: bool = False, lean: bool = False, max_depth: int = MAX_DEPTH, branching: int = BRANCHING, green_first: bool = False, never_liquidate: bool = False, whole_plans: bool = False,
                  also: tuple[tuple[str, Decimal], ...] = ()) -> Problem:
    """The portfolio and the quotes -> a Problem. `prices` must hold the target, the `fill` stock and every holding (the Data Clerk's quotes). `only_green`: only holdings that show a profit may
    be sold. `fill`: a second stock bought with the cash left after the purchase, as many shares as it affords; it is never sold, and the sells are then lean (no step sells more shares than
    the shortfall needs). `explore`: the generator must give both kinds of step (see `missing_kind`). `lean` (always on with a fill stock): no step sells more shares than the shortfall needs.
    `max_depth` and `branching`: the size of the tree, which the prompt tells the model. `green_first`: the score prefers selling the most profitable holdings first. `never_liquidate`: at least one share of every holding is kept, so at most
    held - 1 shares of it can be sold (and the shares that are locked by T+1 count among the ones kept). `also` (entry 142, R10) are the other purchases of the same run, ((symbol, whole shares), ...): their cost is part of the money to raise,
    their prices must be in `prices`, and none of them is ever sold. Raises ValueError otherwise."""
    try:
        shares = sh.quantity(shares)                      # a bool and a string are refused by the door (a numeric string such as "2" too: entry 111, M5)
    except ValueError:
        raise ValueError("the number of shares to buy must be a whole number of at least 1") from None
    if not sh.whole(shares) or shares < 1:
        raise ValueError("the number of shares to buy must be a whole number of at least 1")
    if fill is not None and fill == target:
        raise ValueError("the stock to fill with must be another symbol than the stock to buy")
    extra = []
    for symbol, count in also:
        try:
            count = sh.quantity(count)
        except ValueError:
            raise ValueError("the number of shares of every purchase must be a whole number of at least 1") from None
        if not sh.whole(count) or count < 1 or symbol == target or symbol == fill or any(symbol == x for x, _ in extra):
            raise ValueError("every other purchase of the run is a different stock, a whole number of at least 1 share")
        extra.append((symbol, count))
    usable = lambda x: isinstance(x, (int, Decimal)) and not isinstance(x, bool) and x > 0      # a float is not tested here: `exact` below raises TypeError for it (it comes in through a door)
    missing = [s for s in [target, *([fill] if fill else []), *(x for x, _ in extra), *portfolio.symbols] if prices.get(s) is None or not usable(exact(prices[s], "a price"))]
    if missing:
        raise ValueError("no price for " + ", ".join(missing))
    never_sold = {target, fill, *(x for x, _ in extra)}
    holdings = {p.symbol: Holding(p.symbol, p.name, p.shares, _sellable(portfolio.sellable_shares(p.symbol, today), p.shares, never_liquidate), exact(prices[p.symbol], "a price"), p.cost_basis)
                for p in portfolio.positions if p.symbol not in never_sold}
    already, filled = portfolio.position(target), portfolio.position(fill) if fill else None
    return Problem(target, shares, exact(prices[target], "a price"), portfolio.cash_avail, holdings, already.shares if already else Decimal(0), only_green,
                   fill, exact(prices[fill], "a price") if fill else Decimal(0), filled.shares if filled else Decimal(0), lean=bool(fill) or lean, max_depth=max_depth, branching=branching, green_first=green_first, never_liquidate=never_liquidate, whole_plans=whole_plans, explore=explore,
                   also=tuple((x, n, exact(prices[x], "a price"), portfolio.position(x).shares if portfolio.position(x) else Decimal(0)) for x, n in extra))


# ---------------------------------------------------------------- a node: the plan so far

@dataclass
class Node:
    id: str
    parent: str | None
    depth: int
    sells: tuple[tuple[str, Decimal], ...] = ()   # ((symbol, whole shares), ...) from the root to this node
    idea: str = ""                            # the words the LLM gave for the last step
    proceeds: Decimal = Decimal("0.00")
    shortfall: Decimal = Decimal("0.00")
    gain: Decimal = Decimal("0.00")           # realised gain (negative: a loss) if the plan were carried out, to the cent
    score: float = 0.0                        # 0 to 100, calculated: a measurement, a float (see the module text)
    parts: dict = field(default_factory=dict)  # the components that exist for this node (0 to 1 each)
    complete: bool = False
    fill_shares: Decimal = Decimal(0)         # a complete plan: shares of the fill stock the cash left buys
    left: Decimal = Decimal("0.00")           # a complete plan: the cash that remains after the purchase and the fill shares

    def plan(self) -> str:
        return ", ".join(f"{s} x{n}" for s, n in self.sells) or "(nothing sold)"


def root(problem: Problem) -> Node:
    return Node(id="n0", parent=None, depth=0, shortfall=problem.need, score=100.0, parts={})


@traced("raise_funds.score_plan")
def score_plan(problem: Problem, sells: tuple[tuple[str, Decimal], ...]) -> tuple[float, dict]:
    """(score 0 to 100, the components) of a plan, calculated. Every component is 0 to 1, higher is better:

      gain       1 - (the realised gain / the proceeds), the gain clamped to 0..1. A plan that sells at a loss scores 1: the share of what you raise that is taxable gain.        weight 0.40
                 With `green_first` (entry 45) this component is `green` instead, the same weight: where the plan's realised profit stands between the least and the most that selling the same
                 amount could realise, so that selling the most profitable (highest percent gain) holdings first scores 1.
      overshoot  1 - (the cash left after the purchase / the cost), clamped. Only known when the plan is complete: whole shares always leave a little.                        weight 0.25
      keep       1 - (positions sold out completely / holdings). Selling all the shares of a position closes it.                                                              weight 0.20
      orders     1 / the number of sell orders. One order is 1, two 0.5, three 0.33.                                                                                           weight 0.15

    A plan that is not complete yet is scored on the components that exist (overshoot is missing) with the weights renormalised, so the beam can compare partial plans."""
    proceeds = sum((trade_amount(n, problem.holdings[s].price) for s, n in sells), Decimal("0.00"))
    gain = sum((trade_amount(n, problem.holdings[s].price) - trade_amount(n, problem.holdings[s].basis) for s, n in sells), Decimal("0.00"))      # the way the book works a sale's gain: proceeds less the cost basis of the shares, each to the cent
    closed = sum(sh.same(n, problem.holdings[s].shares) for s, n in sells)
    first = ("green", green_component(problem, proceeds, gain)) if problem.green_first else ("gain", 1.0 - min(1.0, max(0.0, _ratio(gain, proceeds))) if proceeds else 1.0)
    parts = {first[0]: first[1],
             "keep": 1.0 - closed / max(1, len(problem.holdings)),
             "orders": 1.0 / len(sells) if sells else 1.0}
    if proceeds + problem.cash >= problem.cost:
        wasted = problem.fill_with(proceeds + problem.cash - problem.cost)[1]  # the cash left that buys nothing: with a fill stock the cash it buys is not wasted
        parts["overshoot"] = 1.0 - min(1.0, max(0.0, _ratio(wasted, problem.cost)))
    weights = problem.weights
    total = sum(weights[k] for k in parts)
    return round(100 * sum(weights[k] * v for k, v in parts.items()) / total, 1), {k: round(v, 3) for k, v in parts.items()}


def green_component(problem: Problem, proceeds: Decimal, gain: Decimal) -> float:
    """0 to 1: how close the plan's realised profit is to the most that selling the same amount could realise by selling the most profitable holdings first (`trading_math.profit_bounds`), between
    that and the least (the least profitable first). 1 is the green-first plan. 1 as well when there is nothing to choose between (all holdings at the same percent gain, or nothing sold)."""
    if proceeds <= 0:
        return 1.0
    best, worst = tm.profit_bounds(problem.holdings_list(), proceeds)
    return 1.0 if best <= worst else min(1.0, max(0.0, _ratio(gain - worst, best - worst)))      # the bounds are worked to four places: no difference is no choice


def _clean(thought) -> tuple[str | None, Decimal | None]:
    """(symbol, shares) of a thought as the checks read them: the symbol stripped, without a leading $ and in capitals; the shares a whole number (a Decimal made from the digits of the int the model wrote). A STRING is not a share count, "2" included, and neither is a bool (design choices
    entry 111, M5: it was accepted as 2 until then); anything else is None. Used by the checks and by the search to see that two thoughts are the same one."""
    if not isinstance(thought, dict):
        return None, None
    symbol, shares = thought.get("symbol"), thought.get("shares")
    symbol = symbol.strip().lstrip("$").upper() if isinstance(symbol, str) else None
    return symbol, Decimal(shares) if isinstance(shares, int) and not isinstance(shares, bool) else None


def check_step(problem: Problem, parent: Node, thought, node_id: str) -> tuple[Node | None, str | None]:
    """The hard checks on one thought. (the child node, None) when it passes, else (None, the reason it is rejected). A thought is one sell ({"symbol", "shares"}) or, with `whole_plans`
    (entry 50), a whole plan ({"sells": [{"symbol", "shares"}, ...]}): its sells are checked one after the other, each against the plan the ones before it make, and the result is ONE node, one
    level below the parent, whatever the number of sells; the first sell that fails rejects the plan and the reason names it. Nothing the thought says is trusted: the price, the proceeds, the
    shortfall and the score are all calculated here from the Problem."""
    steps = thought.get("sells") if isinstance(thought, dict) else None
    if steps is None:
        return _check_one(problem, parent, thought, node_id)
    if not isinstance(steps, list) or not steps or not all(isinstance(s, dict) for s in steps):
        return None, "the plan is not a non-empty list of sells, each with a symbol and shares"
    node = parent
    for i, step in enumerate(steps, 1):
        if node.complete and node is not parent:
            return None, f"sell {i} ({step.get('symbol')}) comes after the plan has already raised the money: the plan keeps selling"
        node, reason = _check_one(problem, node, step, node_id)
        if node is None:
            return None, f"sell {i} of the plan ({_clean(step)[0]} x{_clean(step)[1]}): {reason}"
    node.parent, node.depth = parent.id, parent.depth + 1
    node.idea = " ".join(str(thought.get("idea") or "").split())[:MAX_IDEA]
    return node, None


def _check_one(problem: Problem, parent: Node, thought, node_id: str) -> tuple[Node | None, str | None]:
    """One sell, checked against the plan so far (`parent`): see `check_step`."""
    if not isinstance(thought, dict):
        return None, "the thought is not an object with symbol and shares"
    symbol, shares = _clean(thought)
    if not symbol or shares is None or shares < 1:
        said = thought.get("shares")
        why = " (a number, not text or true/false: write 2, not \"2\")" if isinstance(said, (str, bool)) else ""
        return None, f"{symbol or 'a thought'}: shares must be a whole number of at least 1{why}"
    if symbol in problem.bought:
        return None, f"{symbol} is the stock being bought: it cannot be sold to pay for itself" if symbol == problem.target else f"{symbol} is another stock being bought in this run: it cannot be sold to pay for the purchases"
    holding = problem.holdings.get(symbol)
    if holding is None:
        return None, f"{symbol} is not in the portfolio"
    if any(s == symbol for s, _ in parent.sells):
        return None, f"{symbol} is already in this plan (one sell line per symbol)"
    if problem.only_green and not holding.green:
        return None, f"{symbol} is not in profit (price ${holding.price:,.2f} is not above its basis ${holding.basis:,.2f}): only profitable holdings may be sold"
    if shares > holding.sellable:
        return None, (f"{symbol}: {shares} shares asked but only {holding.sellable} are sellable ({sh.text(holding.shares)} held" + (", and one share of every holding is always kept: a position is never sold out)" if problem.never_liquidate else ")"))
    if problem.lean and parent.shortfall > 0 and shares > ceil_div(parent.shortfall, holding.price):
        enough = ceil_div(parent.shortfall, holding.price)
        return None, f"{symbol}: {shares} shares sells more than is needed: {enough} share(s) already cover the shortfall of ${parent.shortfall:,.2f} (the sells are lean)"
    sells = (*parent.sells, (symbol, shares))
    proceeds = sum((trade_amount(n, problem.holdings[s].price) for s, n in sells), Decimal("0.00"))
    shortfall = max(Decimal("0.00"), problem.need - proceeds)
    if shortfall > 0:
        left = sum((trade_amount(h.sellable, h.price) for s, h in problem.holdings.items() if s not in {x for x, _ in sells} and problem.eligible(h)), Decimal("0.00"))
        if left < shortfall:
            return None, f"dead end: after selling {symbol} x{shares} the shortfall is ${shortfall:,.2f} and the holdings left can raise only ${left:,.2f}"
    score, parts = score_plan(problem, sells)
    gain = sum((trade_amount(n, problem.holdings[s].price) - trade_amount(n, problem.holdings[s].basis) for s, n in sells), Decimal("0.00"))
    idea = " ".join(str(thought.get("idea") or "").split())[:MAX_IDEA]
    fill_shares, left = problem.fill_with(proceeds + problem.cash - problem.cost) if shortfall == 0 else (Decimal(0), Decimal("0.00"))
    return Node(id=node_id, parent=parent.id, depth=parent.depth + 1, sells=sells, idea=idea, proceeds=proceeds, shortfall=shortfall, gain=gain,
                score=score, parts=parts, complete=shortfall == 0, fill_shares=fill_shares, left=left), None


# ---------------------------------------------------------------- which unfinished plans the beam keeps (entry 45)

def progress(problem: Problem, node: Node) -> float:
    """How much of the money to raise the plan has raised so far, 0 to 1."""
    return 1.0 if problem.need <= 0 else min(1.0, max(0.0, 1.0 - _ratio(node.shortfall, problem.need)))


def priority(problem: Problem, node: Node) -> float:
    """What the beam ranks unfinished plans by. Without `explore` it is the score (entries 42 to 44). With `explore` (entry 45) it is half the score and half the progress (0 to 100):
    the score of an unfinished plan is only the components that exist so far, and they reward SELLING LITTLE (few orders, no position closed), so a beam that ranks by it alone keeps the plans
    that have done the least and prunes the ones that are close to finishing. The complex scenario (tests/test_raise_funds_complex.py) showed it: four levels in, every plan that would have
    finished had been pruned. The choice of the final answer never uses this: it compares completed plans by their score alone."""
    return node.score if not problem.explore else 0.5 * node.score + 50.0 * progress(problem, node)


# ---------------------------------------------------------------- the kinds of step the generator must give (entry 44)

FEEDBACK = "FEEDBACK: "  # a rejection-list entry that starts with this is the program's note on the model's last answer, not a rejected step


def kinds_of(problem: Problem, node: Node, thoughts: list) -> tuple[list, list]:
    """(the thoughts that would FINISH the plan (a COVER step), the thoughts that would not (a PARTIAL step)), among those that pass the hard checks. Nothing is created: it is a look."""
    cover, partial = [], []
    for thought in thoughts:
        child, _ = check_step(problem, node, thought, "probe")
        if child is not None:
            (cover if child.complete else partial).append(thought)
    return cover, partial


def wanted_kinds(problem: Problem, node: Node, max_depth: int) -> tuple[list[Holding], bool]:
    """(the holdings that could finish the plan alone, whether a PARTIAL step makes sense here). Both are facts about the holdings, so code decides them, not the model: a COVER step is
    wanted when some unused, sellable holding is worth enough on its own, a PARTIAL one when two or more unused holdings can be sold and the plan has room for another step."""
    used = {s for s, _ in node.sells}
    free = [h for h in problem.holdings.values() if h.symbol not in used and problem.eligible(h)]
    can_cover = [h for h in free if trade_amount(h.sellable, h.price) >= node.shortfall]
    return can_cover, len(free) >= 2 and node.depth + 1 < max_depth and not problem.whole_plans   # with whole plans no partial step is wanted: each thought is meant to finish the plan


def missing_kind(problem: Problem, node: Node, thoughts: list, max_depth: int) -> str:
    """Why the generator's answer is missing a kind of step it should have given ("" when it is not): written for the model, with the numbers calculated by code."""
    can_cover, partial_makes_sense = wanted_kinds(problem, node, max_depth)
    cover, partial = kinds_of(problem, node, thoughts)
    if can_cover and not cover:
        names = ", ".join(f"{h.symbol} ({ceil_div(node.shortfall, h.price)} shares)" for h in can_cover)
        return f"none of your steps finishes the plan, and {names} could finish it alone: give a COVER step with exactly that many shares"
    if partial_makes_sense and not partial:
        return "all your steps finish the plan in one step: give a PARTIAL step as well (some or all of a stock, not enough to finish) so that stocks can be combined"
    return ""


@traced("raise_funds.choose")
def choose(problem: Problem, node: Node, thoughts: list, branching: int, max_depth: int) -> list:
    """From the first answer and the second one, at most `branching` distinct thoughts, keeping one of each wanted kind first (the first COVER, the first PARTIAL) and then the rest in order."""
    unique, seen = [], set()
    for thought in thoughts:
        key = _clean(thought)
        if key in seen and key[0] is not None:
            continue
        seen.add(key)
        unique.append(thought)
    cover, partial = kinds_of(problem, node, unique)
    can_cover, partial_makes_sense = wanted_kinds(problem, node, max_depth)
    first = [cover[0]] if can_cover and cover else []
    first += [partial[0]] if partial_makes_sense and partial else []
    return [*first, *[t for t in unique if not any(t is f for f in first)]][:branching]


# ---------------------------------------------------------------- the LangGraph

class TreeState(TypedDict, total=False):
    frontier: list            # the unfinished nodes to expand next (the beam)
    depth: int                # steps taken so far
    complete: list            # every completed plan found
    raw: list                 # (parent node, thoughts) pairs the last expand step produced
    rejected: list            # every rejection reason, oldest first (the next call is told the latest)
    trace: Annotated[list, lambda old, new: old + new]  # one record per level
    calls: int                # LLM calls used
    nodes: int                # nodes created (including the root)
    stop: str                 # why the search ended
    errors: list              # generator failures
    reasked: list             # the nodes the generator was asked about a second time in the last expand step, and why (entry 44)
    states: dict              # every set of sells reached so far (a frozenset of (symbol, shares)) and the first node that reached it (entry 45): the same sells in another order is not a new plan


Generate = Callable[[Problem, Node, list], list]  # (problem, the node to expand, recent rejection reasons) -> up to BRANCHING thoughts; may raise


def effective_limits(problem: Problem, beam_width: int | None = None, branching: int | None = None, max_depth: int | None = None, max_calls: int | None = None) -> dict:
    """The size of the tree for this problem: what was asked (or the problem's own `max_depth` and `branching`), the depth never more than the holdings that can be sold."""
    eligible = sum(problem.eligible(h) for h in problem.holdings.values()) or 1
    return {"beam_width": beam_width or BEAM_WIDTH, "branching": branching or problem.branching, "max_depth": min(max_depth or problem.max_depth, eligible), "max_calls": max_calls or MAX_LLM_CALLS}


def build_graph(problem: Problem, generate: Generate, beam_width: int | None = None, branching: int | None = None, max_depth: int | None = None, max_calls: int | None = None,
                parallel: bool = False, deadline: float | None = None):
    """The tree search as a LangGraph. `generate` is the thought generator (the LLM in the Account Manager; a scripted function in tests). Compile once per problem. The limits default to
    the problem's own (`max_depth`, `branching`) and the module's (`BEAM_WIDTH`, `MAX_LLM_CALLS`).

    `parallel` (entry 46): the plans of one level are independent of each other (each is expanded from the same snapshot of what was rejected), so their calls to the generator are made at the same time,
    one thread per plan, and the answers are put back in the order of the frontier: the trace and the result are the same as with `parallel=False`, only the waiting is shared. The model's latency is
    almost all of the time of a search (the checks and the scores take milliseconds), and a level used to wait for its plans one after the other.

    `deadline` (entry 48): seconds from now. A search a person is waiting for has to answer in time: once the deadline has passed no new level is started and no new call is made, and the best plan
    COMPLETED so far is the answer ("the time limit was reached"); with none completed yet, the answer is that no plan was found, and nothing is sold. A level whose calls are already out is waited for
    (a call cannot be taken back), so a search ends at most one level's latency after the deadline. The seconds at the end of every level are in the trace."""
    t0 = time.perf_counter()
    late = lambda: deadline is not None and time.perf_counter() - t0 >= deadline
    limits = effective_limits(problem, beam_width, branching, max_depth, max_calls)
    beam_width, branching, max_depth, max_calls = limits["beam_width"], limits["branching"], limits["max_depth"], limits["max_calls"]

    def expand(state: TreeState) -> dict:
        used, lock = [state["calls"]], threading.Lock()
        recent = state["rejected"][-MAX_REJECTIONS_TOLD:]

        def take() -> bool:  # one call of the budget, or none left (the budget is shared by the plans of the level, whether they run one after the other or together)
            with lock:
                if used[0] >= max_calls or late():
                    return False
                used[0] += 1
                return True

        def one(node: Node):
            """The expansion of one plan: (node, thoughts, why it was asked again or "", the error kind or ""), or None when the budget is used."""
            if not take():
                return None
            why = ""
            try:
                thoughts = generate(problem, node, recent)
                thoughts = thoughts if isinstance(thoughts, list) else []
                missing = missing_kind(problem, node, thoughts, max_depth) if problem.explore else ""
                if missing and take():  # the generator gave only one KIND of step: ask once more, saying which kind is missing (entry 44)
                    why = missing
                    more = generate(problem, node, [*recent, FEEDBACK + missing])
                    thoughts = choose(problem, node, [*thoughts, *(more if isinstance(more, list) else [])], branching, max_depth)
            except Exception as error:  # a model or network failure: record the kind (never request details) and go on with the other nodes
                return node, [], why, type(error).__name__
            return node, thoughts[:branching], why, ""

        frontier = state["frontier"]
        if parallel and len(frontier) > 1:
            with ThreadPoolExecutor(max_workers=len(frontier)) as pool:
                contexts = [contextvars.copy_context() for _ in frontier]   # one copy of the caller's run per node, so each model call is traced under it (a Context is entered by one thread at a time)
                results = list(pool.map(lambda context, node: context.run(one, node), contexts, frontier))   # in the order of the frontier, whichever finishes first
        else:
            results = [one(node) for node in frontier]
        raw, errors, reasked = [], list(state["errors"]), []
        for result in results:
            if result is None:
                continue
            node, thoughts, why, error = result
            if why:
                reasked.append({"node": node.id, "why": why})
            if error:
                errors.append(f"expanding {node.id} failed ({error})")
                continue
            raw.append((node, thoughts))
        return {"raw": raw, "calls": used[0], "errors": errors, "reasked": reasked}

    def evaluate_and_prune(state: TreeState) -> dict:
        """Check every thought (hard checks), score the survivors, then apply the beam. One node, one record in the trace, for each thought."""
        depth, nodes, rejected, states = state["depth"] + 1, state["nodes"], list(state["rejected"]), dict(state["states"])
        children, records, seen = [], [], set()
        for parent, thoughts in state["raw"]:
            for thought in thoughts:
                key = (parent.id, *_clean(thought))
                if key in seen and key[1] is not None:
                    records.append({"parent": parent.id, "thought": thought, "status": "duplicate", "reason": "the same thought twice from the same node"})
                    continue
                seen.add(key)
                child, reason = check_step(problem, parent, thought, f"n{nodes}")
                if child is None:
                    rejected.append(reason)
                    records.append({"parent": parent.id, "thought": thought, "status": "rejected", "reason": reason})
                    continue
                same = frozenset(child.sells)
                if same in states:  # the same sells in another order (MCD then GM, or GM then MCD) is the same plan: it would only take a place in the beam from a different one (entry 45)
                    records.append({"parent": parent.id, "thought": thought, "status": "duplicate", "reason": f"the same sells as {states[same]} in a different order"})
                    continue
                states[same] = child.id
                nodes += 1
                children.append(child)
                records.append({"parent": parent.id, "node": child.id, "thought": thought, "plan": child.plan(), "score": child.score, "parts": child.parts,
                                "proceeds": child.proceeds, "shortfall": child.shortfall, "status": "complete" if child.complete else "open"})
        complete = [*state["complete"], *[c for c in children if c.complete]]
        open_nodes = sorted((c for c in children if not c.complete), key=lambda c: (-priority(problem, c), len(c.sells), c.id))
        frontier, pruned = open_nodes[:beam_width], open_nodes[beam_width:]
        kept = list(frontier)
        retry = problem.whole_plans and not frontier and not complete and depth < max_depth and state["calls"] < max_calls and not late()   # every plan was rejected: ask again from the start, with the reasons
        if retry:
            frontier = [root(problem)]
        by_id = {r.get("node"): r for r in records}
        for node in kept:
            by_id[node.id]["status"] = "kept"
        for node in pruned:
            by_id[node.id]["status"] = "pruned"
            by_id[node.id]["reason"] = (f"below the beam: score {node.score} is not in the top {beam_width} unfinished plans (cut-off {frontier[-1].score})" if not problem.explore else
                                        f"below the beam: priority {priority(problem, node):.1f} (score {node.score}, {progress(problem, node):.0%} of the money raised) is not in the top {beam_width} "
                                        f"unfinished plans (cut-off {priority(problem, frontier[-1]):.1f})")
        level = {"level": depth, "expanded": [{"node": p.id, "plan": p.plan(), "score": p.score} for p, _ in state["raw"]], "children": records, "reasked": state.get("reasked", []),
                 "seconds": round(time.perf_counter() - t0, 2), "retry": retry}
        return {"frontier": frontier, "depth": depth, "complete": complete, "rejected": rejected, "nodes": nodes, "states": states, "trace": [level]}

    def route(state: TreeState) -> str:
        if not state["frontier"]:
            return "decide"
        if state["depth"] >= max_depth:
            return "decide"
        if state["calls"] >= max_calls or late():
            return "decide"
        return "expand"

    def decide(state: TreeState) -> dict:
        if not state["frontier"]:
            reason = "no unfinished plan is left"
        elif state["depth"] >= max_depth:
            reason = f"the depth limit ({max_depth}) was reached"
        elif late():
            reason = f"the time limit ({deadline:g} s) was reached"
        else:
            reason = f"the call budget ({max_calls}) was used"
        if state["complete"]:
            reason += "; the best completed plan is the answer"
        return {"stop": reason}

    graph = StateGraph(TreeState)
    graph.add_node("expand", expand)
    graph.add_node("evaluate_and_prune", evaluate_and_prune)
    graph.add_node("decide", decide)
    graph.add_edge(START, "expand")
    graph.add_edge("expand", "evaluate_and_prune")
    graph.add_conditional_edges("evaluate_and_prune", route, {"expand": "expand", "decide": "decide"})
    graph.add_edge("decide", END)
    return graph.compile()


@traced("raise_funds.search")
def search(problem: Problem, generate: Generate, **limits) -> dict:
    """Run the tree search and return the answer (`answer`): status, the chosen plan, the alternatives, the whole trace and the numbers a reader needs.
    status is `funded` (the cash already covers the purchase: nothing to sell, no search), `plan` (a completed plan was found) or `infeasible` (none was)."""
    parallel, deadline = limits.pop("parallel", False), limits.pop("deadline", None)
    started = time.perf_counter()
    if problem.need == 0:
        return answer(problem, [], {"trace": [], "calls": 0, "nodes": 1, "stop": "the free settled cash (cash_avail) already covers the purchase", "errors": []}, effective_limits(problem, **limits))
    sizes = effective_limits(problem, **limits)
    final = build_graph(problem, generate, parallel=parallel, deadline=deadline, **limits).invoke(
        {"frontier": [root(problem)], "depth": 0, "complete": [], "raw": [], "rejected": [], "trace": [], "calls": 0, "nodes": 1, "stop": "", "errors": [], "reasked": [], "states": {frozenset(): "n0"}},
        config={"recursion_limit": 20 * sizes["max_depth"] + 40})
    return answer(problem, final["complete"], final, {**sizes, "deadline": deadline}, round(time.perf_counter() - started, 2))


@traced("raise_funds.answer")
def answer(problem: Problem, complete: list[Node], final: dict, limits: dict | None = None, seconds: float = 0.0) -> dict:
    ranked = sorted(complete, key=lambda n: (-n.score, len(n.sells), n.id))
    best = ranked[0] if ranked else None
    status = "plan" if best else "funded" if problem.need == 0 else "infeasible"
    return {"status": status, "target": problem.target, "shares": problem.shares, "price": problem.price, "cost": problem.cost, "cash": problem.cash, "need": problem.need,
            "also": [{"symbol": x, "shares": n, "price": p} for x, n, p, _ in problem.also],  # the other purchases of the run, paid for by the same plan (entry 142, R10)
            "only_green": problem.only_green, "never_liquidate": problem.never_liquidate, "whole_plans": problem.whole_plans, "fewest_stocks": problem.fewest_stocks(),
            "fill": {"symbol": problem.fill, "price": problem.fill_price, "held": problem.fill_held} if problem.fill else None, "lean": problem.lean,
            "max_affordable": problem.max_affordable,
            "plan": describe(problem, best) if best else None,
            "alternatives": [describe(problem, n) for n in ranked[1:1 + KEPT_ALTERNATIVES]],
            "trace": final["trace"], "stop": final["stop"], "errors": final["errors"],
            "weights": dict(problem.weights), "green_first": problem.green_first, "limits": limits or effective_limits(problem), "stats": {"llm_calls": final["calls"], "nodes": final["nodes"], "levels": len(final["trace"]), "completed_plans": len(ranked), "seconds": seconds,
                                                                                                                          "first_plan_seconds": next((lv.get("seconds") for lv in final["trace"] if any(c["status"] == "complete" for c in lv["children"])), None)},
            "before": {"shares": {**{s: h.shares for s, h in problem.holdings.items()}, **({problem.target: problem.held} if problem.held else {}), **({problem.fill: problem.fill_held} if problem.fill and problem.fill_held else {}), **{x: held for x, _, _, held in problem.also if held}}, "cash": problem.cash},
            "after": after(problem, best)}


def describe(problem: Problem, node: Node) -> dict:
    lines = []
    for symbol, n in node.sells:
        h = problem.holdings[symbol]
        lines.append({"symbol": symbol, "shares": n, "price": h.price, "proceeds": trade_amount(n, h.price), "basis": h.basis, "gain": trade_amount(n, h.price) - trade_amount(n, h.basis),
                      "closes_position": sh.same(n, h.shares)})
    return {"node": node.id, "sells": lines, "proceeds": node.proceeds, "left_over": node.left if problem.fill else node.proceeds + problem.cash - problem.cost, "gain": node.gain,
            "fill_shares": node.fill_shares, "fill_cost": trade_amount(node.fill_shares, problem.fill_price), "score": node.score, "parts": node.parts, "idea": node.idea}


def after(problem: Problem, node: Node | None) -> dict:
    """The holdings after the plan, and the purchase: {symbol: shares}, cash included. With no plan: {} (infeasible), or, when the free cash alone pays for the purchase, the holdings
    with the target added and the cost taken out of the cash."""
    if node is None and problem.need > 0:
        return {}
    shares = {s: h.shares for s, h in problem.holdings.items()}
    for symbol, n in (node.sells if node else ()):
        shares[symbol] = sh.minus(shares[symbol], n)
    shares[problem.target] = sh.plus(problem.held, problem.shares)
    for symbol, n, _, held in problem.also:      # the other purchases of the run (entry 142, R10): their shares are added to what is held
        shares[symbol] = sh.plus(held, n)
    cash = (node.proceeds if node else Decimal("0.00")) + problem.cash - problem.cost
    if problem.fill:
        bought, cash = problem.fill_with(cash)
        shares[problem.fill] = sh.plus(problem.fill_held, bought)
    return {"shares": {s: n for s, n in shares.items() if n > 0}, "cash": cash}


# ---------------------------------------------------------------- what the user reads

def _money(value: Decimal) -> str:
    return f"${value:,.2f}" if value >= 0 else f"-${-value:,.2f}"


def _table(headers: list[str], rows: list[list[str]], right: set[int] = frozenset()) -> list[str]:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    cell = lambda i, text: text.rjust(widths[i]) if i in right else text.ljust(widths[i])
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [line, "| " + " | ".join(cell(i, h) for i, h in enumerate(headers)) + " |", line]
    out += ["| " + " | ".join(cell(i, c) for i, c in enumerate(r)) + " |" for r in rows]
    return out + [line]


def format_trace(result: dict) -> str:
    """The reasoning trace as text: for each level, every thought with the program's verdict."""
    lines = ["REASONING TRACE (the tree: each level expands the kept plans; every number below was calculated by code; the score of an unfinished plan counts only the components "
             "that exist so far, so it is optimistic and only ranks plans against each other; the LLM's ideas are its own words and are not checked)"]
    if not result["trace"]:
        lines.append("  (no search: " + result["stop"] + ")")
    for level in result["trace"]:
        expanded = ", ".join(f"{e['node']} [{e['plan']}]" for e in level["expanded"]) or "nothing"
        lines.append(f"\nLEVEL {level['level']}  expanded: {expanded}" + (f"   [{level['seconds']:.1f} s]" if level.get("seconds") is not None else ""))
        rows = []
        for c in level["children"]:
            t = c["thought"] if isinstance(c["thought"], dict) else {}
            step = ("plan: " + ", ".join(f"{x.get('symbol')} x{x.get('shares')}" for x in t["sells"] if isinstance(x, dict)))[:44] if isinstance(t.get("sells"), list) else f"sell {t.get('shares')} {t.get('symbol')}" if t else "?"
            status = {"complete": "COMPLETE", "kept": "kept", "open": "open", "pruned": "PRUNED", "rejected": "REJECTED", "duplicate": "duplicate"}[c["status"]]
            detail = (f"{c['plan']}; raises {_money(c['proceeds'])}, short {_money(c['shortfall'])}" if "plan" in c else c["reason"])
            rows.append([c.get("node", "-"), c["parent"], step, str(c.get("score", "-")), status, detail])
        lines += ["  " + line for line in _table(["node", "from", "thought", "score", "verdict", "plan / reason"], rows, right={3})]
        if level.get("retry"):
            lines.append("    every plan was rejected: the first plan is asked for again, with the reasons")
        for r in level.get("reasked", []):
            lines.append(f"    {r['node']}: asked again: {r['why']}")
        for c in level["children"]:
            t = c["thought"] if isinstance(c["thought"], dict) else {}
            if t.get("idea") and c["status"] != "duplicate":
                lines.append(f"    {c.get('node', 'x')}: LLM idea: {str(t['idea'])[:MAX_IDEA]}")
            if c["status"] == "pruned":
                lines.append(f"    {c['node']}: pruned: {c['reason']}")
    lines.append(f"\nSTOP     {result['stop']}")
    for e in result["errors"]:
        lines.append(f"ERROR    {e}")
    return "\n".join(lines)


def format_answer(result: dict, quotes: dict | None = None) -> str:
    """The final answer as text: the quotes, the trace, the order table, what was sold and bought, the score, and the portfolio before and after."""
    out = []
    t, s = result["target"], result["stats"]
    others = result.get("also") or []
    out.append(f"GOAL     buy {result['shares']} x {t} at {_money(result['price'])}" + "".join(f", {o['shares']} x {o['symbol']} at {_money(o['price'])}" for o in others) + f" = {_money(result['cost'])} (free settled cash {_money(result['cash'])}, to raise {_money(result['need'])}); "
               f"at most {result['max_affordable']} share(s) of {t} are affordable by selling everything sellable"
               + ("; only holdings in profit (green) may be sold, and never {t} itself".format(t=", ".join([t, *(o["symbol"] for o in others)])) if result.get("only_green") else "; {t} itself is never sold".format(t=", ".join([t, *(o["symbol"] for o in others)]))))
    if result.get("requested") is None and "requested" in result:
        out.append(f"         QUANTITY: no number of shares was given, so as many as the portfolio can pay for: at most {result['most_affordable']} x {t} (cash plus everything that may be sold), "
                   f"{result['resolved_shares']} bought")
    elif result.get("up_to") and result.get("requested") is not None:
        out.append(f"         QUANTITY: up to {result['requested']} x {t}; the portfolio can pay for at most {result['most_affordable']}, so {result['resolved_shares']} bought")
    if result.get("auto_depth"):
        out.append(f"         DEPTH: not asked for, so chosen from the purchase: it needs at least {result['fewest_stocks']} stock(s) sold, so up to {result['limits']['max_depth']} steps")
    fill = result.get("fill")
    if fill:
        out.append(f"         then the cash that is LEFT buys as many shares of {fill['symbol']} (at {_money(fill['price'])}) as it can; the sells are lean (no step sells more shares than the shortfall needs); "
                   f"if selling everything cannot pay for {result['shares']} x {t}, nothing is sold")
    if quotes:
        rows = [[sym, _money(q["price"]), f"{q['change_percent']:+.2f}%", q["as_of_utc"], q["source"]] for sym, q in quotes.items()]
        out += ["", "QUOTES (from the Data Clerk's get_quote tool)", *_table(["symbol", "price", "day change", "as of (UTC)", "source"], rows, right={1, 2})]
    out += ["", format_trace(result), ""]
    plan = result["plan"]
    if plan is None and result["status"] == "funded":
        out.append("RESULT   funded: the free settled cash (cash_avail) already covers the purchase, nothing is sold."
                   + (f" The cash left buys {result['after']['shares'].get(fill['symbol'], 0) - fill['held']} x {fill['symbol']}, ${result['after']['cash']:,.2f} remains." if fill else ""))
    elif plan is None:
        out.append(f"RESULT   INFEASIBLE: no plan found that raises {_money(result['need'])} (at most {result['max_affordable']} share(s) of {t} can be bought by selling everything sellable). NOTHING IS SOLD.")
    else:
        rows = [["SELL", x["symbol"], str(x["shares"]), _money(x["price"]), _money(x["proceeds"]), f"basis {_money(x['basis'])}, gain {_money(x['gain'])}"
                 + (", closes the position" if x["closes_position"] else "")] for x in plan["sells"]]
        rows.append(["BUY", t, str(result["shares"]), _money(result["price"]), _money(-result["cost"]), "the goal"])
        if fill and plan["fill_shares"]:
            rows.append(["BUY", fill["symbol"], str(plan["fill_shares"]), _money(fill["price"]), _money(-plan["fill_cost"]), "with the cash left"])
        out += ["FINAL ANSWER (the plan with the highest calculated score)", *_table(["order", "symbol", "shares", "price", "cash", "detail"], rows, right={2, 3, 4})]
        out.append(f"  raised {_money(plan['proceeds'])} + free cash {_money(result['cash'])}, spent {_money(result['cost'])}"
                   + (f" + {_money(plan['fill_cost'])} on {plan['fill_shares']} x {fill['symbol']} (the cash left, {_money(plan['proceeds'] + result['cash'] - result['cost'])}, buys {plan['fill_shares']} at {_money(fill['price'])}"
                      f"{'' if plan['fill_shares'] else ': less than one share'})" if fill else "")
                   + f", left over {_money(plan['left_over'])}; realised gain {_money(plan['gain'])}")
        ok = success(result)
        out.append(f"  CHECK    every sale in profit: {'yes' if ok['profitable'] else 'NO'} | no position sold out: {'yes' if ok['kept'] else 'NO'} | shares bought: "
                   f"{ok['bought']}" + ((f" of {ok['asked']} asked" + (f" (up to: the most the portfolio could pay for is {ok['wanted']})" if ok["wanted"] != ok["asked"] else "")) if ok["asked"] is not None else " (as many as possible)") + f" -> {'SUCCESS' if ok['ok'] else 'NOT MET'}")
        out.append(f"  score {plan['score']} / 100   components (0 to 1): " + ", ".join(f"{k} {v}" for k, v in plan["parts"].items()) + "   weights: " + ", ".join(f"{k} {w}" for k, w in (result.get("weights") or WEIGHTS).items()))
        if plan["idea"]:
            out.append(f"  idea of the last step (the LLM's words): {plan['idea']}")
        if result["alternatives"]:
            alt = [[a["node"], ", ".join(f"{x['symbol']} x{x['shares']}" for x in a["sells"]), _money(a["proceeds"]), _money(a["left_over"]), _money(a["gain"]), str(a["score"])]
                   + ([str(a["fill_shares"])] if fill else []) for a in result["alternatives"]]
            out += ["", "OTHER COMPLETED PLANS (lower score)", *_table(["node", "sells", "raises", "left over", "gain", "score"] + ([f"{fill['symbol']} bought"] if fill else []), alt,
                                                                           right={2, 3, 4, 5, 6})]
        if result["after"]:
            names = [*result["before"]["shares"], *([t] if t not in result["before"]["shares"] else []), *([fill["symbol"]] if fill and fill["symbol"] not in result["before"]["shares"] else [])]
            names = list(dict.fromkeys(names))
            rows = [[sym, str(result["before"]["shares"].get(sym, 0)), str(result["after"]["shares"].get(sym, 0))] for sym in names]
            rows.append(["cash", _money(result["before"]["cash"]), _money(result["after"]["cash"])])
            out += ["", "PORTFOLIO (shares)", *_table(["symbol", "before", "after"], rows, right={1, 2})]
    lim = result.get("limits") or effective_limits_default()
    out.append(f"\nSEARCH   {s['llm_calls']} LLM calls, {s['nodes']} nodes, {s['levels']} levels, {s['completed_plans']} completed plan(s); {s['seconds']:.1f} s" + (f" (first completed plan at {s['first_plan_seconds']:.1f} s)" if s.get("first_plan_seconds") is not None else "") + f"; budget {lim['max_calls']} calls, beam {lim['beam_width']}, branching {lim['branching']}, depth {lim['max_depth']}")
    return "\n".join(out)


def effective_limits_default() -> dict:
    return {"beam_width": BEAM_WIDTH, "branching": BRANCHING, "max_depth": MAX_DEPTH, "max_calls": MAX_LLM_CALLS}


def success(result: dict) -> dict:
    """Jacob's success measure (2026-11-03), calculated from a result: the plan found profitable stocks to sell (every sale shows a gain), no position was sold out (one share of every holding is
    kept), and the requested shares were bought. {"profitable", "kept", "asked", "wanted", "bought", "all_bought", "ok"}. `asked` is what was asked (None for "as many as possible"); `wanted` is what
    counts as bought in full: the shares asked, or, for "up to X" and "as many as possible", the most the portfolio could pay for (`resolved_shares`). `ok` is False when there is no plan. A result of
    `funded` (the cash paid for it) needs no sale and passes the first two."""
    sells = (result.get("plan") or {}).get("sells", [])
    profitable, kept = all(x["gain"] > 0 for x in sells), all(not x["closes_position"] for x in sells)
    done = result["status"] in ("plan", "funded")
    asked, bought = result.get("requested"), result["shares"] if done else 0
    wanted = result.get("resolved_shares", asked)
    all_bought = done and (wanted is None or bought >= wanted)
    return {"profitable": profitable, "kept": kept, "asked": asked, "wanted": wanted, "bought": bought, "all_bought": all_bought, "ok": bool(done and profitable and kept and all_bought)}


def _json_default(o):
    """A dataclass as its dict, a Decimal as a JSON NUMBER (the one encoder: `money.json_default`), anything else as its text."""
    return asdict(o) if hasattr(o, "__dataclass_fields__") else json_default(o)


def as_json(result: dict) -> str:
    return json.dumps(result, indent=1, default=_json_default)
