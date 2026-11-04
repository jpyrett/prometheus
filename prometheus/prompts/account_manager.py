"""The Account Manager's prompt for RaiseFunds, the Tree of Thought (design choices entry 42).

The model is only the THOUGHT GENERATOR. It never writes a price, a total or a score: the numbers in the message come from the Data Clerk's quotes and the portfolio file, and a program
checks every thought and calculates every score (prometheus/reasoning/raise_funds.py). The model is told how plans are judged, because a generator that knows the criteria proposes
better steps, and it is told what the program rejected, so that its next call does not repeat a mistake.

Entry 44 changed how the model is asked for thoughts, after the first live runs (entries 42 and 43): every time the model was shown the problem it proposed three plans that each finished
the job alone ("sell 5 TSLA", "sell 4 MSFT", "sell 19 GM"), so the tree never went below level 1 and a tree search was a list of three. Now the prompt
  * explains the tree (the model proposes BRANCHES; the search asks again for the next step of the best unfinished ones);
  * names two kinds of step, COVER (one stock raises all that is still needed) and PARTIAL (some or all of a stock, not enough: another stock finishes it at the next level), and asks for at
    least one PARTIAL step whenever two or more stocks can still be sold (the condition is computed by code and the instruction is only added then);
  * says, per holding, whether it can cover the shortfall alone, and when it cannot, what ALL of it is worth (so that it is offered as a partial step instead of an impossible one);
  * says, per holding, what share of the money a sale raises would be taxable gain (the first component of the score, calculated by code), so that the model can choose the holdings the score
    prefers: in the second live run the model never proposed GOOGL, which a calculation shows scores highest of the single sales (the search can only compare what the generator proposes);
  * with a stock to fill, says that the cash left buys it automatically and that a step selling more shares than the shortfall needs is thrown away (lean).
Nothing about the checks, the score or the pruning changed: the generator only proposes.
"""
import math

SYSTEM = """You are the Account Manager of a trading desk. The user wants to buy a stock but holds no cash, so something must be sold first. You build the plan one step at a time: every step is "sell n shares of one stock". You are given the holdings with their current prices, the plan built so far and the money still to raise.

Rules
- Reply with up to {branching} DIFFERENT next steps, each a different idea (for example: the sell that saves the most tax, the sell that needs the fewest orders, the sell that keeps every position open). Do not give the same step twice.
{explore}- A step is {{"symbol": "TSLA", "shares": 3, "idea": "one short sentence on why"}}. Shares is a whole number of at least 1, no more than the sellable shares shown.
- Never sell the stock being bought: the money must come from OTHER symbols. {green}Never repeat a symbol that is already in the plan so far. Use only the symbols and prices given.
- A program checks every step and calculates the proceeds and the score from the prices given; do not do the totals yourself and never invent a price. A step that breaks a rule is thrown away and you are told why.
- Plans are scored (higher is better) on: {first_score}; the cash left over after the purchase (less is better); how many positions are sold out completely (fewer is better); the number of sell orders (fewer is better).
- A plan is finished when it raises the money needed. A step may raise less than needed: the plan then continues with another step.
{lean}
Reply with JSON only, no other text, in exactly this shape:
{{"thoughts": [{{"symbol": "TSLA", "shares": 3, "idea": "TSLA is below its cost basis, so selling it realises a loss"}}]}}"""

SYSTEM_PLANS = """You are the Account Manager of a trading desk. The user wants to buy a stock and must sell other holdings to pay for it. You propose WHOLE PLANS: each plan is a list of sells that together raise the money still needed. You are given the holdings with their prices, the plan so far (empty at the start) and the money still to raise.

Rules
- Reply with up to {branching} DIFFERENT plans. Each plan is {{"sells": [{{"symbol": "KO", "shares": 20}}, {{"symbol": "WMT", "shares": 5}}], "idea": "one short sentence"}}. Shares is a whole number of at least 1, no more than the sellable shares shown. One sell per stock in a plan.
- A plan must raise at least the money still to raise. Add up the sells yourself: shares x price for each sell. List the biggest sell first.
- {green}Never sell the stock being bought: the money must come from OTHER symbols.
- {lean}The LAST sell of a plan must not sell more shares than the money still needed requires: the program throws away a plan that sells a whole share too many.
- Plans are scored (higher is better) on: {first_score}; the cash left over after the purchase (less is better); how many positions are sold out completely (fewer is better); the number of sell orders (fewer is better).
- If a plan so far is given, add sells of OTHER stocks to finish it; do not repeat a stock that is already in it.
- A program checks every plan and calculates the proceeds and the score: a plan that breaks a rule is thrown away and you are told why. Use only the symbols and prices given. Never invent a price.

Reply with JSON only, no other text, in exactly this shape:
{{"plans": [{{"sells": [{{"symbol": "KO", "shares": 20}}, {{"symbol": "WMT", "shares": 5}}], "idea": "the most profitable holdings first"}}]}}"""

RETRY = ("Your reply was not valid JSON in the required shape ({problem}). Reply again with JSON only, in exactly the shape described, "
         "with no other text.")


EXPLORE = ("- You do not choose the final plan. You propose BRANCHES of a search: the program keeps the best unfinished plans and asks you again for the next step of each, up to {max_depth} steps in all (one per stock). "
           "Different branches are what the search needs.\n"
           "- There are two kinds of step. COVER: sell the fewest shares of one stock that alone raise all the money still needed (only possible for a stock whose sellable shares are worth enough). "
           "PARTIAL: sell some or all of a stock but NOT enough to finish, so that another stock completes the plan at the next level. A partial step is how several stocks are combined, "
           "for example selling all of a cheap, low-gain stock first. Give BOTH kinds: at least one COVER step if any stock can cover alone, and at least one PARTIAL step when two or more "
           "stocks can still be sold. The program checks which kind each step is and asks again if one kind is missing.\n")
SCORE_TAX = "the share of the money raised that is taxable gain (selling at a loss is best)"
SCORE_GREEN = ("how green-first the plan is: sell the MOST PROFITABLE holdings first, those with the highest percent gain (the order is listed with the holdings); a plan that sells a less profitable "
               "holding while a more profitable one is still unsold scores lower")
GREEN = "Only holdings that show a profit (price above cost basis) may be sold; one at a loss is thrown away. "
KEEP = "Never sell a position out: at least one share of every holding is always kept, and the sellable shares shown are already one fewer than the shares held. "


LEAN = ("- Sell no more shares than the shortfall needs. A COVER step sells EXACTLY the number of shares shown for that stock (\"N share(s) alone would raise the shortfall\"): fewer does not finish the plan "
        "and more is thrown away. {fill}\n")
FILL = "After the purchase, the cash that is left buys as many shares of {fill} as it can: that is done automatically, do not propose it and do not sell {fill}."


def system_prompt(branching: int, only_green: bool = False, fill: str | None = None, lean: bool = False, explore: bool = False, max_depth: int = 3, green_first: bool = False, never_liquidate: bool = False, whole_plans: bool = False) -> str:
    """The system prompt. `only_green`, `fill` (a stock bought with the cash left), `lean` (no step sells more than the shortfall needs) and `explore` (give both kinds of step) add their rules
    only when they apply: without them the prompt is the one of entries 42 and 43."""
    if whole_plans:   # whole plans (entry 50): the same rules, asked for as a list of sells
        rules = (GREEN if only_green else "") + (KEEP if never_liquidate else "")
        return SYSTEM_PLANS.format(branching=branching, green=rules, lean="" if lean else "(No lean rule is on: the program does not check this.) ", first_score=SCORE_GREEN if green_first else SCORE_TAX)
    lean_text = LEAN.format(fill=FILL.format(fill=fill) if fill else "").rstrip(" ") if lean else ""
    return SYSTEM.format(branching=branching, green=(GREEN if only_green else "") + (KEEP if never_liquidate else ""), lean=lean_text, explore=EXPLORE.format(max_depth=max_depth) if explore else "", first_score=SCORE_GREEN if green_first else SCORE_TAX).replace("\n\n\nReply", "\n\nReply")


def user_message(problem, node, rejected: list[str]) -> str:
    """The problem, the plan so far and what was rejected. Every figure is calculated by code from the quotes and the portfolio."""
    lines = [f"Goal: buy {problem.shares} share(s) of {problem.target} at ${problem.price:,.2f} each = ${problem.cost:,.2f}. Free settled cash (spent first): ${problem.cash:,.2f}. "
             f"Money to raise: ${problem.need:,.2f}." + (" Only holdings that show a profit may be sold." if problem.only_green else "") + (" One share of every holding is always kept." if problem.never_liquidate else "")
             + (f" The cash left after the purchase buys as many shares of {problem.fill} (${problem.fill_price:,.2f} each) as it can, automatically; sell no more shares than the shortfall needs."
                if problem.fill else ""), "", "Holdings (symbol, shares held, sellable now, current price, cost basis per share, gain or loss per share, and whether it can raise the shortfall alone):"]
    used = {s for s, _ in node.sells}
    holdings = sorted(problem.holdings.values(), key=lambda h: (-(h.price / h.basis - 1), h.symbol)) if problem.green_first else list(problem.holdings.values())
    if problem.green_first:
        lines.insert(1, "Most in the green first (percent gain): " + ", ".join(f"{h.symbol} {h.price / h.basis - 1:+.0%}" for h in holdings))
    for h in holdings:
        alone = math.ceil(round(node.shortfall / h.price, 6)) if node.shortfall > 0 else 0
        flag = "  (already in the plan)" if h.symbol in used else "  (at a loss: may not be sold)" if problem.only_green and not h.green else ""
        can = (f"{alone} share(s) alone would raise the shortfall" if alone <= h.sellable else
               f"cannot raise the shortfall alone: all {h.sellable} sellable shares are worth ${h.sellable * h.price:,.2f}, so it is a PARTIAL step")
        lines.append(f"- {h.symbol} ({h.name or h.symbol}): held {h.shares}, sellable {h.sellable}, price ${h.price:,.2f}, basis ${h.basis:,.2f}, "
                     f"{'gain' if h.price >= h.basis else 'loss'} ${abs(h.price - h.basis):,.2f} per share, {can}{flag}"
                     + (f"; taxable share of what selling it raises: {max(0.0, (h.price - h.basis) / h.price):.0%}" if problem.explore and not problem.green_first else "")
                     + (f"; percent gain {h.price / h.basis - 1:+.0%}" if problem.green_first else ""))
    if problem.whole_plans:   # the money in each holding, so that the sums are easy to check
        lines += ["", "What each holding can raise if all its sellable shares are sold: " + ", ".join(f"{h.symbol} ${h.sellable * h.price:,.2f} ({h.sellable} x ${h.price:,.2f})" for h in holdings
                                                                                            if h.symbol not in used and problem.eligible(h))]
    lines += ["", f"Plan so far: {node.plan()}" + (f" (raises ${node.proceeds:,.2f}, still short ${node.shortfall:,.2f})" if node.sells else f" (still short ${node.shortfall:,.2f})")]
    notes = [r[len("FEEDBACK: "):] for r in rejected if r.startswith("FEEDBACK: ")]
    rejected = [r for r in rejected if not r.startswith("FEEDBACK: ")]
    if rejected:
        lines += ["", ("Plans" if problem.whole_plans else "Steps") + " the program rejected earlier, and why (do not repeat them):", *[f"- {r}" for r in rejected]]
    if notes:
        lines += ["", "The program's note on your last answer for this plan: " + notes[-1] + ". Answer again with the steps that were missing, and the other good ones."]
    if problem.whole_plans:
        return "\n".join([*lines, "", f"Give up to {problem.branching} different plans" + (" that add to the plan so far and finish it." if node.sells else " that each raise the money still needed.")])
    choices = [h for h in problem.holdings.values() if h.symbol not in used and problem.eligible(h)]
    lines += ["", "Give the next steps." + (f" {len(choices)} stocks can still be sold, so at least one step must be PARTIAL (it must not finish the plan)."
                                              if problem.explore and len(choices) >= 2 and node.depth + 1 < problem.max_depth else "")]
    return "\n".join(lines)
