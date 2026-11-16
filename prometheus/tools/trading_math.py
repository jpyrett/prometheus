"""Simple trading arithmetic that the Tree of Thought's EVALUATOR uses: how green-first a plan is (design choices entry 45).

No LLM, no network: loops and sums. It decides nothing about WHAT to sell: the Tree of Thought does that (prometheus/reasoning/raise_funds.py). This module only measures a plan the tree has built.

Jacob's rule (2026-11-03): "sell the most profitable (the stocks most in the green) first". Holdings are ranked by PERCENT GAIN, (price - cost basis) / cost basis, highest first. That is also the order
that realises the most profit for a given amount of money raised: the profit per dollar of proceeds is (price - basis) / price = gain% / (1 + gain%), which grows with gain%. So the most profit that
selling a given amount can realise is found by going down that ranking (`profit_bounds`), and a plan is as green-first as its realised profit is close to that.

A holding here is a plain dict {"symbol", "shares" (the SELLABLE shares), "price", "basis"}. Money and shares are `decimal.Decimal` (design choices entry 112; design/decimal_money_spec.md): the percent gain is the exact ratio, the amounts are exact
products and quotients, and the bounds are rounded ONCE, to four places, half up (the scale a price has), at the end. A float is a TypeError (`money.exact`): it comes in through a door.
"""
from decimal import ROUND_HALF_UP, Decimal

from prometheus.common.money import EXACT, exact

BOUND_STEP = Decimal("0.0001")      # the profit bounds are worked to four places, as a price is


def pct_gain(price: Decimal, basis: Decimal) -> Decimal:
    """Percent gain as a fraction: 0.5 is 50% in the green, -0.2 is 20% in the red. The exact ratio, a Decimal."""
    return EXACT.divide(exact(price, "a price"), exact(basis, "a cost basis")) - 1


def rank_green(holdings: list[dict], only_green: bool = False) -> list[dict]:
    """The holdings with sellable shares, most in the green first (highest percent gain), ties by symbol. With `only_green`, holdings at or below their cost basis are left out."""
    usable = [h for h in holdings if h["shares"] > 0 and (not only_green or h["price"] > h["basis"])]
    return sorted(usable, key=lambda h: (-pct_gain(h["price"], h["basis"]), h["symbol"]))


def profit_bounds(holdings: list[dict], amount: Decimal, only_green: bool = False) -> tuple[Decimal, Decimal]:
    """(the most, the least) profit that selling `amount` dollars' worth could realise, fractions of a share allowed: the most by selling the most profitable holdings first, the least by selling the
    least profitable first. Used to say how green-first a plan is. (0, 0) when `amount` is 0; when `amount` is more than the holdings are worth, the bounds of selling all of them."""
    def fill(order):
        left, total = exact(amount, "an amount"), Decimal(0)
        for h in order:
            take = min(left, EXACT.multiply(h["shares"], h["price"]))      # what the sellable shares are worth: the exact product (a bound is worked to four places at the end, not to the cent here)
            total += EXACT.divide(take * (h["price"] - h["basis"]), h["price"])
            left -= take
            if left <= 0:
                break
        return total
    ranked = rank_green(holdings, only_green)
    return fill(ranked).quantize(BOUND_STEP, rounding=ROUND_HALF_UP), fill(list(reversed(ranked))).quantize(BOUND_STEP, rounding=ROUND_HALF_UP)
