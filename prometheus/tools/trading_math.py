"""Simple trading arithmetic that the Tree of Thought's EVALUATOR uses: how green-first a plan is (design choices entry 45).

No LLM, no network: loops and sums. It decides nothing about WHAT to sell: the Tree of Thought does that (prometheus/reasoning/raise_funds.py). This module only measures a plan the tree has built.

Jacob's rule (2026-11-03): "sell the most profitable (the stocks most in the green) first". Holdings are ranked by PERCENT GAIN, (price - cost basis) / cost basis, highest first. That is also the order
that realises the most profit for a given amount of money raised: the profit per dollar of proceeds is (price - basis) / price = gain% / (1 + gain%), which grows with gain%. So the most profit that
selling a given amount can realise is found by going down that ranking (`profit_bounds`), and a plan is as green-first as its realised profit is close to that.

A holding here is a plain dict {"symbol", "shares" (the SELLABLE shares), "price", "basis"}.
"""


def pct_gain(price: float, basis: float) -> float:
    """Percent gain as a fraction: 0.5 is 50% in the green, -0.2 is 20% in the red."""
    return price / basis - 1.0


def rank_green(holdings: list[dict], only_green: bool = False) -> list[dict]:
    """The holdings with sellable shares, most in the green first (highest percent gain), ties by symbol. With `only_green`, holdings at or below their cost basis are left out."""
    usable = [h for h in holdings if h["shares"] > 0 and (not only_green or h["price"] > h["basis"])]
    return sorted(usable, key=lambda h: (-pct_gain(h["price"], h["basis"]), h["symbol"]))


def profit_bounds(holdings: list[dict], amount: float, only_green: bool = False) -> tuple[float, float]:
    """(the most, the least) profit that selling `amount` dollars' worth could realise, fractions of a share allowed: the most by selling the most profitable holdings first, the least by selling the
    least profitable first. Used to say how green-first a plan is. (0.0, 0.0) when `amount` is 0; when `amount` is more than the holdings are worth, the bounds of selling all of them."""
    def fill(order):
        left, total = amount, 0.0
        for h in order:
            take = min(left, h["shares"] * h["price"])
            total += take * (h["price"] - h["basis"]) / h["price"]
            left -= take
            if left <= 0:
                break
        return total
    ranked = rank_green(holdings, only_green)
    return round(fill(ranked), 4), round(fill(list(reversed(ranked))), 4)
