"""The brute-force best plan for the complex RaiseFunds scenario: what the Tree of Thought's answer is measured against (design/week4_tot_guide.md section 9i).

    python scripts/raise_funds_oracle.py          # no network, no model, about 2 seconds

For each purchase (3 LLY, 1 SNDK, up to 3 MU, in order, from jacob_pyrett_complex.json at the recorded 2026-11-03 close) it tries EVERY whole-share combination of the holdings that may be sold (in
profit, one share of each kept: the evaluator of entry 47) and keeps the best by the code's own score (`--green-first`, lean). A plan is lean when removing one share of its dearest sold stock would leave it
short (that stock can be the last step; the steps before it cannot overshoot). The next purchase starts from the portfolio the best plan of the one before leaves, so each line is "the best plan for this purchase, given the best before". It is a check on the tree, not part of it.
`oracle()` is also used by scripts/raise_funds_beam_experiment.py.
"""
import itertools
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prometheus.common import portfolio as pf  # noqa: E402
from prometheus.reasoning import raise_funds as rf  # noqa: E402

PRICES = {"LLY": 1409.38, "SNDK": 1126.52, "MU": 855.10, "F": 12.31, "GM": 88.34, "FITB": 53.93, "PG": 162.06, "MCD": 263.04, "GOOGL": 280.03, "TSLA": 398.50, "MSFT": 520.99}
TODAY = date(2026, 11, 3)
SCENARIO = Path(__file__).resolve().parents[1] / "configs" / "portfolios" / "jacob_pyrett_complex.json"


def oracle(portfolio, target, shares, max_stocks, prices=PRICES, never_liquidate=True):
    """(best score, its sells), the number of valid lean plans, for buying `shares` of `target` from `portfolio`: every whole-share combination of the holdings in profit (one share of each kept unless
    `never_liquidate` is off), at most `max_stocks` stocks. (None, 0) when there is no plan."""
    prob = rf.build_problem(portfolio, target, shares, prices, TODAY, only_green=True, lean=True, green_first=True, never_liquidate=never_liquidate)
    holdings = [h for h in prob.holdings.values() if prob.eligible(h)]
    need, best, count = prob.need, None, 0
    for counts in itertools.product(*[range(h.sellable + 1) for h in holdings]):
        k = sum(1 for c in counts if c)
        if k == 0 or k > max_stocks:
            continue
        total = sum(c * h.price for c, h in zip(counts, holdings))
        if total < need or total - max(h.price for c, h in zip(counts, holdings) if c) >= need:   # short, or not lean: even the last step, of the dearest sold stock, could have sold a share fewer
            continue
        sells = tuple((h.symbol, c) for c, h in zip(counts, holdings) if c)
        score = rf.score_plan(prob, sells)[0]
        count += 1
        if best is None or score > best[0]:
            best = (score, sells)
    return best, count


def apply(portfolio, target, shares, sells, prices=PRICES):
    """The portfolio after a purchase paid for by `sells` ([(symbol, shares)])."""
    proceeds = sum(k * prices[s] for s, k in sells)
    return pf.after_purchase(portfolio, target, shares, prices[target], sells, round(portfolio.cash_avail + proceeds - shares * prices[target], 2), TODAY)


PURCHASES = [("LLY", 3), ("SNDK", 1), ("MU", 3)]


def main() -> int:
    started, current = time.time(), pf.load_portfolio(SCENARIO)
    for target, asked in PURCHASES:
        shares, best = asked, None
        while shares >= 1:   # "up to": the most shares, at most the number asked, that some plan can pay for
            best, count = oracle(current, target, shares, 7)
            if best:
                break
            shares -= 1
        if not best:
            print(f"{target} x{asked}: no plan pays for even one share: nothing is sold")
            continue
        for max_stocks in (5, 7):
            print(f"{target} x{shares} (asked {asked}): best with <= {max_stocks} stocks: {oracle(current, target, shares, max_stocks)[0]}  [{time.time() - started:.0f}s]")
        current = apply(current, target, shares, list(best[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
