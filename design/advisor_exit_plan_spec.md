# Advisor: the exit plan, and outcomes that do not depend on it (spec)

Open item 1 of the 2026-11-20 handoff: "the TREND family is the weak one (60% of its closed recommendations won against 93% for conviction, with a stop about three times as far as its target); the Advisor's highest scores did worst". Jacob asked for a spec (2026-11-20, night). Design choices entry 139 (written by the main session).

## 1. What the data says (measured 2026-11-20 on the four newest fast six-month runs: oil, AI, restaurants, consumer staples; 610 unique recommendations)

Two measurements. The PLAN's outcome (`outcomes[].result`, exit by stop / target / time stop) depends on where the Advisor put the target and the stop. The FORWARD excess return (the stock's close N sessions after the decision day against SPY's, from the stored bars) does not: it measures the PICK.

| family | n | plan: target / stop (avg % from entry) | plan: won / closed | forward excess 10 sessions | 20 sessions | 40 sessions |
|---|---|---|---|---|---|---|
| trend | 141 | +2.0% / -5.6% | 70/130 (54%) | +0.26% (se 0.64) | -0.16% (se 0.91) | +0.35% (se 1.32) |
| conviction | 90 | +0.8% / -8.6% | 63/76 (83%) | +0.02% (se 0.75) | -0.89% (se 0.96) | +1.70% (se 1.22) |
| capital_return | 10 | none / -7.7% | | +3.11% (se 1.91) | +6.89% (se 2.53, n 7) | |
| neutral (no family) | 335 | stop only | | **-1.08% (se 0.40)** | **-1.20% (se 0.62)** | -1.02% (se 0.82) |

Advisor score buckets (forward 20 sessions, families only): under 60 +0.74% (n 12), 60-65 -1.90% (25), 65-70 +1.08% (66), 70-75 -1.08% (77), 75+ +0.70% (36): every bucket within about one standard error of zero.

Findings:
1. **Trend does not pick worse than conviction.** Forward, the two are the same within chance. Conviction's 83-93% "won" is its target: +0.8% above entry against a stop 8.6% below, so almost any wiggle up is a "win" and a loss is ten times the size. Trend's 54% is the same geometry, a little less lopsided. The win rate measures where the target was put, not the pick.
2. **"The highest scores did worst" is not in the data**: no score bucket differs from another beyond chance.
3. **The target formula is the defect.** `target = entry x (1 + 1.5 x edge_20d x horizon/20)`; with the families' small priors (0.006 a month for trend) the target is 1-2% for a 40- to 63-session idea, while the stop is 3 ATR (5-9%). Reward to risk is 0.1-0.4. A trend-following idea (Jegadeesh and Titman 1993, the family's own basis) is held to let the winners run; the plan cuts every winner at +2%.
4. **Neutral purchases lag SPY** by about 1.1% at 10 sessions (2.7 standard errors) and 1.2% at 20 (1.9 se). Not built here: Jacob's setting (`ALLOW_NEUTRAL_RECOMMENDATIONS`), reported to him.
5. **AI**: both families lag SPY by about 3% at 20 sessions in the AI theme (trend -3.51% se 1.69, conviction -2.69% se 1.36): a theme effect, not a family one. The Critic v2's AI gap (+1.86, se 1.58) is within chance.

Limits: the weeks overlap (the standard errors flatter), one stretch of the market, the thresholds below are conventions chosen by reason and NOT tuned on these outcomes (entry 133's rule).

## 2. The rules

| # | Rule |
|---|---|
| E1 | **A target is at least `trade.min_reward_to_risk` (2.0) times the stop distance above the entry mid**: `target = entry_mid + max(entry_mid x 1.5 x edge, min_reward_to_risk x stop_distance)`. A convention (2:1 is the textbook minimum), not tuned. Applies to `conviction`, `event`, `reversion`. `capital_return` keeps no target. |
| E2 | **TREND has no target and a TRAILING stop**: the stop starts at the plan's stop price and, at each close, rises to `highest close since entry - stop_distance` (never falls). The time stop stays (`horizon_bars`, 63). The record's `trade` gains `trailing: true` and `trail_distance` (= `stop_distance`, a price distance fixed at the decision); `target_price` is null. The family's own invalidation (`below_sma_200`) is unchanged. |
| E3 | **Nothing changes in the Trader or a live account.** The Trader places no stop or target order (unchanged); the Live view only WATCHES (P11) and now reports the trailing stop's current level ("XOM's trailing stop is now 142.10 (highest close 150.30 - 8.20)"). |
| E4 | `simulator.follow` gains `trail_distance: Decimal | None`: when given, the stop for bar i is `max(stop_price, highest close of bars 0..i-1 - trail_distance)` (the close BEFORE bar i, so no bar's own high is used to move its own stop: no look-ahead inside a day); the exit reason is `trailing_stop` when the stop that fired was above the original stop, else `stop`. Decimal throughout (the doors: `price_scale`, `bar_price`). The Proving Ground's `outcomes.py` and `portfolio.py` and the venue pass it. |
| E5 | **Every outcome gets a FORWARD block that does not depend on the plan**: `forward: [{"sessions": 5|10|20|40, "return_pct", "benchmark_return_pct", "excess_return_pct"}]` from the decision day's close to the close N sessions later (None when not yet stored). Computed once in `proving/stats.py`/`outcomes.py` (P12); the page computes nothing. |
| E6 | **The family and score tables read the forward excess** (`stats.py`, the report card, the run view): per family and per Advisor score bucket (`<60, 60-65, 65-70, 70-75, 75+`), with n and se and the chance band (P12, `CHANCE_BAND`). The win rate stays, labelled "won by its plan", beside it. |
| E7 | The Critic's `reward_to_risk` value measure (`critic/value.py`) treats a plan with no target and `trailing: true` as NOT APPLICABLE (out of the value score's weights), never as zero and never as unknown-credit. |
| E8 | Old records (no `trailing`) load and follow exactly as before (a test on a stored run: the outcomes of a saved run re-followed with the old records are unchanged). |

## 3. Config (`configs/advisor_v1.json`)

`trade.min_reward_to_risk: 2.0`, `trade.trailing_families: ["trend"]`. The `_note` says both are conventions. Bump the advisor config version.

## 4. What changes, module by module

- `advisor/trade_plan.py`: E1, E2. `advisor/record.py`: `TradePlan.trailing: bool = False`, `trail_distance: float | None = None` (the Advisor's plan stays float measurements: entry 112). `advisor/config.py` validates the keys.
- `agents/trader/simulator.py`: E4. `agents/trader/venue.py`, `proving/outcomes.py`, `proving/portfolio.py`, `proving/record.py` (`PlanView` gains `trailing`, `trail_distance`): pass it.
- `proving/live.py`: E3 (watch only: a test reads its source for sells/places/writes, P11).
- `proving/stats.py`, `proving/report_card.py`, `application/web/proving_views.py`, `ui/src/proving/`: E5, E6.
- `critic/value.py`: E7 (after the Critic v3 worker is done; that worker owns `critic/`).
- Not touched: the Trader's tickets, `decisions.py`, the Account Manager, the Critic's rules.

## 5. Tests (real data)

- E1/E2 on real recorded metrics (`tests/fixtures/`), E4 on the real stored bars of a symbol with a clear run-up then a drop (find one with a throwaway script; name it in the test with how it was found), E8 on a saved run's outcomes, E5 against the stored bars by hand for one outcome, E7 on a real record. List every existing test whose expected value changes.
- One Time Machine re-run per theme FROM THE CACHE if the research cache still serves (the plan is built in phase 1: if the cache holds the plans, the re-run must rebuild them; if that means a paid phase 1, the four fast themes cost about $0.14-0.30 each and are covered by Jacob's standing yes for paid runs that TEST the desk; no fetch). Report before/after: per family, won by plan, plan excess, forward excess.

## 6. Open for Jacob

- 2:1 and a trailing stop at the original stop distance are conventions; the trail could be tighter (2 ATR).
- Neutral purchases lag SPY (finding 4): keep `ALLOW_NEUTRAL_RECOMMENDATIONS`, or not.
