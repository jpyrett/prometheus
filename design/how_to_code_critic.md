we coded the spec by pro,pt: to Fable 5.2
Wew used SOnnet 5.5 to code the spec.

---

# How the Critic was built: the notes (Sonnet 5.5, 2026-11-11, from `design/critic_v1_final_spec.md`)

These are the notes of the build, written as it went: what was made, where it lives, what the spec left open and what was chosen, and what the tests prove. The spec is the design; this file is
the record. Where the build differs from the spec, the difference is listed under "Deviations from the spec" at the end.

## Phase 1: shared building blocks (profile, calendar, work order)

**`prometheus/common/market_calendar.py` + `configs/market_calendar.json`.** Holidays are computed by the NYSE's rules (Easter by the anonymous Gregorian algorithm; observed dates on the
Friday or Monday; New Year's Day on a Saturday not observed), so the file never goes stale. The file holds only what a rule cannot know: the early-close days and `extra_closed_dates`.
`session_at(now)` returns a frozen `Session`. `window_closes_utc` is the end of the WHOLE trading window (20:00, or the regular close when there is no after-hours that day), not the end of the
sub-session: `valid_until_utc` of an approval is built from it. After-hours exists only when the next calendar day trades and today is not an early-close day; that one rule gives Jacob's "nothing
after 4 PM on a Friday" and also covers the eve of a holiday. `eastern_date(now)` is the date a recommendation is made on, and it is NEVER the UTC date (between 19:00 and 20:00 Eastern the UTC
date is already tomorrow). A naive `now` raises. Tests: `tests/test_market_calendar.py` (every vector of spec Appendix B.1, the published 2026 to 2028 holiday lists, daylight-saving
boundaries in March and November, each boundary at the minute before and at it).

**`prometheus/common/user_profile.py` + `configs/profiles/jacob_pyrett.json`.** `UserProfile` is shaped like `Portfolio` (Pydantic, `load_profile`, `ValueError` naming the file).
`risk_view()` returns a frozen `RiskView` with only the limits: no name, no mode, no auto-trading mark, because a verdict must not depend on whether the desk asks or trades. The sample is
moderate/Ask with no age (nothing was guessed). Tests: `tests/test_user_profile.py`.

**`prometheus/common/work_order.py`.** New action `verify_data`; statuses `reviewed` and `decided`; `HAS_DATA` replaces the six literal `("data_ready", "recommended")` tuples (three in
`work_order.py`, three in `data_clerk_agent.py`) so a reviewed order is never searched again. New fields `verify`, `reviews`, `critic_runs`, `trade_decisions`, `DataResult.verification`.
`verify_verification_request` (the Data Clerk's gate), `review_of`, `decision_of`, `parse_utc`, and `verify_trade`, which is the Trader's precondition: no review, no approval, no
`yes_auto`/`yes_user` decision, an approval that ran out, or more shares than approved means no trade. Tests are in `tests/test_work_order_critic.py` (a separate file: `test_work_order.py`
builds real stores for the Data Clerk and none of this needs one).

## Phase 2: the Data Clerk's part

**Three metrics** `sma_126`, `high_126d`, `low_126d` (`technicals.py`, `metrics_catalog.py`, group `price`, `None` below 126 bars with the usual note; the window is never shortened). The Advisor
does not ask for them, so nothing in the Advisor changed.

**`DataClerkTools.read_sources / read_evidence / recent_sources`** (not in `DATA_CLERK_TOOLS`, so the model is not offered them and the 13-tool pin of `test_data_clerk_tools.py` still holds).
`read_sources` reads chunks by id with `store.get(ids=...)` from the collection the id's prefix names; a chunk that is missing, an `evidence:` id and an unreadable store are each
`found: False` with a reason, never an exception. `recent_sources` has no relevance bar (the Critic wants what is NEWER, on topic or not) and asks for `newest_first`.

**`DataClerk.request_verification(order, now)`.** No model call (a test hands the clerk a model that raises if invoked). A symbol whose lookup FAILED is `listed: None`, never `False`: a
rate limit is not "not listed". The order is `data_failed` only when the request is refused or every part failed; a chunk that is not found is a result. The retrieval lock is taken around the
store reads, as the sweep does. `tests/conftest.py`'s `EmptyStore` gained `get(...)`.

**Finding while testing:** the retriever's `since_days` window has only a lower bound (`published_ts >= now - N days`), so a stored item dated AFTER the `now` given still counts as inside. That
is harmless for the real clock and worth knowing when a test passes `now`.

**The Critic's model (spec 11.1).** `ROLES["critic"]` now defaults to `z-ai/glm-5.3` (and `CRITIC_MODEL` in `.env.example`); `.env` is Jacob's: a `CRITIC_MODEL` line there overrides the
default. `get_chat_model` gained a `timeout` argument (default 60, unchanged for every other role) because GLM 5.3 reasons at `max` effort before it answers and a non-streaming reply can
take longer than the client's 60-second read wait; the Critic asks for 240 s (`calls.MODEL_TIMEOUT` is 300 s).

## Phase 3 and 4: the Critic itself (`prometheus/agents/critic/`)

```
critic.py     class Critic: the model (built lazily; a model that cannot start is a note, never an exception), the Data Clerk, the config, the reviews ledger; review_recommendations
review.py     the nine steps. `Work` carries one record through them; `_check` is steps 4 to 8 for one record; `_limit_exposure` and `_build` are step 9
integrity.py  step 2: the record against itself (the ported simple_critic checks + four new ones). No network
claims.py     the quotes a record stands on (`claims_of`): a signal and an event with the same words are ONE claim
recheck.py    steps 3 and 4: `gather` (the three Data Clerk requests), `compare` (old against new), `check_sources`, `cross_checks`, `own_equity`, `accuracy_score`. A name no longer in the catalogue is not asked for (Amendment 4: `dilution_current`)
evidence.py   step 5: the prompt, the reply's checks, the control claim, the counterfactual (the return without the quotes that did not hold)
timing.py     step 6: session, expiry, the quote, the deal price, `valid_until`
risk.py       step 7: effective tolerance, the size the profile allows, the risk score and level
value.py      step 8: the label, the blockers, the score, the bonus, the three warnings of spec 8.6
chart.py      step 8, second part (Amendment 1, 2026-11-12): fifteen chart patterns, the chart score, the dead cat bounce blocker (see "The chart dimension" below)
scoring.py    step 9: dimensions, the verdict, the cap, the summary sentence
report.py     the scorecard text (`scorecard`, `review_section`): the Ask prompt and the report print the same lines
record.py     Review, Finding, the closed list of blocker codes; config.py: configs/critic_v1.json, checked at load
```

**How a review reads.** Open `order.reviews[i]`. `approved`/`score`/`summary` is the verdict. `blockers` deny; `warnings` do not; `denial_reasons` lists the blocker codes in the order of the spec's closed
list (`record.BLOCKER_ORDER`: accuracy, timing, risk fit, value, chart, evidence), plus `score_below_minimum`. `dimensions` shows each of the SIX parts' points of its weight and its `measures`
(accuracy 30, evidence 15, risk fit 20, timing 10, value 15, chart 10 since Amendment 1; the first build had five, value 25). `chart` lists the patterns that hold and the numbers behind them. `accuracy.rechecked` is every number: recorded, fetched again, tolerance,
match. `evidence.claims` is each quote's label; `evidence.control` is the reading check; `evidence.counterfactual` is the return without the quotes that did not hold. `timing.deal_price` is the price
every judgement used and `price_basis` says where it came from. `audit` names the model, the three Data Clerk orders and the config version.

**How to add a check.** One finding in the right module, using `common.blocker/warning/info`; a blocker needs its code in `record.BLOCKER_CODES` (the closed list: the test of its size reminds you); its
threshold goes in `configs/critic_v1.json` and `config.py`, never in the code; a test that fails on its own tampering. Every check must still run when another one failed (spec P6).

**What the real captures showed** (`tests/test_critic_real.py`, Thursday 2026-11-05 10:30 ET, moderate profile). The Data Clerk's answers there are the Advisor's own numbers, so what is decided is the Critic's
own judgement. SNDK: denied, `risk_above_profile` (70.0, very high) and `falling_knife` (52% in 21 sessions); it would have scored 65. PG (0.01% edge in the first build): denied, `no_profit_potential` (since the `conviction` family it is approved: see Amendment 5). PG_open: approved, 44 of 75
shares, score 83 (B) in the first build (87 (B) since the chart dimension and the new weights: see "The chart dimension"), risk 7.1, `evidence_old` (its newest quote is 13 days old), `thin_edge`. Without a six-month average (the captures hold none) PG_open is still approved but `value_not_assessable`
makes it `requires_human`: in Auto mode it would still go to the user.

**What was found while building (each is pinned by a test).**
- `1 == True` in Python: a recorded number matched a fetched bool. `_same` now requires the same type.
- `130 / 100 - 1` is `0.30000000000000004`: a price exactly 30% above its average was "overpriced". The premium is rounded to six places.
- A default Critic built by `main` in a test wrote three lines to the REAL `data/ledger/reviews.jsonl` (the first run of the old runner test, before it got `--no-critic`). I removed that one untracked file
  and added an autouse guard to `tests/conftest.py`: no test may append to anything under `data/ledger/`.
- `./test_recommendations.sh --cash 1000` made a portfolio whose user is "cash", and the Account Manager refuses a profile and a portfolio of two users, so no decision was made. The command gives the
  cash the profile's user.
- The runner's report crashed when the position size was unknown (a portfolio whose holdings have no stored bars). It prints `n/a`.
- Jacob's own `.env` has a `CRITIC_MODEL` line, and it overrides the new default (`z-ai/glm-5.3`). The tests do not depend on it (they read the role's setting); a live run does.

## Deviations from the spec (all small; the reasons are here so a reviewer can disagree)

1. No `critic/__init__.py`: the other agent folders are namespace packages.
2. The work-order tests are in `tests/test_work_order_critic.py`, not an extension of `test_work_order.py` (that file builds real stores for the Data Clerk and none of this needs one).
3. Claims (spec 7, 8.2): a signal and an event with the SAME words from the same source are one claim; a different quote from the same source is another (the spec says one claim per source, which would lose
   the event whose class the `event_not_supported` rule needs). Refs are unique (`#2` for a second event of a class).
4. The prompt shows each source as `[ref] publisher, date, tickers` under `rag.citations.HEADER` with `_defuse`, not with `format_source`, which prints a "relevance 0.00" that means nothing here.
5. `thin_edge` is not added when `no_profit_potential` already blocks (it would say the same thing twice). `single_publisher` counts only the quotes of articles and releases (a filing-evidence item has no publisher).
6. Added: `Finding.requires_human`; `RiskDetail.allowed_shares` (the size the profile allows whatever the verdict, so a denied card does not say "size cut to 0"); config keys `limits.model_timeout_seconds` (240) and
   `limits.max_source_chars` (4000); `get_chat_model(timeout=)`; `Critic(effort=None)` (spec updated: do not lower GLM 5.3's effort); `terminal_approver` (the Ask prompt, shared by the desk and the runner).
7. A reduction: `approved_shares` is the Advisor's size, else the whole holding, and it is NEVER traded automatically (the Advisor sizes no sale): it always goes to the user, even in Auto mode.
8. The reviews ledger is append-only and never read back by this step, so the spec's "ledger problems on the order" has nothing to report and was left out.
9. `size_above_profile` is also the blocker when a position cannot be sized at all (no price, unknown equity, no stop distance): the message says which.
10. The decision logic is in `account_manager/decisions.py`; `AccountManager.decide_trades` and `record_user_decision` delegate to it (so it is testable without a model).

## What is open (yours to decide; the build did not wait)

- **A live run** (NEEDS-YES: Finnhub and the critic's model): `./test_recommendations.sh PG --no-fetch --no-ledger` during a regular session. Set `CRITIC_MODEL=z-ai/glm-5.3` in `.env` first.
- **The bake-off** (spec question 7): `--critic-model` on two or three models over the same recorded evidence with a few claims made wrong on purpose; count what each catches. A few cents.
- **Your profile**: age and horizon are empty, and the limits are the spec's defaults. The thresholds in `configs/critic_v1.json` (30% over the six-month average is overpriced, a P/E over 80 is extreme, a 25%
  fall in a month is a knife, a 0.05% profit floor) are choices, not measurements: the captures hold no six-month average to try them on.
- The early-close days are in `configs/market_calendar.json` through 2028.
- Not built: the Trader (`work_order.verify_trade` is its precondition), a refresh node before the Advisor in the desk graph, a live approval flow (a person answering `pending_user` later is
  `AccountManager.record_user_decision`, with no front end yet).

## The chart dimension (Amendment 1, 2026-11-12)

Why: Jacob asked the Critic to guard against a falling knife and a dead cat bounce and to credit a golden cross and other bullish patterns, as a weaker voice (10%) with the financial points reduced to
make room. Spec: `design/critic_v1_final_spec.md` Amendment 1. Reasons: `docs/design_choices.md` entry 62. Code: `prometheus/agents/critic/chart.py`, the six metrics in
`prometheus/agents/data_clerk/technicals.py`, the numbers in the `chart` section of `configs/critic_v1.json`.

### The score now

```
score = round( sum(weight x dimension score) / sum(weights of the applicable dimensions)  +  great-deal bonus ), at most 100; with a blocker at most 49; approved: no blocker and score >= 70
```

| dimension | weight (first build) | weight now | applies to a reduction |
|---|---|---|---|
| accuracy | 30 | 30 | yes |
| evidence | 15 | 15 | yes |
| risk fit | 20 | 20 | no |
| timing | 10 | 10 | yes |
| value | 25 | **15** | no |
| chart | - | **10** | no |

The 10 points came from `value` (the financial-metrics dimension: discount to the six-month average, expected excess, reward to risk, earnings yield, each of its four measures keeps its share of the 15). A
reduction is judged over accuracy, evidence and timing only (55 points). The weights are in `configs/critic_v1.json` and must add up to 100 (the loader refuses a file that does not).

### The numbers the patterns read

The Critic does not read bars: the Data Clerk calculates, as for every other number. Six new metrics (plain arithmetic, a window is never shortened, `None` below the bars they need):

| metric | bars | meaning |
|---|---|---|
| `sma_50_vs_200` | 200 | `sma_50 / sma_200 - 1` |
| `cross_50_200_20d` | 220 | the last crossing of the two means in the last 20 sessions: `+age` golden (it went from at-or-under to above), `-age` death, `0` none; age 1 is the latest bar. The two means are worked out again at every bar |
| `structure_63d` | 63 | three windows of 21 sessions: `up` when the highest high AND the lowest low both rise window to window, `down` when both fall, else `mixed` |
| `decline_from_peak_63d` | 63 | lowest close after the peak close (of the last 63), over the peak, minus 1 (0 when the peak is today) |
| `rebound_from_trough_63d` | 63 | latest close over that lowest close, minus 1 |
| `sessions_since_trough_63d` | 63 | how many sessions ago that lowest close was (0 = today). A tied peak is the first one, a tied low the latest one |

With them the Critic also asks for `sma_200`, `breakout_20d`, `volume_ratio_1d`, `macd_hist`, `rsi_14`, `obv_20d_change` (`chart.CHART_METRICS`, added to `recheck.OWN`: not on the record, so nothing is
compared, only judged). A real review asks for 43 metrics, under `limits.max_metrics` (60).

### The fifteen patterns

Each is true, false or UNKNOWN (a number it needs is missing: it is listed in `chart.unknown` and never guessed). Points and thresholds are in the config; the code has none.

| pattern | holds when | points |
|---|---|---|
| `golden_cross` | `cross_50_200_20d > 0` | +15 |
| `bull_alignment` | `close > sma_50 > sma_200` | +10 |
| `uptrend_structure` | `structure_63d = up` | +10 |
| `breakout_on_volume` | `breakout_20d` and `volume_ratio_1d >= 1.5` | +10 |
| `near_high` | `pct_of_52w_high >= 0.90` | +10 |
| `macd_positive` | `macd_hist > 0` | +5 |
| `healthy_rsi` | `50 <= rsi_14 <= 70` | +5 |
| `obv_rising` | `obv_20d_change > 0` | +5 |
| `death_cross` | `cross_50_200_20d < 0` | -15 |
| `bear_alignment` | `close < sma_50 < sma_200` | -10 |
| `downtrend_structure` | `structure_63d = down` | -10 |
| `dead_cat_bounce` | all five: `decline <= -0.25`, `rebound >= 0.05`, `sessions_since_trough <= 15`, `(1 + decline) x (1 + rebound) - 1 <= -0.15` (six places), `close < sma_50` | -20 |
| `overbought` | `rsi_14 >= 80` | -10 |
| `macd_negative` | `macd_hist < 0` | -5 |
| `obv_falling` | `obv_20d_change < 0` | -5 |

`breakout_on_volume` is false when there is no 20-session breakout, whatever the volume; it is unknown when there is a breakout and no volume ratio.

**Chart score** `= clip(50 + the points of the patterns that hold, 0, 100)`. 50 is a chart with no pattern: neutral, the same as the unknown credit of the other dimensions. All eight bullish patterns are
+70 (120, capped at 100); all seven bearish are -75 (0). The dimension's points are `score x 10 / 100`, so the chart is worth 0 to 10 points of the final score: a neutral chart gives 5, the best 10, the worst 0 (at most 5 points either way from neutral).

**Worked example (the captured PG_open, which predates the six new metrics).** Four patterns hold: `near_high` (98% of the high, +10), `macd_positive` (+5), `healthy_rsi` (67.9, +5), `obv_rising` (+5): 50 + 25 = 75,
7.5 of 10 points. Seven patterns are unknown (no `sma_200`, no crossing, no structure, no peak and trough numbers). The whole score: accuracy 30 + evidence 15 + risk fit 18.42 + timing 10 + value 5.6 + chart 7.5 =
86.52, rounded 87 (it was 83 with value at 25: 30 + 15 + 18.42 + 10 + 9.33 = 82.75).

### What each pattern does to a review

- **Blocker, `dead_cat_bounce`** (the only chart blocker; closed list of blocker codes is now 29; listed after the value blockers): it denies the recommendation, caps the score at 49, and forbids the great-deal
  bonus (`scoring.great_deal_bonus_allowed`: a discount on a bounce off a collapse is a trap). It needs the 50-day mean, so without one it is unknown and cannot block.
- **Warnings**: `death_cross` (a death cross in the last 20 sessions), `weak_chart` (the score under `chart.weak_below`, 35), `chart_not_assessable` (no pattern could be judged).
- **Note**: `golden_cross`.
- The falling-knife blocker (a fall of 25% in 21 sessions, or 45% under the 63-session peak) stays in `value.py`, unchanged: it is the quick rule for a collapse; the dead cat bounce is the second leg, the bounce
  after one.
- Nothing about the chart is read by a model.

### What the user sees

The scorecard (`report.py`) has a `chart` row with its points of 10 and `bullish: ...; bearish: ...` (or `no clear pattern`, `could not be judged`, `not judged for a reduction`). `Review.chart` has `patterns`
(name, side, points, the numbers in words), `unknown`, `points` and `facts` (the numbers the patterns read). A review in the ledger written before this change has the old weights and no `chart` field.

### How to add a pattern

1. A fact it needs that does not exist: a metric in `technicals.py` (with `MIN_BARS` and a line in `test_technicals.py`'s list, which fails until you add one) and in `metrics_catalog.py`; add its name to
   `chart.CHART_METRICS`.
2. A judge in `chart.JUDGES` (returns `(True | False | None, the numbers in words)`) and its name and side in `config.PATTERNS`; an assert in `chart.py` fails until both lists agree.
3. Its points in `configs/critic_v1.json` (`chart.patterns`: positive for a bullish one, negative for a bearish one; the loader refuses the wrong sign) and any threshold it needs in the `Chart` model.
4. A row in `tests/test_critic_chart.py`'s table: one case that holds at its bound and one just under it. Update the table in this file and in the spec.

### Related changes made the same night (the Advisor's side, found by the Moderna run)

These are not the Critic's, but they decide what reaches it (Advisor spec Amendments 3 and 4; design choices entries 60 and 61):
- `dilution_current` was narrowed the same night (a share count up by 1.0% or more, proceeds from issuing common stock, or an 8-K about an equity offering). It was removed later the same night: see Amendment 4 below.
- The `trend` family has a breakout path: above both the 50- and 200-day means the one-month limit is 25% (was 15%), and with 12-1 momentum of 50% or more the proximity to the 52-week high needs 80% (was 90%).
- The weights of the Advisor's blend (40% news and press releases, 30% performance, 20% historical, 10% filings) are unchanged; splitting the press releases from the news is open.
- Later the same night (Advisor spec Amendment 5, entry 63): a sixth strategy family, `conviction`, lets a high blend (0.30 or more, three of four parts, no negative news, RSI under 75) be a recommendation by itself. The captured PG (blend 78) is now a `conviction` recommendation with 0.125% after cost, so the Critic approves it (87, thin-edge warning); before it was denied `no_profit_potential` at 0.010%. The Critic's counterfactual drops the family when the blend, without the quotes the review did not accept, is under 0.30 (`test_a_conviction_recommendation_falls_when_the_quotes_it_stands_on_do_not_hold`).

### The build, in the order it was done

1. **Data Clerk first.** Six metrics in `technicals.py` with `MIN_BARS` (no window is ever shortened) and in `metrics_catalog.py`; `tests/test_technicals.py` works each by hand and checks the crossing against a plain reference loop written apart from the indicator.
2. **Config.** `weights` now has `chart: 10` and `value: 15`; `chart` section (thresholds, `dead_cat`, the 15 patterns with their points). `config.py` refuses a missing pattern, a bullish pattern with negative points and thresholds out of order.
3. **`critic/chart.py`.** A table of fifteen judges (`True`, `False` or `None` for unknown) and `assess()`. `CHART_METRICS` joins `recheck.OWN`, so the Critic asks for them in its one metric order.
4. **Wiring.** `review.py` (step 8, after the value), `record.py` (`ChartDetail`, `PatternHit`, a `chart` blocker code), `scoring.py` (label, the bonus rule), `report.py` (the `chart` row).
5. **Tests that moved.** Six dimensions in `dims()`, 29 blocker codes, the captured PG at 87 (was 83) in four files. Each new number is worked out in the test.

Things to know: the three captures predate the new metrics, so seven patterns are unknown on them (the score stays neutral for those); a live run has all fifteen. The dead cat bounce is the only chart blocker.
Tests: `tests/test_critic_chart.py` (each pattern at its bound and just under it, the cap at 100 and 0, the five conditions of the dead cat bounce one by one, unknown numbers, a reduction, and three
end-to-end reviews: a golden cross on PG, a dead cat bounce on SNDK, the scorecard row), `tests/test_technicals.py` (the six metrics by hand and against a plain reference), `tests/test_critic_config.py`
(the loader refuses a missing pattern, a wrong sign, thresholds out of order). The full suite: 2905 passed.

## What is open after Amendment 1

- **A live run with the chart** (NEEDS-YES: Finnhub and the critic's model): the first one with all fifteen patterns and the six new metrics, on a stock that has a real chart (Moderna was the reason).
- **The chart's points and thresholds are choices, not measurements** (a 25% fall for a dead cat bounce, +15 for a golden cross, 10 of 100 for the dimension): look at them after a few live runs. A dead cat
  bounce is a blocker by Jacob's request; it can be a warning by changing one line in `record.BLOCKER_CODES` and the finding in `chart.py`.
- Patterns not built: head and shoulders, double top and bottom, flags, triangles, support and resistance levels (they need swing detection, which is a judgement, not a comparison). Candles other than the
  Advisor's existing `candle_reversal_1d` are not read.
- The reviews written before the change (`data/ledger/reviews.jsonl`) are on the old five-dimension weights.

## Amendment 2: the final gate shown, and the feedback on the recommendation (2026-11-12)

Spec Amendment 2, design choices entry 64. The gate did not change: a weighted blend of the six dimensions, a blocker caps it at 49, approved is no blocker and a score of 70 or more. What changed is where the answer lives.

- **`critic/feedback.py`** builds the `critic` block of a recommendation from its review (`feedback`), puts it on the record after the review (`attach`, called at the end of `review_order`) and adds the Account
  Manager's decision to it (`attach_decisions`, called by `decide_trades` and `record_user_decision`). `Recommendation.critic` is the new optional field (`prometheus/agents/advisor/record.py`).
- **What a screen reads** (all in `order.recommendations[i]["critic"]`): `verdict` (approve or deny, score, grade, band, summary); `gate` (the blended base score, the bonus, the mark 70, the auto mark 80, the cap 49,
  what blocked, the margin); `dimensions` (the six in a fixed order: weight, score, points, share, weight share, band, headline: one bar each); `patterns` (the chart patterns that hold and those unknown);
  `findings` (blockers, warnings, notes: code, dimension, message); `size`, `risk`, `value`, `needs_human`, `audit`; `decision` (filled when the Account Manager decides, updated when the user answers).
  `band` is `strong` from 80, `ok` from 60, else `weak` (a colour). The full review is still on `order.reviews`, linked by `review_id`.
- **P5, narrowed:** the Critic never edits a field the Advisor wrote and adds only this key. `tests/test_critic.py` compares each record without `critic` byte for byte; `tests/test_critic_feedback.py` has the block's own
  tests (order of the dimensions, the points adding up to the blended score, the gate of a denial, a reduction, the decision following the user's answer).
- **Not built:** the GUI itself, and a ledger of the blocks (the reviews ledger has the reviews). The Advisor's own ledger line is written before the review and has no block.

## Amendment 3: no profit-potential rule (2026-11-12)

Spec Amendment 3, design choices entry 65. Removed from the Critic: the blocker `no_profit_potential`, the warning `thin_edge`, the warning `better_candidate_left_out`, the config keys `value.min_excess_after_cost` and
`value.thin_excess`, the `expected_excess` measure of the value score (now discount 50, reward to risk 30, earnings yield 20), `ValueDetail.excess_reviewed`, and the excess, composite and scale-factor part of the
counterfactual (it keeps the news, filings and blend worked again with the review's labels, and `blend_recorded`). Added: the evidence blocker `blend_not_supported` (a `conviction` position whose blend, without the
quotes the review did not accept, is under 0.30). The closed list is still 29 blocker codes.

Numbers that moved (the captured PG_open): value 6.58 of 15 (it was 5.6), score 88 (B) (it was 87). The Appendix B.3 value examples are 96 (a price 12% under its average) and 42 (8% over). The earlier notes in this file
that name `no_profit_potential` or the thin-edge warning describe the first build.

Same night, the Advisor's side (Advisor spec Amendment 6, entry 66): no cost hurdle. A name is bullish when a bullish strategy passed; the blend ranks the candidates and is the score (`round(50 + 50 x blend)`);
`excess_20d_after_cost` stays on the record (the Critic's integrity check still works its arithmetic) but decides nothing and no report, thesis or card shows it; the thesis no longer has to say "unmeasured".
The Critic's integrity checks follow: `stance.action` needs a composite above zero and `recompute.score` is the blend's. A Critic review of an Advisor record now sees a stance made by the strategy, not by the cost.

`IGNORE_MARKET_HOURS` (entry 69): set `true` in `.env` and a closed market is a note (`market_hours_ignored`), not the `market_closed` blocker. A recommendation from an earlier trading day is still `record_expired`. Tests force the flag off.

Advisor Amendment 7 (entry 68): `ranking.excess_20d_after_cost`, `cost_pct` and `scale_factor` are removed from the records, with the config keys behind them (`ranking.edge_scale`, `default_half_spread`, `slippage`,
`stress_cost_multiplier`, `blend.scale_strength`, `no_strategy_edge_scale`). The Critic's integrity checks `recompute.scale` and `recompute.excess` are removed; it still works again the composite, the blend, the
confidence and the score. A realized-outcomes tracker was declined (past outcomes do not predict the future).

## Amendment 4: `dilution_current` is not evaluated (2026-11-12)

Spec Amendment 4, Advisor spec Amendment 8, design choices entry 70. Jacob: issuance is not a reason to deny, haircut or score a recommendation. Visa (2026-11-12) never reached a review because the metric was true. The Critic had no dilution blocker of its own. It denied an opening whose stored `red_flag` gate had failed, and it asked the Data Clerk again for `dilution_current` whenever the record's strategy or blend had used it (`recheck.key_metrics` keeps a name only when it is still in `metrics_catalog.METRIC_NAMES`).

**What stopped.**

- The Data Clerk no longer computes it. `fundamentals.dilution_current` and the catalogue entry are gone, so a request that names it is refused as an unknown metric.
- The Advisor no longer asks for it (`recommend.GATE_METRICS`). Removed with it: the `red_flag` leg, the quality haircut (factor 0.6), the filings blend measure (a bad reading scored −1 and was silent when false), `capital_return`'s required input, that family's bearish vote (score 0.6, prior −0.3%, including a cited 8-K or a derived dilution item), and the family's invalidation `metric_crossed: dilution_current`. The config keys `families.capital_return.prior_bearish` and `bearish_score` go with the vote.
- A review therefore neither fetches nor compares the number. An older ledger line whose invalidation still says `metric_crossed: dilution_current` does not fire that rule (`ranking.invalidation_fired` no longer recognizes it).

**What a review still does.** The integrity check `gates` (`integrity.py`) still denies an opening whose stored gates failed, as `record_defect`. A recommendation written while the red-flag leg existed keeps that failure until the Advisor writes it again; the Critic does not reopen a stored gate. `share_count_change`, when the record used it, is still asked for again and compared. On the Advisor's side that metric is unchanged: a rising share count still fails `capital_return`, and a rise over 1% still haircuts quality (factor 0.8).

**What is not this rule.** A news item the model classes `dilution_offering` can still keep a name out of `reversion` (a fall with a company-specific cause). A dilution passage the model cites still counts as one bearish vote in the filings part of the blend, the same way any cited filing passage votes. The filing store still keeps dilution passages; they are evidence, not a metric.

Tests: `tests/test_families.py` (a cited dilution item leaves `capital_return` bullish), `tests/test_gates.py` and `tests/test_quality.py` (no issuance flag or haircut), `tests/test_ranking.py` (an old invalidation rule does not fire), `tests/test_metrics_catalog.py` (the name is not in the catalogue). The full suite: 2913 passed.

## Amendment 5: the close check and the second quote (2026-11-12)

Spec Amendment 5, design choices entry 71. Two checks were using the wrong partner.

- **`price_mismatch`** (`recheck._close_from_quote`). The stored bar is the regular-session close. The quote's `price` is the last trade, and it is that close only when the trade's time is in the regular session, including the print at the close. A pre-market or after-hours print on the bar's date is not compared, so a move after the close does not cap the score. The bar from the trading day before the quote's date is still compared with `previous_close`.
- **The counterfactual** (`evidence.counterfactual`). Two quotes from one source are two blend rows with one ref and two claim refs (`[4]` and `[4]#2`, and the same for `E3` and `E3#2`). Each row takes its own label. A rejected second quote moves the blend, including `blend_not_supported`.


## Note of 2026-11-14 (design choices entry 80): the Critic does not size and does not receive the portfolio

The text above is left as it was written; where it describes a size, this note wins. `Critic.review_recommendations(order, risk, now)` takes no portfolio, and `RiskView` holds no size limit, so the Critic cannot learn the user's money. How much to buy or sell is the Account Manager's (`prometheus/agents/account_manager/sizing.py`, called by `decisions.decide_trades`). What was removed from the Critic: `risk.size_caps` and `exposure_cap` (the arithmetic moved to `sizing.py`), `approved_shares` and `suggested_shares` of the review, `RiskDetail`'s caps, `allowed_shares`, `equity` and `equity_basis`, the blockers `size_above_profile` and `exposure_limit` (the closed list has 27 codes), the warning `size_reduced`, step 9's exposure limit (`_limit_exposure`) and the `position_pct` risk measure; the integrity checks `trade.size` and `recompute.position` went with the Advisor's size (`trade.order_style` stays). The other risk weights were scaled to total 100 (27.8, 16.7, 11.1, 11.1, 11.1, 11.1, 5.6, 5.5) and the scale of the risk score and its `levels` are kept, so PG_open's risk is 5.2 and SNDK's 77.8. In "Deviations from the spec", `RiskDetail.allowed_shares` in item 6, and items 7 and 9 (a reduction's `approved_shares`, `size_above_profile` for a position that cannot be sized) describe what no longer exists; the rest of item 6 stands. A reduction is still never traded automatically: its amount, the whole holding, is the Account Manager's.
