# Week 5: how the Advisor generates stock recommendations (a build guide)

Jacob Pyrett, CMU Agentic AI capstone, written as the code was built (2026-11-05) and extended on 2026-11-06 (sections 9 and 10: the typed-request runner, negative equity, the blend). Sections 1 to 8 describe the build as it was on the 5th; where a later change replaced something, a note says so and points to the section that explains it. For a one-page orientation read `docs/SESSION_NOTES.md` first. This is the record of HOW the "Generate Recommendations" step of
`design/advisor_v1_final_spec.md` was implemented, in the order it was built, so that you can read the code with the reasons beside it. It
is a guide, not a second spec: when it says "the spec says X, the build does Y" the reason is given. The design decisions are also numbered
entries at the end of `docs/design_choices.md` (entry 51 onward).

---

## 0. The idea in one page

A trading desk's Advisor answers one question: **"given the data the Data Clerk gathered, which stocks are worth a trade, and why?"**. A
prompt-only Advisor answers fluently and invents numbers (week 2: "a prompt will write a score that hallucinates"). So this Advisor is split
into what a program can do and the two things only a model can do:

| Done by code (checked, repeatable) | Done by the model (and then checked by code) |
|---|---|
| collecting the candidate stocks | PLAN call: which strategies each company's text supports, with exact quotes, the corporate events it sees |
| every number: returns, RSI, ATR, ratios (the Data Clerk calculates them) | THESIS call: three sentences and bull/bear bullets written from the finished record |
| the gates (is it liquid, fresh, free of red flags) | |
| scoring each strategy family, the quality haircut, the regime | |
| the blend (40% news and press releases, newer weighs more; 30% performance; 20% historical; 10% filings) that scales each edge | |
| the expected excess return after cost, the ranking, the sleeve rules | |
| the trade plan: entry, stop, target, size, order style, invalidation rules | |
| checking that every quote is in its source and every number in the thesis exists in the record | |

The **agentic part** is the shape of the loop, not the amount of model: the Advisor reads shared state (the work order), asks another agent (the
Data Clerk) for the data it is missing, reasons over it with tools it owns (its functions), writes a structured result that the next agent (the
Critic) can audit, and records what it did (`advisor_runs`, the ledger). The model is used where language is the problem; arithmetic is never
left to it.

### Where it lives (one method on the existing class)

You asked for it in the Advisor class and not as a separate `AdvisorV1` (the spec's name), so:

```python
order = Advisor().generate_recommendations(order, portfolio, case="funds", today=date(2026, 11, 5))
```

`Advisor.advise` (version 0, the week 3 baseline) is untouched and still works. `generate_recommendations` is plain Python that a LangGraph node
can call: it reads ONE work order (status `data_ready`) and returns the SAME order with `status == "recommended"`, `recommendations`,
`candidates` and `advisor_runs` filled. Inside it, the request for metrics is a sub-step (a call to the Data Clerk), not a separate graph node.

### The files, by layer

```
prometheus/agents/data_clerk/   bars.py  technicals.py  fundamentals.py  metrics_catalog.py     (phases 1-2: the Data Clerk calculates numbers)
                                data_clerk_agent.py: request_metrics          filing_signals.py (extended)    data_toolbox.py (more Finnhub fields)
prometheus/tools/data_clerk.py  calculate_metrics + metrics_for               prometheus/common/work_order.py (new fields, verify_metric_request)
prometheus/agents/advisor/      advisor.py (the method)  config.py  record.py  events.py  gates.py  quality.py  families.py  blend.py  ranking.py  trade_plan.py  candidates.py  thesis.py  calls.py
prometheus/prompts/advisor_v1.py   prometheus/memory/ledger.py   configs/advisor_v1.json   configs/sector_etfs.json
prometheus/application/recommendations.py + ./test_recommendations.sh      (section 9: a typed request to a ranked report)
tests/                          one test file per module, and test_advisor_sndk_pg.py (the real-data check with a tiny Critic)
```

(Sections 3 onward are added below as each phase is built.)

---

## 1. Phase 1: daily bars and indicators (the Data Clerk's raw material)

**Problem.** Every strategy needs price history: returns, RSI, volatility. The history is on disk as the `day_aggs_v1` files (one gzip CSV per
trading day holding EVERY US ticker). Nothing could read it as a time series.

**What was built.**

- `bars.py`: `BarStore(root).load(symbols, end=today)` reads the newest 300 day files ONCE, keeps only the symbols asked for (always SPY and the
  11 sector ETFs too) and exposes `bars(symbol, end, max_bars)`. `window_start` (nanoseconds since the epoch, UTC) becomes the trading date. It
  never writes. A weekday with no file between the first and last file is named in `gaps` ("a market holiday, or not synced": the store cannot
  tell which); a corrupt file is named with its date and the error kind. **No bar is ever invented to fill a gap.**
- `technicals.py`: pure functions over a list of `Bar`: `ret`, `mom_12_1`, `pct_of_52w_high`, `sma`, `breakout_20d`, `gap_1d`, `candle_reversal_1d`,
  `rsi_14` (Wilder), `macd_hist`, `bollinger_pct_b`, `zscore_21d`, `adx_14`, `atr_14`, `atr_pct`, `obv_20d_change`, `stoch_k_14`, `vol_ann`,
  `max_drawdown_63d`, `beta_and_corr`, `adv_20d_dollars`, `volume_ratio_1d`. Each has a `MIN_BARS` and returns `None` below it.
- `configs/sector_etfs.json`: the 11 SPDR sector ETFs (XLK, XLF, ... XLC). Their `industries` lists are empty on purpose: the industry names
  belong to Finnhub's profile endpoint, which is not added yet, and I will not guess its vocabulary.

**The rule that matters most: never shorten a window to make a number appear.** Only October 2026 is on disk (17 sessions). So `rsi_14` (30 bars),
`sma_200`, `mom_12_1` (252) and `beta_63d` are `None`, with the note `needs 30 bars, 17 stored`. A model that is shown a stretched 17-bar
"200-day average" would treat it as one. Every test that matters runs the 17-bar case.

**How to read an indicator.** Take RSI. The textbook step is `avg = (prev * 13 + current) / 14`. `tests/test_technicals.py` checks it by hand:
14 changes of +1 (8 times) and -1 (6 times) give gains 8/14 and losses 6/14, so RSI = 100 - 100/(1 + 8/6) = 57.14; one more +1 change gives
`avg_gain = (8/14 * 13 + 1) / 14`. A ramp has RSI 100 (no losses) and the MACD histogram of a ramp is exactly 0 (an EMA of a ramp lags by a
constant, so the line and its signal are equal): that is a *proof* the seeds and the recursion are right, not just a number that looks plausible.

**Mutation check** (break one rule, see a test fail, restore): I changed the Wilder smoothing from 13/14 to 12/13: the first version of the tests
did NOT notice (they only tested the first 14 changes, and a symmetrical alternating series gives about 50 either way). I added the one-step
smoothing tests above, broke it again and the suite failed. That is why those tests exist.

---

## 2. Phase 2: the Data Clerk calculates metrics (`request_metrics`)

**Problem.** The Advisor needs about 80 numbers for each candidate. Who computes them? Not the Advisor (it is a reasoner, not a spreadsheet), and
never the model. The Data Clerk is the desk's data agent and already has `request_quotes` for prices; metrics follow the same pattern.

**The call.** The Advisor builds ONE work order with `action_type="calculate_financials"`, `subject` = the symbols and `metrics` = a list of
`{"name": "rsi_14"}`; the Data Clerk answers with `DataClerk.request_metrics(order)`. No model call: a number is a calculation. The result is on
`order.data.metrics[symbol][name]`, every one the same object:

```
{"value": 61.2, "unit": "0-100", "as_of": "2026-10-23", "window": "last 30 bars", "source": "day_aggs:2026-10-01..2026-10-23",
 "inputs_summary": {"bars": 45, "last_close": 101.2}, "note": null}
```

- `source` says where a number came from: `day_aggs:<first>..<last>`, `finnhub_metric:<date>`, `filing:<accession>`, or `derived:a+b` when two sources were combined.
- `inputs_summary` holds what a Critic needs to **recompute** it.
- `note` is a gap in words. A `value: None` always has a note ("needs 30 bars, 17 stored", "Finnhub has no value for ps_ratio for AAA"). **A gap is a result, never an error and never a guess.**

**Pieces.**

| Piece | What it does |
|---|---|
| `work_order.py` | new optional fields `metrics`, `case`, `sector`, `candidates`; `DataResult.metrics`, `market_brief`; `verify_metric_request` (the check, like `verify_quote_request`); an order that carries a new field is `schema_version` 2, an old order is untouched |
| `fundamentals.py` | ratios from Finnhub's payload and from the stored filings: `earnings_yield`, `net_payout_yield_ttm`, `share_count_change`, `payout_coverage`, `sbc_to_revenue`, `accrual_ratio`, `receivables_vs_sales`, `inventory_vs_sales`, `cfo_negative_with_profit`, `dilution_current`; each returns `(value, inputs)` or raises `Missing(reason)` |
| `metrics_catalog.py` | the table: `METRICS` (82 entries), `compute(...)`. Adding a metric = one entry + one test |
| `filing_signals.py` | extended (still scores nothing): receivables, inventories and cash from operations rows; each cash-flow row now carries `units` and `months` |
| `data_toolbox.py` | `FINANCIAL_METRICS` grew from 10 to 19 Finnhub fields (market cap, beta, 52-week high/low, P/S, P/B, payout ratio, EPS growth...); `parse_metric_payload` is the pure part of `Financials.get` so a test can feed it a recorded payload |
| `tools/data_clerk.py` | `DataClerkTools.calculate_metrics(symbols, metrics)` (the tool the model and the MCP server see, max 10 symbols) and `metrics_for(...)` (the code path, max 30) |
| `data_clerk_agent.py` | `request_metrics`, `DEFAULT_CALLS["calculate_metrics"] = 2` |

**Things learned from the real filings (they shaped the code).**

1. *A 10-Q's cash-flow statement is year to date, and does not say so.* Apple's first-quarter 10-Q covers 3 months, META's third-quarter one 9.
   The statement that DOES say it is the income statement's column headings ("Three Months Ended | Nine Months Ended"); a first-quarter 10-Q has
   only one heading, but only two columns instead of four. `filing_signals.period_months` reads that, and the net payout yield is annualized by
   `12 / months`. If the months cannot be read the yield is `None` with that reason.
2. *Units differ.* "(In millions, except ...)" vs thousands: every row carries its `units` line and `fundamentals.multiplier` turns it into dollars.
   Finnhub quotes market cap in millions: converted to dollars in `parse_metric_payload` and the unit says `USD`.
3. *The income statement has two shapes.* Four values (this quarter, last year's quarter, year to date, last year's year to date) or two. `fundamentals.ytd` picks the year-to-date pair so that "net income" and "dividends paid" cover the SAME period.

**Spec vs build (flagged for your review).**

| Spec says | Build does | Why |
|---|---|---|
| `MAX_METRICS = 40` | 100 | the union of what the families, the gates and the plan read is 82 distinct metrics |
| `sector: str` on the order | `str` or `{symbol: ETF}` | one order covers several companies from several sectors; a single ETF would give the wrong residual to every holding that is not in it |
| `net_payout_yield_ttm`: "any input missing -> None" | needs at least one of repurchases/dividends; an absent row counts as 0 and is named in `inputs_summary.absent` | a statement leaves out a line that is zero (Apple has no "proceeds from issuing stock" line); requiring all three would make the metric `None` for nearly everybody |
| `dilution_current`: an 8-K **or a 10-Q** passage about dilution within 120 days | 8-K passages only; plus shares outstanding rose, or a cash-flow row of proceeds from an *issuance* of common/class stock (not employee options) | the dilution topic's words ("dilutive effect", "convertible notes", "warrants") are in every company's 10-Q (the anti-dilutive securities note), so the literal rule marks every company as diluting and would fail the red-flag gate for all of them |
| `risk_factor_change` (embedding similarity of two 10-Ks) | `None` with the note "not built" | its 0.35 threshold is calibrated on embedding cosine; a word-overlap fallback has a different scale and would haircut everyone. Skipped, not faked |
| earnings (`sue_4q`, `days_to_next_earnings`...), `median_spread_bps_1d`, `breadth_200` | catalogue entries that are always `None` with the reason (`UNAVAILABLE`) | they need endpoints you have not approved or a universe file that does not exist; having the entry means the gap is visible in the record |

**Tests.** `test_bars`, `test_technicals`, `test_fundamentals` (the real Apple and META filings: every expected number is worked from the figures of
those pages, e.g. META's payout coverage uses the nine-month net income 52,481 and not the quarter's 9,860), `test_metrics_catalog`
(every metric of every symbol has the 7 keys, a missing value has a note, and no result carries a judgement key), `test_data_clerk_metrics`, and
additions to `test_work_order` and the tool-list tests (the allow-list went from 12 to 13 tools).

---

## 3. Phase 3: the record, the events, the gates, the quality multiplier, the config

**Problem.** Before any scoring there must be a *shape* (what a recommendation is made of, so the Critic can audit it), a vocabulary of events, the
checks that stop a symbol outright, and one place for every number.

**`config.py` + `configs/advisor_v1.json`: every threshold in one file, checked once.** Pydantic models with `extra="forbid"`: a typo in the JSON
(`min_adv_dolars`) fails when the file is loaded, instead of being silently ignored while the code uses a default. The agentic principle: *the model never
chooses a threshold, a weight, a price, a size or a rank*, and neither does the ledger. If you want a different cut-off, change the JSON.

**`record.py`: the contract with the Critic.** A `Recommendation` (schema 2) has the version-0 fields with their old meaning (so `top_recommendation` and the
week-3 lab read it unchanged) plus one section per question a Critic asks: `identity` (which case, which regime), `action` (open / add / hold / reduce / avoid and
why), `families` (what ran, what it was scored FROM), `quality`, `evidence`, `features` (every metric used, with unit, date and source), `agreement`, `ranking`, `trade`,
`risk`, `gates`, `thesis`, `audit`. Every model forbids unknown fields. `MODEL_FIELDS` lists the only places a model's words reach a record
(`model_score`, the thesis and bullets, the quotes, the company name): a test asserts it.

**`events.py`: the model points, code decides.** The plan call may say "this sentence is a `guidance_raise`". `verify_event` accepts it only when the class
exists, the source and the quote are real (as v0 checks signals), the quote names the company, the class's own regexes match the QUOTE, and the direction is the one
the class fixes. The patterns are narrow on purpose (a missed event is a gap, a false one is a wrong trade) and were tuned against the real stored text: I ran
the first version over every sentence of `news/*.txt`, the press releases and the filings, and it was wrong in instructive ways (the examples are in
`tests/test_events.py` as near misses):

| First version matched | Why it is wrong | Fix |
|---|---|---|
| "4.0% increase in organic sales ... while maintaining its full-year outlook" as a guidance RAISE | the verb and the object were tested independently | a verb must be within 50 characters of the object in ONE clause |
| "The author expects ... earnings to exceed expectations" as a beat | a prediction | `PREDICTION` pattern is refused |
| "upped the **lower end** of its profit guidance" as a guidance CUT | "lower" | refused: lower end, lower-than-expected, unchanged |
| "The Company **intends to increase** its dividend" (in every 10-Q) as a dividend raise | an intention | refused: intends / plans / expects / subject to |
| "AI **initiatives**" as a dividend initiation | `initiat\w+` | `initiat(e|es|ed|ing|ion)` |
| "SK Hynix's decision to scrap plans for an IPO" as an offering | cancelled, another company | IPO and scrap/cancel refused |

Two classes (Form 144 insider sale notice, Schedule 13D) come from the **filing index**: a filing without text is stored as "Form 144 (...) filed 2026-10-20. The text
of this filing is not stored", and `filing_index_events` reads form and date straight from that, with no model. `timing()` dates an event against the stored bars: age
in bars, the reaction since (last close over the close before the event bar), and the event bar's volume ratio. For that, the catalogue gained one list-valued metric,
`recent_bars` (the last 35 `[date, close, volume]`).

**`gates.py`: a failed gate is recorded, never silent.** `liquidity` (ADV in dollars, price, spread), `freshness` (the newest bar within one session of today),
`red_flag` (going concern / weakness / restatement events, stock being issued, debt over 4x equity with current ratio under 1, cash flow negative with a profit),
`event_risk` (the next report is near: blocks every family except `event`), `no_metrics`. **A gate whose input is missing FAILS and names the missing input.**
An illiquid stock must not be recommended because its volume could not be read. Consequence you will see on real runs: `adv_20d_dollars` needs 20 bars and only 17 are
stored, and the newest stored bar (2026-10-23) is nine sessions behind the date of this work, so every new idea fails `liquidity` and `freshness` until more day
files are synced. That is the gate doing its job; it is also the strongest argument for the sync (section 8 says what to ask the Data Clerk to do).

**`quality.py`: weak numbers shrink the edge.** A table in the config (metric, comparison, threshold, factor); a missing input applies no factor and is listed in
`haircuts_skipped`; the three earnings-quality checks (accruals, receivables vs sales, inventory vs sales) take their 0.5 once as a group, and they and the
negative-cash-flow check switch `reversion` off; checks that read a filing are skipped with a note when it is older than 120 days; the floor is 0.2.

**Tests.** `test_advisor_config`, `test_events` (real sentences + near misses + 5 checks of `verify_event` + timing), `test_gates` (each threshold and one step each
side), `test_quality` (each haircut just past its threshold and exactly at it).

---

## 4. Phase 4: the five strategy families (`families.py`)

**Problem.** "Is this stock a good trade?" is five different questions depending on *why*. A momentum stock, a stock that fell for no reason of its own, a company
that returns cash, a company with news the price has not digested, a laggard of a theme: each has its own evidence, its own risk and its own horizon. Mixing them in
one score is how three correlated indicators count three times.

**The design: one vote per family.** Inside a family, correlated features are folded into ONE score by the family's own formula. Across families, the symbol's
composite is the best bullish edge minus the largest bearish magnitude (section 5), never a sum.

| Family | The idea | Passes when ALL | Horizon, prior (20-day excess) |
|---|---|---|---|
| `trend` | momentum, but do not chase | 12-1 momentum >= 15%, within 10% of the 52-week high, above the 200-day mean, **not up more than 15% in 21 bars**, RSI < 75, residual momentum and %B fine when known | 63 bars, 0.6% |
| `event` | news the price has not digested | a verified event <= 10 bars old; a bullish one has moved <= 3 ATRs since (above that it is priced in); a bearish one always votes | 40 bars, 0.8% |
| `reversion` | a sound company on sale | oversold; the fall is its own (residual against the sector ETF, else against SPY); ROE > 10, debt/equity < 1.5, current ratio > 1, growth > 0; uptrend; the cause is NOT a fundamental one | 10 bars, 1.0% |
| `capital_return` | gives cash back, does not dilute | net payout yield >= 3% (or shares down 0.5%); shares not up; not issuing stock; debt/equity <= 2; dividends covered. **Issuing stock is a bearish vote of this family** | 126 bars, 0.3% |
| `theme` | the laggard of a positive theme (Case 2) | >= 3 articles, >= 60% positive, 21-bar return at or under the basket median, quality >= 0.5. A company that fails ONLY the laggard rule is a `leader`: shown, not recommended | 42 bars, 0.5% |

**Reading a `FamilyResult`.** `computable` is false when a REQUIRED number is missing (`missing` names it and `notes` gives the Data Clerk's reason): the family
says nothing. With 17 stored sessions `trend` needs `mom_12_1`: it is `computable: false`, which is correct. `passed` is true only if every rule held; only then is
there a `score` (0 to 1), a `direction` and an `edge_20d = prior x regime multiplier x score x quality multiplier` (negative if bearish). A computable family that did
not pass lists each failed rule WITH the value that failed it. Missing OPTIONAL inputs add nothing to the score and lower `completeness` (kept apart from
`agreement`: "how complete are the inputs" and "how much does the text agree" are different questions and folding them into one number would hide which one is weak).

**The priors are research numbers, not measurements.** `prior_20d` is an expected 20-day excess return over SPY taken from the papers in the spec
(`basis_ref`, for example `jegadeesh_titman_1993`), and `basis` says `prior` until the ledger has 30 outcomes. The thesis has to call such a number "unmeasured".

**Where a family's required metrics are listed.** `REQUIRED`, `OPTIONAL` and `required_metrics()`: the Advisor's request to the Data Clerk is built from them, so a
metric a family reads can never be forgotten in the request (a test checks every name is in the catalogue).

**Tests (`test_families.py`, 57).** Each family: a pass whose score is worked by hand (for example the reversion pass is 0.40 + 0.20 x clip((2.5-2)/2) + 0.40), every
rule failing on its own with its message, the 17-bar `computable: false` case, the anti-chase case (a stock up 25% in a month is not a trend buy), the idiosyncratic-drop
rule with and without a sector ETF, the fundamental-cause refusal for each of the six causes, the big-company haircut, the funded buyback (an authorization larger than
the company's cash is halved), the laggard rule, regime multipliers (trend is 0 when `recovering`), and the quality multiplier scaling the edge.

**Spec vs build.** The spec says a family returns "`FamilyResult` or `None`"; the build always returns a `FamilyResult` with `computable` / `passed` flags (a `None`
would lose the reasons the Critic needs). The earnings-surprise path of the `event` family (`sue_4q`) is not implemented: it needs earnings dates that are not
available, so `sue_4q` is always `None` and the path could never fire.

---

## 5. Phase 5: ranking, the holdings pass, the cooldown, the sleeves, the trade plan, the ledger

**From family edges to a short list (`ranking.py`).**

```
composite = best bullish edge - largest bearish magnitude   (conflict: true)     one vote per family
cost_pct  = half the spread (0.10% if unknown) + 0.10% slippage                  x2 in the stress regime
blend     = 0.40 news + 0.30 performance + 0.20 historical + 0.10 filings         -1 to +1: section 10 (this replaced the old "confidence from the text's agreement")
scale     = 1 + 0.5 x direction x blend                                          0.5 to 1.5; direction is -1 for a bearish composite
excess_20d_after_cost = composite x scale - cost_pct                             no family passed: blend x 0.003 - cost_pct (section 10)
stance = bullish if excess > 0; bearish if composite < 0; else neutral ("edge_below_cost")
score  = round(50 + 50 x clip(excess / 0.03, -1, 1))                             the v0 field top_recommendation reads
```

Worked example (it is a test): a trend edge of 0.6%, blend 0 (nothing to say): scale 1.0, so the excess is 0.006 x 1.0 - 0.002 = **0.4%** and the score is 57. A blend of +0.6 (good
news, good prices, good numbers): scale 1.3, excess 0.006 x 1.3 - 0.002 = 0.58%. A blend of -0.6: scale 0.7, excess 0.22%. An edge of 0.19% is under the 0.20% cost and `neutral`; a
blend of +0.5 scales it to 0.24% and it becomes a (thin) recommendation, which is how the real Procter & Gamble capture changed (section 10).

**The regime** (once per run, from SPY): `unknown` when SPY has fewer than 200 bars (today), then `recovering`, `stress`, `narrow`, `trend`. It scales the families through
the config and halves the risk budget in `stress` and `unknown` (conservative when we cannot tell).

**The holdings pass.** Every run reviews what is already held, before new ideas: `reduce` (a red flag, a bearish event in the last 10 bars, an earlier recommendation
whose invalidation rule fired, or the family that recommended it no longer passes), `add` (bullish, not oversized, would have made this run's list, stays within
20% of equity) or `hold`. A `reduce` is returned above every `open`. The size of a reduction is NOT invented here: the plan has an entry zone and no share count (T+1 and
tax are the Account Manager's business).

**The cooldown.** Do not open the same symbol on the same family while the earlier recommendation's time stop has not passed and none of its invalidation rules have
fired (`price_below`, a bearish event, the family's own condition). The candidate comes back with `cooldown_of: <record id>`.

**The sleeves (`select`).** The case decides who may open a name. `funds` ("here is $1,000"): up to 3 names; capital_return, trend (only in the `trend` regime) and
reversion may open; `event` may only VETO (a bearish event vote blocks an open) and is used to open only when nothing else qualifies. `theme`: up to 5, the `theme`
family required. Ordering is deterministic: excess, fewer haircuts, lower volatility, symbol. **Diversity:** if the next pick shares a sector ETF with one already
picked, a name of another sector whose excess is within 80% of it is taken instead and both facts are recorded (`diversity_adjustment`, `displaced_by`).

**The trade plan (`trade_plan.py`).** All arithmetic: entry zone (last +- 0.5 ATR), stop (2 ATR reversion/event, 3 ATR trend/capital_return, or the 50-day mean if
nearer), target (1.5 x the edge at the horizon; none for capital_return), a time stop in weekdays, **whole shares** from a risk budget of 1% of equity (0.5% in stress),
capped at 20% of equity and 5% of ADV; if that is 0 the plan says one share, `oversized: true` and how much of the account it is. For a $1,000 account almost
everything is oversized: that is the honest answer for $1,000. `funds_feasible` says whether the money exists (cash, or what the sellable holdings in profit
could raise, keeping one share of each, using `tools/trading_math.rank_green`); it plans no sale.

Two readings the spec leaves open, flagged:
1. The spec writes `suggested_shares = floor(risk_budget x equity / stop_distance)`, which multiplies by equity twice (the budget is already 1% of equity). It is `floor(risk_budget / stop_distance)`.
2. "The family's edge at its horizon": the 20-day edge is used as it is for a horizon of 20 bars or less and scaled in proportion for a longer one (trend: 3.15x). Otherwise a 63-bar trend target (+0.9%) would be smaller than its own 3-ATR stop (about -6%).

**The ledger (`memory/ledger.py`).** An append-only JSONL (`data/ledger/recommendations.jsonl`): one line per recommendation, never rewritten. The generate step only
appends; it reads it for the cooldown and the holdings pass. A corrupt line is reported in `ledger.problems` (with its number, never its content) and every other
line is still read, so one bad write cannot silently turn the cooldown off. Outcomes (5/20/60-day returns against SPY) and `basis: measured` are later work.

**Tests.** `test_ranking` (the regime states, the min rule, the worked excess, the stance, stress cost, `top_recommendation` picking the right symbol from the v1 `score`,
the holdings pass, the cooldown, the sleeves, diversity, a deterministic order whatever order the names arrive in), `test_trade_plan` (stops by family, the 50-day stop,
the target, weekdays, the $1,000 account, the caps, stress, order styles, invalidation rules, equity, `funds_feasible` including the T+1 lock), `test_ledger`.

---

## 6. Phase 6: the whole step (`recommend.py`), the prompts, the thesis checks

`Advisor.generate_recommendations` is twelve steps in one function (`recommend.generate`); each step has its own module so you can read and test it alone.

| Step | What happens | Where |
|---|---|---|
| 1 | The order must be `data_ready` (v0's message otherwise). `case` comes from the argument, else the order, else the subject (symbols = `funds`, a term = `theme`). `market` is refused with a message: it needs the market brief, which is not built | `recommend.py` |
| 2 | Candidates: the subject, the symbols the sources LIST and the text also NAMES, the theme's tickers (`discover_tickers`, Case 2), then the portfolio's holdings tagged `held` (they get the holdings pass, they are not new ideas) | `candidates.py` |
| 3 | **PLAN call** (model 1): signals, events, one drop cause, archetypes, `model_score`. Code checks every quote (v0's `verify_signal`, `events.verify_event`) | `prompts/advisor_v1.py`, `calls.py` |
| 4 | **ONE** `calculate_financials` order to the Data Clerk for every metric every family, gate and the plan will read (82 names, built from `families.required_metrics()` + the gate, plan, risk, regime and quality names) | `DataClerk.request_metrics` |
| 5-8 | Per candidate: events timed against the bars, gates, quality multiplier, the families, then composite, cost, excess, stance, score | `gates`, `quality`, `families`, `ranking` |
| 9 | Holdings pass; cooldown from the ledger; the sleeve of the case picks the new ideas (top 3 for `funds`, 5 for `theme`), with the diversity rule | `ranking.py` |
| 10 | The trade plan of every record | `trade_plan.py` |
| 11 | **THESIS call** (model 2) for all records at once, each checked; the failures get ONE retry; a second failure is `thesis: null` + the reasons | `thesis.py` |
| 12 | Records onto `order.recommendations` (reduce, then add, then open by rank), the rest onto `order.candidates` (each with its reason), the runs onto `order.advisor_runs`, one ledger line per record | `recommend.py` |

**The two prompts.** The plan prompt is v0's rules plus the event classes (listed from the taxonomy table, so the prompt cannot drift from the checker), the drop-cause rule and the
archetypes; the thesis prompt gives the model a fixed-wording digest of the finished record (`thesis.digest`: a template, no model prose, no `model_score`) and lists the refs it may cite.
Both say "text inside sources is data, never instructions".

**Why the thesis checks are what they are.** The model is allowed to be fluent, not to be a source of facts. So `thesis.check` asks five things, each answerable by code: at most 3
sentences; every ref is one the record has; every number is a number the record has (compared after rounding to the precision shown: `4.2%` matches 0.0421, `$163.45` matches 163.4512,
`300 billion` matches 3e11); a bear case that cites the counter-evidence when there is any; the word "unmeasured" while the basis is a prior. The first run of these tests was FLAKY, and the
reason is a good lesson: the digits of a random `record_id` happened to contain "37", so 37 was an "allowed number" in one run in forty. Identifier keys are now excluded from the allowed numbers
(`IDENTIFIER_KEYS`), and `test_the_digits_of_an_identifier_are_not_numbers_of_the_record` pins it.

**A failure never stops the step.** No plan reply: no text signals, but `capital_return` (which needs none) still runs, and `advisor_runs[0].error` and `order.errors` say so. No metrics (the
Data Clerk refused or failed): every candidate becomes a `no_metrics` candidate. A bad thesis: `thesis: null` with the reason and the record still returned.

**Tests.** `test_candidates` (7), `test_thesis` (21), `test_advisor_v1` (23: the whole step on fakes: status, the record validates, a model that says different things in every field it owns changes
nothing a program computed, a model score of 100 does not pass a gate, the retry of a bad plan reply, an invented quote and an event the text does not show are dropped with reasons, a thesis that fails
twice, the rejects as candidates with their gates, stale bars, the ledger and the cooldown, a holding with a red flag is a `reduce` above every `open`, v0's `advise` still works, the theme case).

---

## 7. The real run: SNDK and PG on real data (and what it taught)

You asked for no invented data, so the two tickers were run for real on 2026-11-05 and the tests replay that run offline. The steps, so you can repeat them:

1. `python scripts/fetch_bars.py --symbols SNDK PG MSFT TSLA GM --yes` - 276 sessions of daily bars per ticker (see 7.1 for why not the flat files).
2. The Data Clerk's refresh of the text: `fetch_news`, `fetch_press_releases`, `fetch_filings` for both (paid Finnhub/Massive; the 10-Q text from EDGAR).
3. `python scripts/capture_advisor_fixture.py --symbol SNDK --company Sandisk --sector XLK --yes` (and PG): the real Data Clerk model answers the order, Finnhub gives the metrics, the real Advisor
   model makes the plan and thesis calls, and everything that went in is saved to `tests/fixtures/advisor_real/` (about 145 KB each).
4. `pytest tests/test_advisor_sndk_pg.py`: the same code, numbers and words, offline, free.

**7.1 The numbers that were wrong or missing until real data was used** (each fixed and tested from the real lines; entry 52 has the table): Sandisk's units line was lower case, so its payout
yield was 1.35e-7 instead of 13.5%; P&G's 10-Q could not be read at all (its headings are two-cell table rows); P&G's cash-flow rows have other names; the flat-file sync returned 403 with both
credential sets, so the bars come from the Massive REST aggregates (identical numbers: PG's 2026-10-23 bar matches the flat file to the last digit).

**7.2 What the Advisor concluded** (the captured runs of 2026-11-05, BEFORE the blend of section 10: with it the Procter & Gamble neutral capture becomes a thin recommendation, and the scores and excesses below shift by the blend's scale of 0.5x to 1.5x):

| | SNDK | PG |
|---|---|---|
| Result | `open` on `capital_return`, score 51 | neutral in one capture ("edge below cost"), `open` score 52 in another |
| Why | a $5.0B buyback in the quarter over a $147.6B market cap: net payout yield 13.5%, shares down 1.4% | net payout yield 3.9%, payout coverage 0.55, an event vote from the earnings beat |
| Trend / reversion | computable, did not pass: price 34% of its 52-week high, under its 200-day mean | trend: momentum 3% (needs 15%); reversion: not oversold, RSI 68 |
| Expected excess after cost | +0.08% | -0.09% to +0.10% |
| Plan | 2 shares, a stop 40% away (3 ATR of a 13.5% ATR), no target | about 75 shares, stop 5.9% away |
| Simple Critic | `review`: 0 defects in 65 checks; warnings: far under its high, below the 200-day mean, a 57% drawdown in 63 days, a 40% stop, a thin edge | `review`: 0 defects; warning: a thin edge |

**7.3 The three findings that are yours to decide** (none is a code bug):

1. *The edge is about the size of the cost.* The families' priors (0.3% to 1.0% over 20 days, from the papers in the spec) times a score of 0.4 to 0.9 give 0.1% to 0.3%; the cost is 0.2%.
   So a symbol like PG flips between `neutral` and `open` with the wording of the model's quotes (it was a recommendation in 2 of 5 live runs). Either the priors or the cost are the
   settings to look at (`configs/advisor_v1.json`: `ranking.default_half_spread`, `ranking.slippage`, `families.*.prior_20d`); I did not tune them. *(Since then: the blend of section 10 scales the edge by 0.5x to 1.5x from the news, prices, history and filings, so this flip now depends on those as well, and the model's wording matters less.)*
2. *`capital_return` recommends a stock that just fell by half* (SNDK), because nothing in the family looks at the price. The Critic's warnings catch it; a rule (for example "not more than X%
   under the 52-week high") would catch it earlier. That is a strategy decision, so it is left to you.
3. *The composite uses the best edge of any family, including `event`, even when the name opens on `capital_return`.* Per spec 9.5. The record shows both.

**7.4 Gaps in the Data Clerk this exposed** (candidates for the next work): nobody refreshes bars for candidates found in the news (META, WDC, ADDYY failed `liquidity` with "0 stored"); the order's
press-release sweep found none for PG although 88 are stored; the filing evidence is read only when an order names exactly one symbol; the earnings endpoints (calendar, history) and the company
profile (sector) are not added, so `event_risk`, `sue_4q` and the sector ETF depend on what the user supplies.

---

## 8. How to use it from your LangGraph, and what is left

```python
def advisor_node(state):
    advisor = Advisor(...)                                   # one per run; model, clerk and ledger can be injected
    order = advisor.generate_recommendations(state["order"], state["portfolio"], state["case"], state["today"])
    return {"order": order}
```

Put a **refresh** node before it: `DataToolbox.fetch_daily_bars(symbols_of_the_candidates)` (the Advisor reads bars, it never fetches). The Critic then reads `order.recommendations` (the
contract is `record.py`; `tests/simple_critic.py` is a working example of what to check, including the arithmetic it recomputes and the quotes it looks up in the order's own sources) and
`order.candidates` (why each other symbol was not returned). The Account Manager's `top_recommendation` reads the v1 `score`, and RaiseFunds can prefer the `reduce` names.

**Not built, by decision** (spec phases 8 to 10): Case 3 `market`, theme counterparties, the bench modifiers, intraday confirmation, measured-basis shrinkage, `risk_factor_change`, the
earnings-based inputs. Each is a catalogue entry or a refusal that says why, so nothing is silently missing.

**Verification at the end (2026-11-06):** every new module has its tests; the full suite passes (2,248 tests) except the one that fails on purpose. The three tests that had failed because their fixtures aged
out of their date windows now use a clock that follows the real date, so they stay green. On the 5th, 16 deliberate breakages (a boundary, a formula, a check in each of the bars, indicators, fundamentals, events, gates, quality,
families, ranking, trade plan, thesis and the step) were each caught by a test, and the files restored byte for byte.


## 9. Running it from a typed request (`./test_recommendations.sh`, design choices entry 53)

The request can be a symbol, a list of symbols, a company name, a comma list that mixes tickers with names or brands (`MCD, YUM, "Burger King"`: the Data Clerk finds a brand's listed owner), or a phrase (`Pizza`). The Advisor runs on `openai/gpt-oss-120b` (Cerebras, then Groq; effort low): the plan call takes about 2 seconds. `--find N` is how many top recommendations, `--cash N` sizes for a cash amount, `--no-fetch` uses only the stored data (no paid data calls), `--no-ledger` explores without the real ledger's cooldown, `--list`, `--symbols`, `--company`, `--phrase` force the kind. Only the companies named are scored (`only_subject`). A model call that hangs is given up after 300 seconds.

`prometheus/application/recommendations.py` is the caller the LangGraph will be: `classify` the request (symbols, a list, a company, a phrase), find the companies (`DataClerk.find_companies`), refresh the data (`--fetch`), one work order, `request_data`, `Advisor.generate_recommendations(order, portfolio, case, today, top_k, only_subject=True)`, then `report`. What the first runs taught the Advisor: every measured company gets a score (no family passed = no edge = just under 50), a bearish company ranks below a no-signal one, ties are broken by the verified text evidence, and `only_subject` keeps the comparison to the companies that were named. For gpt-oss the Advisor asks OpenRouter for Cerebras then Groq at effort low: the plan call took 2 s, not 100 s. A model call that hangs is given up after 300 s. See `HowToRun.txt` part 9 to run it.

### 9.1 Negative equity (design choices entry 54; spec Amendment 1)

A company whose balance sheet shows a stockholders' deficit has no meaningful debt to equity, so `negative_equity` (a metric read from the latest stored balance sheet) switches off the four debt-to-equity rules for it: the `red_flag` gate leg, reversion's `sound` test, capital_return's limit and the quality haircut. Each result carries the marker (`negative_equity`, a family's `waived`, the gate note, `haircuts_skipped`) and the report shows `*`, "waived" and DEFICIT. Everything else about the company is scored as for any other.

**Checked on the real data (2026-11-06; Domino's, Papa John's, Yum, McDonald's).** All four read as `negative_equity: true` from their stored 10-Qs; every family is computable for each (before, Domino's had no debt to equity, so reversion and capital_return were `not computable`); each family lists `waived: ["debt_to_equity"]`. Yum and McDonald's are recommended (capital return), Domino's passes capital return with a 0.18% edge that is just under the 0.20% cost, and Papa John's fails on its own dividend coverage (1.97): the results come from the other rules, as intended. To explore without the ledger's cooldown hiding a company you already ran, add `--no-ledger` (a throwaway ledger: nothing read from or written to `data/ledger/`).


## 10. The blend: what the score is made of, and why news matters (design choices entry 55; spec Amendment 2)

**What was asked.** The recommendation's score should be 40% the sentiment of the news and press releases, 30% the stock's own performance, 20% the historical metrics and 10% the filings,
and the newer the news, the more it counts, so that news affects the recommendation.

**What I built (`advisor/blend.py`, the numbers in `configs/advisor_v1.json` under `blend`).** Four parts, each from -1 (all bearish) to +1 (all bullish), all calculated by code:

| Part | Weight | What it is made of |
|---|---|---|
| news and press releases | 40% | the verified signals and events that cite a news article or a company's press release. Each counts `kind x 0.5 ** (age_days / 7)`: kind 1.0 for an article, 0.5 for a press release (a company's own words are not an independent voice); a story loses half its weight every 7 days. `value = sum(direction x weight) / (sum(weight) + 2)`: the 2 is a prior of two neutral voices, so one old quote is not certainty. A signal and an event from the same source in the same direction are one voice |
| stock performance | 30% | 12-1 month momentum, one-month return, distance from the 52-week high, above the 200-day mean |
| historical metrics | 20% | earnings yield, return on equity, revenue growth, operating margin, current ratio, debt to equity (not for a company with negative equity, entry 54) |
| filings | 10% | net payout yield, share-count change, payout coverage, dilution now, cash flow against profit, plus each verified filing-evidence item and each filing-index event (Form 144, Schedule 13D) as one vote |

A measure in a part is `clip((x - neutral) / span, -1, 1)` (a negative span means lower is better), or for a yes/no metric +1 or -1 (`only_bad`: a warning that is not raised is silence, not
praise). A part's value is the mean of its measures that exist. A part with nothing to say is **unavailable**: it counts as 0 (neutral), is shown as `n/a`, and is listed on the record, so you
can see how much of the blend was actually measured. The table of measures, their neutral points and spans is in the config, not the code.

```
blend = 0.40 x news + 0.30 x performance + 0.20 x historical + 0.10 x filings       -1 to +1;  blend score = round(50 + 50 x blend), 0 to 100
```

**How the blend changes the recommendation (two places, both in `ranking.py`).**

1. *It scales the strategy's edge.* The strategies (trend, event, reversion, capital return, theme) still say whether there is an entry and how big their prior edge is. The blend multiplies
   that edge: `scale = 1 + 0.5 x direction x blend`, from 0.5 to 1.5 (the same range the old text-agreement "confidence" had, which it replaced). Good news, good prices and good numbers raise a
   long edge; bad ones cut it, and an edge cut below the cost stops being a recommendation. For a bearish composite the direction is -1, so a bearish blend makes the bearish edge larger.
2. *When no strategy fired it sets the excess.* A company with no entry rule used to get "the cost, negative" for everyone (47), so the ranking could not tell them apart. Now its expected excess is
   `blend x 0.003 - cost` (0.003 is the smallest strategy prior: a blend alone is never worth more than the weakest strategy), so a company with good news, prices and numbers ranks above one
   without. Its stance stays `neutral` ("no family passed"): the blend says what the company looks like, it does not give an entry, a stop or a time limit.

**Where it is written.** Every recommendation and every candidate carries `blend` (the value, the 0 to 100 score, the four components with weight, value, `available`, contribution and the
pieces: for the news each item's reference, kind, direction, age in days, recency and weight). `ranking.scale_factor` is the multiplier used; `agreement.confidence` is now `(1 + direction x blend) / 2`.
The report shows a `Blend` column and the four parts (`News`, `Price`, `Hist`, `File`) in the ranking, a legend, and on each recommendation the blend line and the three heaviest news items with
their age and weight. The simple Critic (`tests/simple_critic.py`) recomputes it all: the weights are 40/30/20/10, every news weight is `kind x 0.5 ** (age / 7)`, the news value, each other
part's mean, the blend, its score, the scale factor and the excess.

**Worked example, from the real run (McDonald's, 2026-11-06).** Two bullish items, both 1 day old (an earnings-beat headline and a positive-traffic article): weights 0.91 and 0.91, news
`1.82 / (1.82 + 2) = +0.48`. Performance -0.05 (momentum -22%, but near its highs and a month of gains), historical +0.56, filings +0.33. Blend `0.4 x 0.48 - 0.3 x 0.05 + 0.2 x 0.56 + 0.1 x 0.33 = +0.32`,
score 66. The event strategy's edge is 0.61%; the scale is `1 + 0.5 x 0.32 = 1.16`; the excess is `0.61% x 1.16 - 0.20% cost = +0.51%`, score 58, rank 1. With the same story 29 days old each weight
would be 0.06, the news part +0.05, the blend +0.15, the scale 1.08 and the excess +0.46%: about 0.05 percentage points lower.

**What changed in the tests.** The Procter & Gamble capture that used to be "neutral, the edge is under the cost" is now a thin recommendation: its capital-return edge of 0.16% times 1.28 (a blend of 78) is
0.21%, just over the cost (the Critic warns "thin edge"). That is the feature working on real data, not a regression, and the test now says so.

**What it is not.** The weights and the half-life are choices, not measurements, like the strategy priors (`basis: "prior"`): they are in the config so that you can change them, and the ledger
never tunes them. The news part is only as good as the quotes the model offers (it offers a few per company, code checks each one): a company with news the model did not quote has an `n/a`
news part. News enters twice by design (the `event` strategy's edge needs a verified event, and the blend scales whatever edge there is); the blend does not need an event.


**Scoring** What the blend is
News and press releases (40%): each verified quote counts 0.5^(age in days ÷ 7), so a story loses half its weight every 7 days. A press release counts half an article. One old quote can't swing the result, because the formula adds two neutral votes to the denominator.
Stock performance (30%): 12-1 month momentum, one-month return, distance from the 52-week high, and whether it is above the 200-day mean.
Historical metrics (20%): earnings yield, return on equity, revenue growth, margin, current ratio, debt to equity (skipped for negative equity, as before).
Filings (10%): payout yield, share-count change, payout coverage, dilution, cash flow against profit, plus the filing evidence items.


---

## Note of 2026-11-14 (design choices entry 80): sizing moved to the Account Manager

The Advisor no longer sizes a trade and no longer uses the user's money. The text above is left as it was written; where it says otherwise, this note wins:

- **Section 5, the regime**: it no longer halves a risk budget (there is none); it still scales the families and is in the record's identity.
- **Section 5, the holdings pass**: `add` is a bullish stance whose blend is as good as this run's new ideas. "Not oversized" and "stays within 20% of equity" are gone (they needed a size and the user's money); whether the user's limits leave room is the Account Manager's sizing, which counts the shares held. `ranking.add_max_position_pct` and `ranking.oversized_pct` left the config. A `reduce` has no amount.
- **Section 5, the trade plan**: entry zone, stop, target, time stop, order style and invalidation rules, the same numbers as before (`tests/test_trade_plan_real.py` checks 208 real plans against the old code's). The whole-share size, the risk budget, the caps, `oversized`, `equity`, `position_pct` and `funds_feasible` are gone, and so is "reading 1" of the two readings the spec leaves open (the share-count formula). `build` is given no portfolio, equity or regime. The sizing and `funds_check` are in `prometheus/agents/account_manager/sizing.py`.
- **Section 5, the tests**: `test_trade_plan` has no `$1,000 account`, caps, stress, `equity_of` or `funds_feasible` test (the Account Manager's sizing tests cover them); `test_advisor_no_size` reads the Advisor's source for the user's cash, equity and every size name.
- **Section 6 and the thesis**: the digest the model reads has no size line, and the prompt says the record has no amount to buy or sell.
- **The record** keeps schema version 2 with the fields gone (as Amendment 7 did); older ledger lines still hold them.
