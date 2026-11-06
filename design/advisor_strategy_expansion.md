# Advisor strategy expansion: review of `advisor_strategy_handoff.md`

Jacob Pyrett, 2026-11-05. A second-pass design for the coding model. It does not replace
`design/advisor_strategy_handoff.md`. It says what to change in that spec, which strategies to add,
and which ideas to leave out. Same rules as the handoff: the Data Clerk returns facts, code scores,
the model proposes quotes and wording, paid fetches need a yes, tests use fakes.

Read the handoff first. Section numbers below ("handoff 5.2", "handoff 6") point at it.

---

## 1. What the handoff gets right

Keep these. They are the part worth building first.

- The model never writes the score. Quotes are checked against the source. Metrics carry `value`, `as_of`, `source`, `inputs_summary`.
- The anti-chase rule on 12-minus-1 momentum (handoff 5.2). A stock that is up hard in the last month is a worse buy.
- Capital return from filings you already parse (handoff 5.4), with dilution as counter-evidence.
- The underreaction test on events (handoff 5.5): a good event the price has not absorbed outranks one that already gapped.
- Reversal only when the company is sound and the cause is non-fundamental (handoff 5.3).
- Gates, ATR sizing, invalidation rules the Critic can re-evaluate, an append-only outcome ledger, and a walk-forward test against "buy whatever just ran."
- One `calculate_financials` order for the whole candidate list, answered by code with no model loop.

---

## 2. Changes to the handoff before any new strategy

These are the tweaks that keep the first design from mixing unlike trades or inventing precision it does not have.

### 2.1 Rank inside one horizon, on the excess left after cost

Handoff section 6 adds a 5-day reversal's expected return to a 12-month buyback's expected return. Those are different units. A 1% bounce and a 3% annual payout are not comparable, and summing them double-counts names that trip several correlated rules.

Change the composite:

- Put each strategy in a **family**. One family, one vote. Proposed families: `event` (5.1 and 5.5), `trend` (5.2 and the residual version in section 3.1), `reversion` (5.3), `capital_return` (5.4), `theme` (5.7). Inside a family, keep the stronger result and attach the other as support.
- Convert that family's edge to a **20-trading-day excess over SPY**, using the horizon it actually has. A 60-day prior is scaled by 20/60 for ranking only. The record still shows the original horizon, the unscaled prior, and `rank_horizon_days: 20`.
- Subtract estimated cost before ranking. Cost in dollars = half the spread (or 10 bps if no spread) + 10 bps slippage, times notional. The ranked number is `excess_20d_after_cost`. A setup whose edge is smaller than its cost is `stance: neutral` with reason `edge_below_cost`.
- Keep literature percentages as the *reason the family exists*. The number on the record is `basis: "prior"` until the ledger has 30 outcomes in that family and regime (handoff section 10). Label it `unmeasured` in the thesis so a 1989 paper is not presented as this month's forecast.

### 2.2 Split completeness, agreement, and sample size

Handoff section 6 folds "how many inputs were present" into `confidence` and then multiplies by expected return. A missing input should already have returned no result. Confidence should be three fields the Critic can read separately:

| Field | Meaning |
|---|---|
| `completeness` | share of the family's optional inputs that were present and fresh. Required inputs missing means no result. |
| `agreement` | verified quotes for the direction, over verified quotes both ways, smoothed the way Advisor v0 smooths `(n+1)/(n+2)`. Company's own press releases and filings count at half weight (independent news counts at full weight). |
| `evidence_n` | how many distinct stories and filing items were counted. |

`confidence` for the rank is `agreement`, shrunk toward 0.5 by sample size: `(evidence_n * agreement + 4 * 0.5) / (evidence_n + 4)`. Four is a prior of "we have seen little." A single syndicated headline cannot read as certainty.

### 2.3 Ranks need a real cross-section

"Top quintile of the candidate set" is meaningless when the set is five names from one news search. Use a **threshold** whenever the set has fewer than 30 names (`sue > 1`, `mom_12_1 > 0.15`, `rsi_14 < 30`). Use a **cross-sectional rank** only in Case 3, over the index universe. The record's `scoring_basis` already allows this; make the small-set path the default for Case 1 and Case 2.

### 2.4 Quality is a multiplier, not its own strategy

Handoff 5.6 (cheap and profitable) is a slow, weak standalone bet for a small account. Build it as a **quality score in code**, from 0 to 1, and multiply the family's 20-day excess by it. Inputs you can compute now: positive operating margin, ROE above 8%, revenue growth above 0, debt-to-equity under 2, current ratio above 1, share count not rising, no dilution item with a current-period dollar figure. Missing inputs lower completeness. They do not zero the score unless a hard gate in handoff section 8 fails.

Gross margin and ROE from Finnhub stay the fresh numbers (design choices entry 8: Finnhub's precomputed fields, not a hand-built statement history). Filing lines are the cross-check, with the filing date in `as_of`.

### 2.5 One small account cannot be a diversified book

Case 1 is about $1,000 and whole shares. Handoff section 7's 35% sector cap and correlation penalty describe a 15-name portfolio. On this account the honest output is **one to three names**, each tagged `concentrated: true`, with the sector and the beta written out so the user sees that three tech names are one bet.

Change top-k by case (section 5 below) and keep the diversity penalty only when it actually changes the pick: if the second name is the same sector as the first, prefer the next name in a different sector when its after-cost excess is within 20% of the leader. Record who was skipped and why. That is the whole diversity rule until the account is large enough that a fifth name fits inside the risk budget.

### 2.6 Every run reviews what is already held

A bot that only emits buys will add to a name it already recommended yesterday, and it will keep a holding whose thesis broke. Add a **holdings pass** on every run, before new ideas:

| Action | When code assigns it |
|---|---|
| `hold` | The holding still passes the family's rules and no invalidation rule has fired. |
| `add` | It still passes, it is not `oversized`, and this run's excess is in the top of the new ideas. Default is `hold`, not `add`. |
| `reduce` | A red flag, a `guidance_cut` / `dilution_offering` / `restatement`, or the invalidation rule fired. This is the name RaiseFunds should prefer to sell when it must raise cash. |
| `avoid` | A new idea that failed a gate. Returned so the Critic can see the refusal. |
| `open` | A new name, not held. |

Cooldown: suppress a fresh `open` for the same symbol and family while an earlier recommendation is inside its horizon and no invalidation has fired. The ledger is what makes this check possible. Without it the desk re-buys the same headline every morning.

### 2.7 The Critic receives the rejects

Handoff section 9 sends the winners. The Critic's job (`find_other_candidates`, "is this a deal") needs the names that almost won. On the work order, next to `recommendations`, put `candidates`: every symbol that reached scoring, including gate failures and the runner-up, each with the same feature block, the gate that failed, and `rank`. Cap it (30). The thesis is written only for the top-k. The Critic can then say "MU ranked above SNDK because SNDK's drop was the whole sector" with the numbers on the order.

### 2.8 Regime from breadth, not only from SPY's moving average

SPY above its 200-day average can hide a market where most stocks are falling. From the same daily bars, compute `breadth_200`: the share of the universe (or of the names whose bars you have loaded) whose close is above their own 200-day average. Regime becomes:

| State | Rule | What the ranker does |
|---|---|---|
| `trend` | SPY above its 200-day and `breadth_200` ≥ 0.5 | All families eligible. |
| `narrow` | SPY above its 200-day and `breadth_200` < 0.4 | Halve the trend family's weight. Leadership is a few names; do not buy laggards just because the index is up. |
| `stress` | SPY 21-day annualized vol > 25%, or SPY below its 200-day | Halve size. Raise the after-cost threshold. |
| `recovering` | SPY drawdown over 63 days > 10% and SPY up more than 5% over 21 days | Turn the trend family off for this run (momentum-crash window). Reversion and capital return stay on. |

Write the state and `breadth_200` on every record.

### 2.9 Stops must respect the next earnings gap

A 2× ATR stop does not contain an earnings gap. Once earnings dates exist, compute `earnings_move_abs`: the median absolute 1-day return on the last four report dates (from the bars, on those dates). If the next report falls inside the holding horizon, size off `max(stop distance, earnings_move_abs × price)` and set `gap_risk: true`. For the event family, the report *is* the setup, so this applies to the *next* report after the one just released. Until the calendar endpoint is approved, estimate the next date as the last earnings press release plus 91 days and say so in `note`.

### 2.10 Order style is part of the recommendation

The entry zone in the handoff is a price band. Add `order_style`, chosen by code from the family:

| Family | Order |
|---|---|
| `reversion`, `event` with a small reaction so far | Limit at last price or better. Do not chase above the zone. |
| `trend` | Marketable limit at last + 0.25 × ATR, only if still above the 200-day. |
| `capital_return` | Limit. The edge is slow; paying the spread is the whole cost. |
| Any name with spread > 10 bps | Limit only. A market order gives the edge away. |

The Trader later needs this. The Critic can reject a marketable order that has already left the zone.

### 2.11 A short list of event classes, matched to text you actually store

Handoff 5.5 lists FDA approvals, index adds, and delistings. Those rarely appear in the stored news and press releases, so the regexes would never get a sample. Ship classes you can point at a real record today, and park the rest.

Ship: `guidance_raise`, `guidance_cut`, `earnings_beat`, `earnings_miss`, `buyback_authorization`, `dividend_increase`, `dividend_cut`, `dilution_offering`, `ceo_departure_unplanned`, `activist_stake_13d` (the filing index already stores Schedule 13D), `material_weakness`, `restatement`, `going_concern`.

Park until a source is named: `fda_approval`, `index_inclusion`, `major_contract_award`, `delisting_notice`.

### 2.12 Fixes inside the existing families

- **5.1 PEAD.** Score on absolute SUE above 1.0 in a small set (section 2.3). Down-weight when this is the fourth positive surprise in a row and each surprise is smaller than the last: the drift is mostly spent. Skip when the announcement-day move already exceeds 3 × ATR (the underreaction test). PEAD is weaker in the largest names; multiply the excess by 0.5 when market cap is above $200B.
- **5.2 momentum.** Require the strength to be the stock's own, not the sector's (section 3.1). Keep the skip-month and the "not in the last-month top decile" rule. Treat 52-week-high proximity as a second feature of this same family, not an extra vote.
- **5.3 reversal.** Require the drop to be idiosyncratic: the stock's 5-day return minus its sector's 5-day return is at most −5%, and the sector itself is not down more than 3%. A sector air-pocket is beta. Buying it is not a reversal edge.
- **5.4 capital return.** Add two checks from numbers already on the filing evidence. **Coverage:** dividends paid this period are below net income, and buybacks plus dividends are below cash from operations once that row is parsed (section 3.2). A yield above 100% of earnings is a cut waiting to happen. **Stock-based pay:** share-based compensation above 15% of revenue haircuts the quality multiplier by 0.7. Finnhub's margins do not subtract that pay. **Cash versus the announcement:** when the press release states a dollar authorization and the balance sheet has cash, and cash is below the authorization, the event is an announcement rather than a funded program; cut that event's excess in half.
- **5.7 themes.** Measure "already priced" against the theme basket (median 21-day return of the names `discover` returned), not only against the stock's own history. Show the crowded leader with its return, and rank the members that lagged the basket and still pass quality.

### 2.13 Bars, splits, and the backtest

- Confirm the day aggregates are split-adjusted before any 12-month return. An unadjusted split looks like a crash or a triple. If a split lands inside the window and the series is unadjusted, set the long-window returns to `None` and say so.
- Price returns ignore dividends. For names with a yield above 3%, momentum is understated. Once a dividend series is available, use total return for `mom_12_1` only.
- A constituent file built from today's S&P 500, used on last year's bars, omits the names that left. The backtest report should say "survivorship: current constituents." That is acceptable for a capstone if it is printed on the result. It is not acceptable to hide.

---

## 3. Strategies and signals to add

Four of these use data the desk already stores or can compute from the day aggregates. Two need a new Finnhub or Polygon call and wait for a yes. Each one is a family feature or a modifier. None of them is a second score the model gets to write.

### 3.1 Residual strength versus the sector (changes trend and reversion)

**Edge.** A stock that beat its sector over the last 12 months, skipping the last month, keeps beating it more cleanly than a stock that merely rose with the market. The same residual, over 5 days, is the reversal test: down hard while the sector was flat.

**Data.** Daily bars for the stock and for a sector ETF. Sector from Finnhub `stock/profile2` (`finnhubIndustry`), mapped by a fixed table in `configs/` to one ETF (XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLRE, XLU, XLC). Those ETFs are in the same day-aggregate files. No new vendor. The profile call is NEEDS-YES; until it exists, let the user or the work order pass `sector` and skip residual when it is absent.

**Code.** `residual_mom_12_1` = stock return from 252 to 21 bars ago, minus the ETF's return over the same window. `residual_5d` = the same over 5 bars. Trend family requires `residual_mom_12_1 > 0` and the handoff 5.2 price rules. Reversion family requires `residual_5d ≤ −0.05` and the sector's own 5-day return above −3%.

**Failure.** The ETF map is coarse (one bucket for all of tech). Record the ETF used. A stock with no profile is scored on its own returns and `residual: null`.

### 3.2 Earnings quality from the statements already stored

**Edge.** Profits that are not cash do not last. High accruals predict weaker returns. Receivables or inventory racing ahead of sales is the same warning in operating form.

**Data.** The 10-Q / 10-K statements `filing_signals.py` already reads. Add rows, still with no score in that module: cash from operations; receivables; inventory; total revenue (already in the income headlines). Accrual ratio = (net income − cash from operations) / total assets, both periods. `receivables_vs_sales` and `inventory_vs_sales` = the percent change in the balance-sheet line minus the percent change in revenue.

**How it enters.** Not a buy signal. A **red-flag modifier and a gate**:

- Accrual ratio above 0.10, or receivables growth exceeding revenue growth by more than 15 percentage points: quality multiplier × 0.5, and the reversion family is turned off for that name. A dip in a company stuffing the channel is the falling knife the handoff 5.3 filter is meant to avoid.
- Cash from operations below zero while net income is positive: same treatment.
- `as_of` is the filing date. If the filing is older than 120 days, the modifier does not apply and `note` says the statements are stale. Finnhub's fresh margins still feed the quality score (entry 8).

### 3.3 Year-over-year change in the risk factors

**Edge.** A 10-K whose risk factors were rewritten has, on average, weaker subsequent returns than one that was filed almost unchanged. The rewrite is the company telling you something the boilerplate hid. (Cohen, Malloy, Nguyen, "Lazy Prices," Journal of Finance, 2020.)

**Data.** Stored 10-K risk-factor text, two filings of the same company. No new fetch when both are already in `data/filings/`.

**Code.** A similarity between the two risk-factor sections (token overlap is enough; a cosine on the same embeddings the RAG store uses is better). `risk_factor_change = 1 − similarity`. Above 0.35: bearish modifier, quality × 0.75, and the model must quote one sentence that appears in the new filing and not the old one. Code checks the quote is in the new section and absent from the old. Below 0.1: no effect. One filing on disk: `null`, not a pass.

**Failure.** A rewrite can be cosmetic (a new regulation paragraph every filer added). Require the quote to be company-specific by the same name test the Advisor uses today. A quote that is only the boilerplate change does not count.

### 3.4 Second-order names in a theme (extends handoff 5.7)

**Edge.** For "AI memory shortage" the first articles name MU and SNDK, which have often already moved. The companies that buy from them or supply them are in the 10-K business section and in Polygon's related-companies list, and they are often where the leftover move sits.

**Data.** (1) Stored 10-K business text: the model proposes up to five counterparties, each with a quote; code checks the quote and resolves the name through the existing company lookup, which refuses an ambiguous name. (2) Polygon related-companies, NEEDS-YES, as a second list the model does not get to invent. A name has to appear in a verified quote or in that endpoint. The model's memory is not a source.

**How it enters.** Theme family only. A counterparty with `residual_21d` below the theme basket's median, that passes quality and the liquidity gate, is a candidate tagged `theme_role: counterparty`. The headline names stay in the candidate table with `theme_role: leader` and their 21-day return, so the Critic sees who was crowded.

### 3.5 Insider clusters and estimate revisions (bench, NEEDS-YES)

Add these only after the core families are scoring. Each is one Finnhub endpoint, facts only, no score inside the Data Clerk.

**Open-market insider buys.** Endpoint: insider transactions. Signal: at least two different insiders, open-market purchases (not option exercises, not gifts), in the last 30 days, and no offsetting open-market sale by an insider larger than the buys. This is a modifier on the trend and capital-return families (+0.25 × the 20-day excess), and a reason to turn reversion *on* after a drop. Option exercises and 10b5-1 sales do not count. Those are scheduled, not opinions.

**EPS revisions.** Endpoints: recommendation trends and EPS estimates. Signal: the current-year estimate is up over 30 days and the share of analysts at "buy" is up. Modifier on the event and trend families. A cut to the estimate vetoes the reversion family even if the price drop looks non-fundamental: the Street is saying the drop has a number behind it.

**Short interest**, only if the current Finnhub plan returns it. Days-to-cover above 5 with a `guidance_cut` or a dilution filing strengthens `reduce` / `avoid`. Days-to-cover above 5 with a positive event and a small price reaction is a squeeze risk, not an automatic buy: raise volatility in the sizer (use 1.5 × vol) and say `squeeze_risk: true`. Short interest alone is not a strategy.

### 3.6 What "what's happening" should actually return (Case 3)

The handoff runs the same stock screens across the index. A desk answering "what's happening" leads with the tape, then the names.

Ask the Data Clerk for a **market brief** computed from the day aggregates and the market-news store, before any stock score:

1. Sector ETFs ranked by 21-day and 63-day return, with `breadth_200` inside each sector.
2. Names in the universe with `volume_ratio_1d` above 3 and a 1-day move above 1 × ATR, joined to the top headline `discover` or the news search returns for that symbol.
3. Event classes from section 2.11 that fired in the last 5 days, with the underreaction flag.
4. The regime state from section 2.8.

Then the stock rank runs **inside the top three sectors by 63-day return**, plus any name from (2) or (3) that passes the gates. A quiet stock in a lagging sector can still win the trend family on a Case 2 theme request. It should not be what Case 3 leads with.

The brief is facts on the work order (`data.market_brief`). The model writes three sentences from it. Code checks that every ticker and every percent in those sentences appears in the brief.

---

## 4. The research checklist the Advisor asks for

Handoff section 4 lets the model read whatever text is already on the order and then request metrics. A symbol that arrived from a single news search has a thin file. The plan step should issue a fixed checklist, as Data Clerk requests, and store each line as `found`, `empty`, or `error`. Empty is a result. It is how the Critic knows nobody hid a filing.

For each candidate, up to a cap of 15 symbols:

| # | Request | Action the desk already has, or the new one |
|---|---|---|
| 1 | News on the symbol, last 14 days | `search_news` |
| 2 | Press releases, last 180 days | `search_news` on `press_releases` |
| 3 | Filing evidence | `fetch_filings_recommendation` (stored text only) |
| 4 | Quote | `fetch_quote` |
| 5 | The metric list for the families still eligible | `calculate_financials` |
| 6 | Theme members, when the work order has a phrase | `discover` |
| 7 | Profile and peers, when approved | new, NEEDS-YES |
| 8 | Earnings dates and the last four surprises, when approved | new, NEEDS-YES |

Stop adding candidates when the checklist for the current set is unfinished. A deeper file on eight names beats a headline on thirty.

Metrics to add to the handoff's catalogue, all from bars or from rows above: `residual_mom_12_1`, `residual_5d`, `sector_ret_5d`, `sector_ret_63d`, `breadth_200`, `accrual_ratio`, `receivables_vs_sales`, `inventory_vs_sales`, `sbc_to_revenue`, `payout_ratio`, `risk_factor_change`, `earnings_move_abs`. Same provenance rules as the handoff: `None` and a `note` when the inputs are short, no score keys anywhere in the payload.

---

## 5. Three cases, three sleeves

The handoff uses one ranker for every request. The user-facing cases in the design PDF want different holdings.

| Case | Question | Sleeve | top-k | What wins |
|---|---|---|---|---|
| 1 | "Here is $1,000, trade it." | `preserve` | 1 to 3 | Capital return and the quality multiplier, low 63-day vol, trend family only in a `trend` regime. Event and reversion families may *veto* a name. They may not add a new name unless the preserve sleeve found nothing. The user did not ask for a headline. |
| 2 | "AI memory shortage", "weight loss pills." | `theme` | 3 to 5 | Theme family, residual versus the theme basket, counterparties from section 3.4, quality multiplier, underreaction. Leaders that already ran are shown and ranked down. |
| 3 | "What's happening?" | `tape` | 5 | Market brief first (section 3.6), then event family and residual trend inside the leading sectors. |

Case 1 still runs the holdings pass (section 2.6) on the existing portfolio before it opens anything new. A `reduce` on a held name outranks a new `open`. That is the trade that protects the $1,000.

Bearish stays a first-class result for names already held (`reduce` / `avoid`). The desk does not open shorts.

---

## 6. Ideas that look smart and should stay out

| Idea | Why it stays out |
|---|---|
| Copying 13F filings | The filing is 45 days late. By the time it is public the move is crowded, and the desk cannot see the hedge that sat next to the long. |
| Social sentiment as a buy signal | `search_socials` is not wired. When it is, use a spike in mentions as a *crowding penalty* on the trend family, the same way a volume spike already works in the handoff. Mentions are not a thesis. |
| RSI below 30 by itself | Allowed only inside the reversion family, after the idiosyncratic-drop test, the quality multiplier, and the "cause is non-fundamental" class. An RSI screen on its own buys broken names. |
| Calendar effects (Monday, month-end, "Santa") | Too small to survive a spread, and they will not reproduce on two years of bars. |
| A new weight for every event class | Classes in section 2.11 share the event family's one weight. Per-class weights with a handful of examples will fit noise. The ledger may *report* per-class hit rates. It may not fit a coefficient until a class has 30 outcomes. |
| Letting the model average the families | Correlated signals (12-minus-1, 52-week high, residual momentum) would be counted three times. Section 2.1 gives the family one vote. |
| Tuning the RSI cutoff or the ATR multiple from the ledger | The handoff already forbids this. Keep it. With a few dozen trades the optimized cutoff is a memory of those trades. |
| Intraday order-flow as a strategy | Design choices entry 9 kept trades and quotes for this agent. Use them as a **confirmation on the final top-k only**, and only for dates you have on disk (the current month): median spread, and the share of volume in prints above $100k. A confirmation can move `agreement` by at most 0.1. It cannot add a symbol. It cannot be backtested on a long window until more months exist, so the record says `intraday_window: "current month only"`. |

---

## 7. What the recommendation must contain so the Critic can grade it

The handoff's schema v2 is enough for a winner. Add these fields, all filled by code unless noted.

| Field | Why the Critic needs it |
|---|---|
| `case`, `sleeve` | A Case 1 name and a Case 3 name are not graded the same way. |
| `action` | `open / add / hold / reduce / avoid`. The Critic checks it matches the holdings and the invalidation rules. |
| `family`, `excess_20d_after_cost`, `cost_dollars` | The rank is reproducible without the model. |
| `completeness`, `agreement`, `evidence_n` | A thin file cannot hide inside one confidence number. |
| `residual_mom_12_1` or `residual_5d`, with the sector ETF | "It fell" and "it fell alone" are different trades. |
| `quality`, and any haircut (`sbc`, `accrual`, `risk_factor_change`) with the input values | The Critic can knock out a haircut it disagrees with and see the rank change. |
| `counter_evidence` and `contradictions` | A contradiction is two verified quotes that the model flagged as opposed (demand strong in the press release, orders delayed in the news). Code checks both quotes exist. One unverified side drops the pair. |
| `order_style`, `gap_risk`, `earnings_move_abs` | The Critic rejects a chase and a position that cannot survive the next report. |
| `cooldown_of` | The earlier recommendation id, when this run stayed quiet because the last one is still in force. |
| `candidates[]` | The rejects and the runner-up (section 2.7). |

Thesis rule, tighter than the handoff: the model writes at most three sentences for the top-k, and it may state a number only if that number is in `features`, `excess_20d_after_cost`, or `trade`. It must mention the best counter-evidence by id when any exists. A thesis with no bear sentence when `counter_evidence` is non-empty is rejected and retried once.

---

## 8. How this changes the build order

Follow the handoff's phase list, with these insertions. Still no live fetch without a yes.

1. With the bar metrics: also compute sector-ETF returns, residuals, and `breadth_200`. Fixture: a few invented symbols plus one invented ETF over 300 days.
2. With filing evidence: add the cash-from-operations, receivables, inventory, and stock-based-compensation rows, plus `risk_factor_change` when two 10-Ks are stored. Keep the "this module contains no score key" test.
3. Before the composite: implement families, after-cost 20-day excess, the quality multiplier, and the small-set threshold path. Test that two correlated trend features produce one vote.
4. With the record: `action`, the holdings pass, the cooldown, and `candidates` including a gate failure.
5. Case sleeves: three tests, one per case, on the same fake universe, asserting Case 1 does not open a fresh event name while a capital-return name qualifies, and Case 3's first lines are the market brief.
6. Bench endpoints (profile, earnings, insider, estimates, related companies) stay behind NEEDS-YES and behind fakes in tests. The families must score with those fields set to `null`.

Leave Advisor v0 callable, as the handoff says, so the week 3 with-and-without-retrieval comparison still runs.

---

## 9. Extra reading, for the priors on the new pieces

Blitz, Huij, Martens 2011 (residual momentum). Sloan 1996 (accruals). Cohen, Malloy, Nguyen 2020 (changes in 10-K language). Lakonishok and Lee 2001 (insider buying). Chan, Jegadeesh, Lakonishok 1996 (estimate revisions). Frazzini and Pedersen 2014 (betting against beta; the reason Case 1 prefers lower vol). The handoff's section 14 list still covers the original seven.
