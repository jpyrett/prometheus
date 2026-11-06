# Advisor v1 handoff: a multi-strategy stock recommendation system for Prometheus

Jacob Pyrett, Nov 2026. Written for the coding model that will build it (Sonnet 5.5). This is a design, not code. It says WHAT to build, WHY each
piece is there (the finance), and WHERE it fits the existing desk. Read `AGENTS.md`, `README.md`, `design/jpyrett_capstone_tradingbot.pdf` and
`docs/design_choices.md` entries 25, 26, 38 to 42 first. Nothing here changes a rule of those documents; where it extends one, it says so.

---

## 0. The one-paragraph brief

Replace Advisor v0's single score (bullish share of verified quotes) with an Advisor that runs **several independent trading strategies** over
the data the Data Clerk puts on the work order, **calculates every number in code** (the model proposes, code verifies and scores: entry 26),
ranks the candidates by **expected risk-adjusted return with a diversity rule**, and returns **top-k recommendations that carry everything the
Critic needs to re-check them without re-asking the model**: the strategy, the thesis, the verified evidence on both sides, the features with
their as-of times, the entry zone, the stop, the horizon, the invalidation rules, the risk, the portfolio fit, and the base rate the expected
return rests on. Then close the loop: record every recommendation with its features, measure its forward return, and let those measurements, not
opinions, move the strategy weights.

Honesty up front: no retail system "makes money" by construction. What an honest design can do is (1) take positions only where there is a
documented, persistent edge, (2) refuse the trades that lose retail money (chasing a run, illiquid names, blind event risk, oversized positions),
(3) size by risk, and (4) measure itself so the weights drift toward what has worked. That is what this document specifies.

---

## 1. Non-negotiables inherited from the project

1. **No LLM in a number.** Scores, features, ranks, sizes, stops and budgets are calculated by code from data on the work order (AGENTS.md; entries
   24, 26, 42). The model's jobs are: classify, propose, quote, explain. Every proposal is checked by code before it counts.
2. **The work order is the state** (entry 25). The Advisor reads `data.*` and appends to `recommendations`; it never fetches on its own. Anything
   it needs that is not on the order is asked of the Data Clerk through a work order action.
3. **The Data Clerk does not judge** (entry 41). It fetches, searches, summarizes, and now also **calculates named metrics** (section 3). It
   returns facts with as-of times and sources; no stance, no score.
4. **Evidence is a quote that code can find in its source** (entry 26), or a metric with a provenance record. A claim with neither is not evidence.
5. **Paid calls and downloads need Jacob's yes** (AGENTS.md). Every new data need below is marked NEEDS-YES. Tests use fakes.
6. Each phase gets a numbered entry in `docs/design_choices.md` (problem, decision, alternatives, consequences), tests with mutation checks, and
   the docs updated (`README.md`, `HowToRun.txt`, `design/week*.md` as appropriate).

---

## 2. What data exists today (the inventory the strategies are built on)

| Data | Where it comes from | What it gives | Already on the work order? |
|---|---|---|---|
| News | `DataToolbox.fetch_news` (Finnhub company news + Polygon news), RAG `retrieve_news` | headline, summary, publisher, date, tickers, keywords, **Polygon per-ticker `insights` (sentiment + reasoning)** | yes: `data.sources`, `data.results.news` |
| Press releases | `fetch_press_releases` (Finnhub), `retrieve_press_releases` | full text of the company's own announcements | yes: `data.results.press_releases` |
| SEC filings | `fetch_filings` (Massive index, 10-K/8-K text), `fetch_filing_text` (10-Q from EDGAR), `retrieve_filings` | sections of 10-K, 8-K items, 10-Q statements; index rows for DEF 14A, 13D/13G, 144 | yes: `data.results.filings` |
| Filing evidence | `fetch_filings_recommendation` (`filing_signals.py`) | balance sheet both periods, cash-flow capital actions (buybacks, dividends, SBC, issuance, debt), income headlines, exact passages by topic (dilution, exec comp, equity comp, buyback, dividend), gaps | yes: `data.filing_evidence[symbol]` |
| Live quote | `get_quote` (Finnhub `/quote`), `DataClerk.request_quotes` | price, change, high, low, open, previous close, as-of | yes on a `fetch_quote` order: `data.quotes[symbol]` |
| Fundamentals | `Financials.get` (Finnhub `metric=all`, 10 metrics kept) | P/E, EPS, net/gross/operating margin, ROE, D/E, current ratio, dividend yield, revenue growth; the raw payload has many more (market cap, 52-week high/low, beta, 10-day ADV, ...) | **not yet** |
| Daily bars | `data/historical/day_aggs_v1/YYYY/MM/*.csv.gz` (Massive flat files: every US ticker, OHLCV + transactions, per day) | the price history for every technical feature; `performance_summary` already reads one month | **not yet** |
| Minute bars, trades, NBBO quotes | `minute_aggs_v1`, `trades_v1`, `quotes_v1` flat files (26 GB, current month) | intraday volume profile, spread, trade-size distribution | not yet; use sparingly (section 5.9) |
| Trend discovery | `DataToolbox.discover(phrase)` over the market news store | which tickers a theme is about, with article counts and insight sentiment per ticker | not yet on the order (Case 2 entry point) |
| Portfolio | `configs/portfolios/*.json`, `common/portfolio.py` | holdings, basis, cash, T+1 locks | available to the Account Manager; the Advisor should read it for portfolio fit (section 7) |

Data the strategies want that does **not** exist yet (each NEEDS-YES, each is one Finnhub or Massive endpoint or one more flat-file month):

- **Earnings history and calendar** (actual, estimate, surprise, next report date). Needed by PEAD (5.1) and by the event-risk gate (section 8).
- **Company profile** (sector/industry, market cap, shares outstanding). Needed by sector diversity (section 7), net payout yield (5.4), value vs peers (5.6).
- **Daily bars back 13+ months** for every ticker: `day_aggs_v1` for earlier months (small files: one month of day aggs is a few MB, unlike trades).
  Needed by momentum (5.2), 52-week high (5.2), volatility and beta (section 8). Without it, momentum is limited to what is stored.
- **Index membership** (S&P 500 / NASDAQ-100 constituents as a file in `configs/`). Needed by Case 3 ("What's happening?") as the universe.
- Optional, later: analyst recommendation trends and EPS revisions, insider transactions (Form 4). Both are documented, modest edges.

---

## 3. New contract: the Advisor asks the Data Clerk to CALCULATE metrics

Jacob (2026-11-05): the Advisor may ask the Data Clerk to calculate specific metrics (RSI, P/E, ...). The work order already names the action
`calculate_financials`; `verify_request` still refuses it. This section is the contract to wire.

**The request.** A work order with `action_type="calculate_financials"`, `subject` = list of symbols (bounded, like `MAX_QUOTE_SYMBOLS`), and a new
field `metrics`: a list of metric names from a fixed catalogue (below), each optionally with parameters (`{"name": "rsi", "window": 14}`). An
unknown metric name is refused by `verify_metric_request` in code, before any work is done, with the list of known names in the message.

**The answer.** `data.metrics[symbol][metric_name]` = `{value, unit, as_of, window, source, inputs_summary, note}`:
- `source` says where the number came from (`day_aggs` with the date range read, `finnhub_metric` with the fetch date, `filing` with accession).
- `inputs_summary` is enough for the Critic to recompute or sanity-check (for RSI: the number of bars, the last close; for P/E: price and EPS used).
- `note` carries a gap in words ("only 40 bars stored, 200-day average not computed"), never a judgement. `None` for a value that cannot be computed.
- The same provenance rule as `filing_signals`: a test walks the result and refuses the keys `score`, `stance`, `sentiment`, `recommendation`, `rating`.

**Where the arithmetic lives.** Plain Python in the toolbox (new module next to `filing_signals.py`, e.g. `technicals.py` for bar-based metrics and
`fundamentals.py` for ratio-based ones), exposed as one Data Clerk tool `calculate_metrics(symbols, metrics)`. No LLM in the path: the Data Clerk's
code answers a `calculate_financials` order the way it answers `fetch_quote` (entry 42 pattern), no model loop.

**The catalogue (version 1).** Bars = daily bars from the stored day aggs.

| Group | Metric | Definition (code) | Used by |
|---|---|---|---|
| Returns | `ret_1d, ret_5d, ret_21d, ret_63d, ret_126d, ret_252d` | close-to-close simple return over N trading days | all |
| Momentum | `mom_12_1` | return from 252 bars ago to 21 bars ago (skip the last month) | 5.2 |
| Momentum | `pct_of_52w_high` | last close / max(high, 252 bars) | 5.2 |
| Reversal | `rsi_14` | Wilder RSI, 14 bars | 5.3 |
| Reversal | `zscore_21d` | (close − 21-bar mean) / 21-bar std | 5.3 |
| Trend | `sma_50, sma_200, above_sma_200` | simple moving averages; last close above the 200 | regime, 5.2 |
| Volatility | `atr_14, atr_pct` | Wilder ATR; ATR / close | sizing, stops |
| Volatility | `vol_21d_ann, vol_63d_ann` | std of daily log returns × √252 | ranking, sizing |
| Volatility | `max_drawdown_63d` | peak-to-trough over 63 bars | risk |
| Market | `beta_63d` | OLS slope of daily returns on SPY's | regime, risk |
| Liquidity | `adv_20d_dollars` | mean(close × volume, 20 bars) | gate |
| Liquidity | `volume_ratio_1d` | today's volume / 20-bar mean volume | 5.5 attention |
| Gap | `gap_1d` | today's open / yesterday's close − 1 | 5.3, 5.5 |
| Fundamental | `pe_ratio, eps, roe, gross_margin, operating_margin, net_profit_margin, debt_to_equity, current_ratio, dividend_yield, revenue_growth` | from `Financials.get` (already written) | 5.4, 5.6 |
| Fundamental | `market_cap, shares_outstanding, beta_provider, high_52w, low_52w` | from the same Finnhub payload (keep more fields) | several |
| Fundamental | `earnings_yield` | 1 / pe_ratio when pe > 0 | 5.6 |
| Capital return | `net_payout_yield_ttm` | (buybacks + dividends − issuance, from `filing_evidence.capital_actions`) / market_cap; `None` when any input is missing | 5.4 |
| Capital return | `share_count_change` | from `filing_evidence.balance_sheet.shares.change_pct` | 5.4 |
| Events (NEEDS-YES) | `eps_surprise_last, sue_4q, days_to_next_earnings` | from the earnings endpoint; SUE = (actual − estimate) / std of the last 8 surprises | 5.1, gate |
| Spread (optional) | `median_spread_bps_1d` | from `quotes_v1` NBBO for one day: median (ask − bid) / mid | gate, 5.9 |

Make the catalogue a data structure (name → function, inputs, unit, description), so adding a metric is one entry plus one test, and so the Advisor's
prompt can list the names it may ask for.

**Sequencing in the desk.** The Advisor runs in two passes: (a) read the text evidence on the order and decide the candidate symbols and which
strategies apply; (b) ask for the metrics those strategies need (one `calculate_financials` order for all candidates), then score. In the LangGraph
this is `advisor_plan -> data_clerk_metrics -> advisor_score`, or a sub-graph inside the advisor node; keep the metric request on the order
(`metric_requests`) so the audit trail shows what was asked and what came back.

---

## 4. The architecture of the Advisor v1

```
work order (data_ready: sources, results, filing_evidence, quotes)
   |
   v
[A] Candidate generation (code + model)
     - symbols named in the order, in the chunks (ticker lists + quote-named companies), in discover() for a theme
     - the model proposes which STRATEGY ARCHETYPES the evidence fits for each symbol, with quotes (checked by code)
   |
   v
[B] Metric request -> Data Clerk calculate_financials -> data.metrics        (section 3)
   |
   v
[C] Gates (code): liquidity, price, data freshness, hard red flags, event-risk window      (section 8)
   |
   v
[D] Strategy modules (code): each produces, per symbol, a strategy score in [0,1], a confidence,
     the features it used, the evidence ids it rests on, and the counter-evidence it found     (section 5)
   |
   v
[E] Composite ranking (code): expected return × confidence, risk-adjusted, portfolio-aware,
     diversity-constrained top-k                                                              (section 6, 7)
   |
   v
[F] Recommendation records (schema v2) with everything the Critic needs                       (section 9)
   |
   v
[G] Outcome ledger + weight update (the self-learning loop)                                   (section 10)
```

The model is called in [A] (archetype + quotes + thesis text) and once at the end of [F] to write the human-readable thesis, bull case and bear
case **from the already-calculated record** (it may only restate what is in the record; code checks that every number it writes appears in the
record, the same way quotes are checked today). Everything else is code.

---

## 5. The strategy catalogue (what makes money, and why)

Each strategy is a module with the same interface: inputs it needs (from `data.*`), a `score(symbol) -> StrategyResult` in code, the evidence
classes it accepts, its default horizon, its base-rate prior, and its failure modes (which become Critic checks). "Prior" numbers below are from
the published literature and are placeholders until section 10 replaces them with measured ones; the record must say which it is.

### 5.1 Post-earnings-announcement drift (PEAD) and guidance events

- **Edge.** After a large positive earnings surprise, prices keep drifting the same way for roughly 60 trading days (Bernard & Thomas 1989; still
  documented in recent decades, smaller in large caps). Guidance raises and cuts in press releases and 8-K Item 2.02 behave the same way.
- **Signal (code).** `sue_4q` in the top quintile of the candidate set (or `eps_surprise_last` > +5% when SUE is unavailable), AND the
  announcement-day return (`gap_1d` or `ret_1d` on the report date) has the same sign (the market agreed), AND the event is 1 to 10 trading days old
  (not stale, not chasing the gap itself). Event date from the earnings endpoint or from the press release / 8-K date.
- **Text evidence (model proposes, code checks).** A quote from the earnings press release or 8-K containing a guidance change; the event
  taxonomy (5.5) must classify it as `guidance_raise` / `guidance_cut` with the direction fixed by the class, as `filing_signals` fixes direction by topic.
- **Horizon.** 20 to 60 trading days. **Prior.** Top-quintile SUE: about +2% to +4% abnormal over 60 days historically; use 1.5% as a conservative prior.
- **Failure modes (→ Critic checks).** Surprise driven by a one-off item; the whole move happened on day 1 (gap > 10% with no follow-through:
  skip); the next earnings date is inside the horizon (fine for PEAD, a risk for everything else).

### 5.2 Momentum with a quality filter, and the "already on a run" trap

- **Edge.** 12-month-minus-1-month momentum (Jegadeesh & Titman 1993): past winners over 2 to 12 months keep winning for 3 to 12 months. Proximity
  to the 52-week high is a separate, robust version (George & Hwang 2004). The **last month reverses** (Jegadeesh 1990): a stock up 20% in 4 weeks
  is a worse buy, not a better one. This is the precise sense in which "buy the obvious ones on a run" is wrong, and the design must encode it.
- **Signal (code).** `mom_12_1` rank in the top 20% of the universe or candidate set AND `pct_of_52w_high` ≥ 0.90 AND `above_sma_200` AND
  **`ret_21d` not in the top 10%** (the anti-chase rule) AND `rsi_14` < 75 AND quality filter: `roe` > 0 and `operating_margin` > 0 (momentum in
  unprofitable names crashes harder). Prefer **lower `vol_63d_ann`** among qualifiers (low-volatility momentum has the better Sharpe).
- **Horizon.** 3 months, re-checked monthly. **Prior.** About +0.5% to +1% a month over the market for the top decile, concentrated in calm regimes.
- **Failure modes.** Momentum crashes after a sharp market drop and rebound (Daniel & Moskowitz 2016): disable the strategy when the regime
  (section 8) is "recovering from drawdown"; crowded names (news-volume spike + 1-month surge) are excluded by the anti-chase rule.

### 5.3 Quality on sale: short-term reversal in sound companies

- **Edge.** Single-stock overreactions to non-fundamental news revert within days to weeks (Jegadeesh 1990; Lehmann 1990), strongly conditional
  on the company being sound and the news not changing the earnings picture. Retail loses here by buying falling knives with real problems; the
  filter is what makes it work.
- **Signal (code).** `ret_5d` ≤ −7% or `zscore_21d` ≤ −2 or `rsi_14` ≤ 30, AND quality: `roe` > 10%, `debt_to_equity` < 1.5, `current_ratio` > 1,
  `revenue_growth` > 0, AND `above_sma_200` (a dip in an uptrend, not a downtrend), AND no hard red flag (section 8), AND **the news that caused
  the drop is classified as non-fundamental** (5.5: analyst downgrade, sector sympathy, macro, index flow, a lawsuit headline with no filing) or
  there is no company-specific news at all (sympathy move).
- **Text evidence.** The model must find the cause of the drop and quote it; the taxonomy class decides whether it qualifies. If the class is
  `guidance_cut`, `restatement`, `going_concern`, `dilution`, `regulatory_action`: not a reversal candidate, it is a 5.1 short or a skip.
- **Horizon.** 5 to 20 trading days. **Prior.** +1% to +2% over 2 weeks for the bottom decile of 1-week returns with the quality filter; smaller in
  large caps. Use 1%.
- **Failure modes.** Fundamental cause mislabelled as non-fundamental (Critic re-reads the quote and the class); the dip continues (stop at
  2 × ATR below entry); earnings inside the horizon (gate).

### 5.4 Capital return: buybacks, dividends and the absence of dilution

- **Edge.** Net payout yield (buybacks + dividends − issuance) predicts returns better than dividend yield alone (Boudoukh, Michaely, Richardson,
  Roberts 2007); buyback announcements earn long-run abnormal returns, more so for value stocks (Ikenberry, Lakonishok, Vermaelen 1995); dividend
  initiations and increases are positive events, cuts are strongly negative (Michaely, Thaler, Womack 1995). Repeated issuance (ATMs, converts,
  secondaries) predicts underperformance (Loughran & Ritter 1995). **This strategy is almost entirely built already**: `filing_signals.py` extracts
  exactly these facts with direction fixed by topic. What is missing is the arithmetic and the ranking.
- **Signal (code).** `net_payout_yield_ttm` ≥ 3% (or, when market cap is missing, buybacks + dividends ≥ 2% of total equity and
  `share_count_change` < 0), AND `share_count_change` ≤ 0, AND no `dilution` evidence item with a current-period dollar figure, AND
  `debt_to_equity` not rising above 2 (buybacks funded by debt in a weak balance sheet are a red flag, not a signal), AND `earnings_yield` > 0.
  Bonus (an event, 20 to 60 day horizon): a press release classified `buyback_authorization` or `dividend_increase` in the last 10 days.
- **Text evidence.** The E-items already on the order (buyback and dividend items are bullish by type; dilution and compensation items bearish).
  A `dilution` E-item that passes the Advisor's check counts against the strategy score and must appear in `counter_evidence`.
- **Horizon.** 3 to 12 months; it is the slow, compounding strategy in the set. **Prior.** +2% to +4% a year over the market for the top quintile.
- **Failure modes.** Buyback announced but not executed (the cash-flow row is the check, the press release is only the announcement); shrinking
  revenue with buybacks (a melting ice cube: require `revenue_growth` ≥ −5%).

### 5.5 Event-driven: a taxonomy of corporate events with fixed directions

- **Why.** The press releases and 8-Ks on the order are the earliest, cleanest statement of what changed. Generic "sentiment" is weak; **specific
  event classes with documented average reactions are not**. Build the taxonomy the way `filing_signals.TOPIC_PATTERNS` is built: regex
  patterns, a fixed direction per class, exact quotes, and the model may only cite a class the code also matched.
- **Classes (direction, typical reaction, horizon).** Positive: `guidance_raise`, `earnings_beat`, `buyback_authorization`, `dividend_increase`,
  `dividend_initiation`, `activist_stake_13d` (Schedule 13D filed: about +5% to +7% abnormal around the filing, Brav, Jiang, Partnoy, Thomas 2008),
  `acquisition_target` (being bought: the premium is mostly paid on day 1, so a skip unless the spread is wide), `major_contract_award`,
  `fda_approval`, `index_inclusion`, `insider_purchase_cluster` (later, needs Form 4). Negative: `guidance_cut`, `earnings_miss`, `dividend_cut`,
  `dilution_offering` (ATM, secondary, convert), `going_concern`, `material_weakness`, `restatement`, `ceo_departure_unplanned`, `delisting_notice`,
  `regulatory_action`, `form_144_sale_notice` (insider intends to sell), `acquirer_large_deal` (acquirers underperform on average).
  Neutral-but-informative: `analyst_rating_change` (short-lived), `sector_sympathy`, `macro`, `lawsuit_headline` (noise unless a filing follows).
- **Signal (code).** Event class, event age in trading days, the price reaction so far (`ret_1d` on the event date, `ret_5d` since), and the
  **underreaction test**: a positive event with a small reaction so far (|reaction| < 1 × `atr_pct`) is the attractive case; a positive event whose
  reaction already exceeds 3 × ATR is "priced", score it low; a positive event followed by a negative reaction is a conflict to show the Critic.
  News-volume spike (`volume_ratio_1d` > 3 and article count > 3 × the symbol's 30-day daily mean) is **attention**, which predicts short-term
  buying by retail then reversal (Barber & Odean 2008): a penalty in 5.2 and 5.3, a mild positive only inside 5.1's window.
- **Horizon.** 5 to 60 days by class. **Prior.** By class, from the papers cited, replaced by the ledger.

### 5.6 Value with a quality guard (anti-value-trap)

- **Edge.** Cheap and profitable beats cheap alone (Novy-Marx 2013: gross profitability; Piotroski 2000: financial strength within value stocks).
  With the ten Finnhub metrics this is a screen, not a model, and it is mostly useful as a **tilt** that other strategies' candidates are ranked by,
  and as a Case 3 generator for the universe.
- **Signal (code).** `earnings_yield` in the top third of the universe or sector AND `gross_margin` and `roe` above the universe median AND
  `revenue_growth` > 0 AND `debt_to_equity` < 2 AND `above_sma_200` or `pct_of_52w_high` ≥ 0.7 (do not buy value that is still falling).
- **Horizon.** 6 to 12 months. **Prior.** Small and slow; weight it low until measured.

### 5.7 Theme exposure with a "not yet priced" filter (Case 2: "AI memory shortage")

- **Why.** `discover(phrase)` finds which tickers a theme is about. The naive version buys the most-mentioned name, which is the most crowded
  one. The better version ranks theme members by **exposure × quality × room**: how central the company is to the theme (article share, insight
  sentiment agreement), whether it is sound (5.6 guard), and whether the price has already moved (`ret_21d`, `pct_of_52w_high`, `volume_ratio`).
- **Signal (code).** Theme membership from `discover` (article count ≥ 3, positive insight share ≥ 0.6) AND quote-verified signals on the order
  that name the company in the theme's context (today's `verify_signal`) AND `ret_21d` below the theme median (the second-tier names, not the
  one everyone already bought) AND quality guard. Show the theme leader too, scored lower with the reason "already moved N% in 21 days".
- **Horizon.** 1 to 3 months. **Prior.** None from the literature; start at the PEAD prior halved, measure.

### 5.8 Pairs inside a theme (optional, later)

For a theme with two comparable members (MU vs SNDK), long the laggard / underweight the leader when their 21-day return spread exceeds 2 σ of
its 6-month history. Long-only desk: this becomes "prefer the laggard" inside 5.7, not a short. Keep for after the ledger shows 5.7 works.

### 5.9 Microstructure confirmation (optional, uses the 26 GB)

Not a strategy; a **confirmation and cost layer** for the others, computed for the final top-k only: median NBBO spread from `quotes_v1` (a gate:
skip > 20 bps), trade-size-weighted buy/sell imbalance on the event day from `trades_v1` (a tick-rule estimate; confirms whether the event was
bought or sold by size), and the closing-30-minute volume share from `minute_aggs_v1` (institutional participation). Use it to adjust confidence by
±10%, never to generate candidates.

---

## 6. Scoring and ranking (all code)

**Per strategy.** `StrategyResult = {strategy, symbol, score_0_1, confidence_0_1, expected_return, horizon_days, features_used: {name: {value, z, as_of}},
evidence_ids, counter_evidence_ids, gates_passed, reasons[]}`.
- `score_0_1`: the strategy's own rule, mapped to [0, 1] by rank within the candidate set when there are ≥ 8 candidates, else by fixed thresholds
  (both documented in the record: `scoring_basis: "rank" | "threshold"`).
- `confidence_0_1`: the share of the strategy's required inputs that were present and fresh, × the evidence agreement (verified bullish quotes / all
  verified quotes for that symbol, smoothed as today), × the microstructure adjustment if computed. A strategy with a missing required input
  returns `None`, not a guess.
- `expected_return`: `prior` (literature) or `measured` (ledger, section 10) for that strategy in the current regime, × `score_0_1`. The record says
  which and gives the sample size when measured.

**Composite per symbol.** Strategies are independent views, so combine as a weighted sum of `expected_return × confidence` with strategy weights
`w_s` (start equal; the ledger updates them), minus a **risk charge**: `λ × vol_63d_ann` with λ chosen so a 40%-vol name needs ~2% more expected
return than a 20%-vol name, minus the **crowding penalty** (`ret_21d` top decile or attention spike). Keep the per-strategy results in the record
so the Critic can see which view carried it.

**Conflicts.** If one strategy is bullish and another bearish on the same symbol (e.g. 5.5 `dilution_offering` with 5.2 momentum), the record gets
`conflict: true` with both results and the composite is **not** the sum: it is the lower of the two expected returns. Conflicts are what a Critic is for.

**Top-k with diversity.** Greedy selection, like MMR: pick the best composite; then repeatedly pick the candidate maximizing
`composite − μ × max_similarity_to_already_picked`, where similarity is 1 for the same sector, 0.7 for the same theme, 0.5 for `beta_63d` within 0.2 and
return correlation > 0.7 over 63 days, else 0. Cap: no more than ⌈k/3⌉ from one sector, no more than ⌈k/2⌉ from one strategy. Record the
`diversity_adjustment` and who was displaced by whom.

**Stance.** `bullish` when composite expected return > risk charge + cost estimate (spread + a fixed 10 bps), `neutral` in between, `bearish` when
the negative view dominates. The desk is long-only: bearish recommendations are returned as "do not buy / consider selling if held" and are what
the Account Manager's RaiseFunds should prefer to sell (a hook, not a change to RaiseFunds now).

---

## 7. Portfolio awareness

The Advisor reads the user's portfolio (the same `Portfolio` the Account Manager loads) and adds, per recommendation:
- `held_already` (shares, basis, unrealised %), `sector_weight_after` (if bought at the suggested size), `correlation_to_portfolio_63d`
  (return correlation of the candidate with the portfolio's daily return, from the bars).
- A **concentration penalty** in the ranking when the sector weight after would exceed 35% or the correlation exceeds 0.8.
- `funds_feasible`: whether settled cash or RaiseFunds under its rules (only sell in profit, keep one share, T+1) could pay for one unit of the
  suggested size. It is a flag for the Critic, computed with the existing `profit_bounds` / portfolio helpers, not a run of the tree.

---

## 8. Gates, risk, sizing and regime (code; a failed gate is a recorded reason, never silence)

**Hard gates (skip, with reason).**
- Liquidity: `adv_20d_dollars` < $5M, price < $5, or median spread > 20 bps.
- Data freshness: quote older than 1 trading day, bars missing more than 2 of the last 10 sessions, fundamentals older than 100 days.
- Red flags from the taxonomy or filings: `going_concern`, `material_weakness` (controls section), `restatement`, `delisting_notice`, an active
  `dilution_offering` with a current-period figure, `debt_to_equity` > 4 with `current_ratio` < 1.
- Event risk: `days_to_next_earnings` ≤ 3 for every strategy except 5.1 (where the event is the point). Without the earnings endpoint, use the last
  earnings press-release date + 91 days as an estimate and say so in `note`.

**Regime (computed once per run from SPY, or QQQ, bars).** `trend`: SPY above/below its 200-day SMA; `stress`: SPY `vol_21d_ann` > 25%;
`recovering`: SPY max drawdown over 63 days > 10% and `ret_21d` > +5%. Rules: in `stress`, halve all sizes and raise the composite threshold;
in `recovering`, disable 5.2 (momentum crash risk) and favour 5.3 and 5.4; below the 200-day, 5.6 and 5.7 need an extra margin. The regime and
the rule applied are in every record.

**Sizing (volatility targeting, whole shares).** Risk budget per trade `r` = 1% of equity (0.5% in stress). Stop distance = 2 × `atr_14` for 5.3
and 5.5, 3 × ATR or the 50-day SMA for 5.2 and 5.4 (whichever is nearer). Shares = ⌊(r × equity) / stop_distance⌋, capped at 20% of equity in one
name and at 5% of `adv_20d_dollars`. For a $1,000 account this yields 0 shares for many names at a 1% budget: the record then says
`size_note: "one share is X% of equity; the risk budget would be exceeded"` and sets `suggested_shares` to 1 with `oversized: true` so the Critic
and the user decide. Never round up silently.

**Exits (so the recommendation is a complete trade, not a tip).** Stop (price), target (expected return × 1.5 as a take-profit level for the
short-horizon strategies; none for 5.4), time stop (horizon end), and **invalidation rules** a program can evaluate later: a list of
`{kind: "price_below" | "event_class" | "metric_crossed" | "days_elapsed", value}`. The Critic's `is_expired` and `is_valid` read these.

---

## 9. The recommendation record (schema v2): what the Critic gets

Keep every v0 field (`symbol`, `company`, `stance`, `model_score`, `score`, `supported`, `consistent`, `signals[]`, `advisor`), so existing tests
and reports stay valid, and add the sections below. Pydantic model in `prometheus/common/` next to the work order; `schema_version: 2` on the order.

| Section | Fields | Who fills it | What the Critic does with it |
|---|---|---|---|
| `identity` | symbol, company, sector, market_cap, as_of_utc, regime | code | staleness, scope |
| `strategy` | primary strategy, all `StrategyResult`s, `conflict`, strategy weights used, `scoring_basis` | code | `is_deal`: does the thesis still hold; which view carried it |
| `thesis` | ≤ 3 sentences; `bull_case[]`, `bear_case[]`, each bullet tied to an evidence id or a metric name | model, checked by code (every number must exist in the record; every claim cites an id) | relevance of evidence; is the bear case taken seriously |
| `evidence` | verified quote signals (as today) + E-items + event classes with their matched text; `counter_evidence` listed separately, never dropped | model proposes, code verifies | re-read quotes; check direction vs class |
| `features` | every metric used: value, z within candidate set, as_of, source, window | Data Clerk (section 3) | recompute spot checks; freshness |
| `expected` | expected_return, horizon_days, basis (`prior` with citation key or `measured` with n and window), win_rate, confidence | code | plausibility; compare to other candidates |
| `trade` | entry_zone (last price ± 0.5 × ATR), stop, target, time_stop_date, suggested_shares, oversized, position_pct, invalidation[] | code | `in_risk_level`, `is_expired` |
| `risk` | vol_63d_ann, atr_pct, beta_63d, max_drawdown_63d, liquidity (adv, spread), event_risk (days_to_earnings), red_flags[] | code | `in_risk_level` against the user's profile |
| `portfolio_fit` | held_already, sector_weight_after, correlation_to_portfolio, funds_feasible, concentration_penalty | code | `find_other_candidates`, diversification |
| `ranking` | composite, risk_charge, crowding_penalty, diversity_adjustment, rank, displaced_by | code | why this one and not that one |
| `gates` | each gate: passed, value, threshold | code | nothing hidden |
| `audit` | metric_requests, model, tokens, dropped, prompt version, strategy module versions | code | reproducibility |

A test asserts that **no field a program could compute is written by the model** (the model's outputs are confined to `thesis`, `bull_case`,
`bear_case`, the proposed quotes, the proposed archetypes and `model_score`), and that every number in the thesis text appears in `features`,
`expected` or `trade`.

---

## 10. The self-learning loop (the capstone's second question)

**Outcome ledger.** Append-only JSONL (`data/ledger/recommendations.jsonl`, later Postgres): the full record at recommendation time, then
forward returns at 5, 20 and 60 trading days (close-to-close and vs SPY), whether the stop or target hit first, and whether any invalidation
rule fired. A small job fills the outcomes from the daily bars (no paid call; bars are already synced monthly). Nothing is ever overwritten.

**What is measured, per strategy and per regime.** Hit rate vs SPY at the strategy's horizon, mean excess return, the calibration of
`confidence` (Brier score: does 0.8 confidence win 80% of the time), the share of recommendations the Critic rejected and why, and the gates
that fired most. Shown in a short report (`scripts/ledger_report.py`), with sample sizes.

**How it changes behaviour (code, bounded).**
- `expected_return` basis flips from `prior` to `measured` for a strategy-regime cell once n ≥ 30, using a shrinkage estimate (the measured mean
  pulled toward the prior with weight 30/(30+n)), so one lucky month does not take over.
- Strategy weights `w_s` are the normalised posterior hit rates (Beta(α, β) per strategy with a prior of 2 wins and 2 losses), re-read at the
  start of each run. Weights are floored at 0.05 so a strategy can recover.
- Thresholds (RSI level, SUE quintile, ATR multiples) are **not** tuned automatically: with tens of samples that is overfitting. They are
  parameters in one config file with the measured table beside them, for Jacob to change.

**Memory for the model.** Before the thesis is written, retrieve the 5 most similar past recommendations (same strategy, same event class or
theme, nearest features) with their outcomes, and put them in the prompt as "what happened the last times this setup fired". This is the memory
tool from the design; it informs the text, never the score.

---

## 11. Evaluation before anyone trusts it

1. **Point-in-time discipline.** Every feature is computed from data with `as_of` ≤ the recommendation time; news and filings by their published
   date; fundamentals by fetch date (note the limitation: Finnhub metrics are "latest", so the backtest of 5.4 and 5.6 is approximate and the
   record says so).
2. **Walk-forward backtest** over the stored bar history for the price-only strategies (5.2, 5.3) and the event strategies where the event date
   is known (5.1, 5.4 announcements, 5.5), on the S&P 500 universe, with costs (10 bps + half the spread), T+1 settlement, whole shares, no
   lookahead. Report hit rate, mean excess return, max drawdown, turnover, and the same for three dumb baselines: buy SPY, buy the 1-month top
   winners ("the obvious run"), random picks from the universe. The system has to beat the second baseline or the anti-chase rule is wrong.
3. **Paper trading** through the Trader when it exists; the ledger is the same.
4. **Ablations**: each strategy alone; composite without diversity; without the regime switch; without the quality guard in 5.3. Keep the table in
   `design/` with the run date.
5. **Stability**: the same order scored 5 times must give the same composite (it is code) and theses that cite the same evidence ids.

---

## 12. Build order for the coding model (each step: tests with fakes, mutation checks, a design-choices entry, docs)

1. **Bars reader + technical metrics** (`technicals.py`): read the stored day aggs for a symbol into a bar series; the Returns/Momentum/Reversal/
   Trend/Volatility/Liquidity/Gap metrics of section 3; SPY series for beta and regime. Fixtures: a small CSV.GZ of a few tickers over 300 days,
   invented. No network.
2. **`calculate_financials` on the work order**: `metrics` field, `verify_metric_request`, `data.metrics`, the catalogue structure, the Data Clerk
   code path (no model loop), desk wiring `advisor_plan -> data_clerk_metrics -> advisor_score`. Fundamentals through the existing `Financials.get`
   behind a fake in tests. NEEDS-YES for a live run.
3. **Event taxonomy** (`events.py`, modelled on `filing_signals.py`): classes, patterns, fixed directions, exact-quote extraction over press
   releases, 8-K items and news summaries; the underreaction test. Fixtures from real stored press releases that are already in `data/press_releases`.
4. **Gates, regime, sizing, exits** (`risk.py`): pure functions; tests for every threshold and for the $1,000 oversize case.
5. **Strategy modules** 5.4 first (its evidence exists), then 5.5, 5.3, 5.2, 5.1 (needs the earnings endpoint: NEEDS-YES), 5.6, 5.7. One file per
   strategy under `prometheus/agents/advisor/strategies/`, one shared interface, a registry.
6. **Composite ranking + diversity + portfolio fit** (`ranking.py`), schema v2 record, the Advisor prompt v2 (archetype proposal; thesis/bull/bear
   from the record with the number check), `desk.report` extended to print strategy, expected, trade and risk per recommendation.
7. **Ledger + outcomes + weights** (`prometheus/memory/ledger.py`, `scripts/ledger_report.py`), the similar-past-recommendations retrieval.
8. **Backtest harness** (`prometheus/evals/backtest.py`) with the three baselines and the ablation table.

Keep Advisor v0 runnable throughout (`Advisor(version=0)` or the v1 falling back to v0 scoring when no metrics are on the order), so the week 3
comparison (`advisor_with_without.py`) still reproduces.

---

## 13. Questions for Jacob (answers change the build, so they are listed, not assumed)

1. Universe for Case 3: S&P 500 only, or NASDAQ-100 too? Is a constituents file in `configs/` acceptable (updated by hand quarterly)?
2. May the day-aggs flat files be synced back 24 months (small), and may the Finnhub earnings and profile endpoints be added to the toolbox?
3. Horizon preference for the first live use: weeks (5.1, 5.3, 5.5) or months (5.2, 5.4)? It sets which strategies ship first.
4. Long-only confirmed? Then "bearish" recommendations are sell/avoid advice for held names only.
5. Whole shares only (no fractional)? The $1,000 sizing note depends on it.
6. Where should the ledger live now: JSONL under `data/` (safe, local) or straight to Postgres from the week 2 design?

---

## 14. References (for the priors; one line each so the coding model can cite them in the record's `basis`)

Bernard & Thomas 1989 (PEAD); Jegadeesh & Titman 1993 (momentum); Jegadeesh 1990 and Lehmann 1990 (short-term reversal); George & Hwang 2004
(52-week high); Daniel & Moskowitz 2016 (momentum crashes); Boudoukh, Michaely, Richardson & Roberts 2007 (net payout yield); Ikenberry,
Lakonishok & Vermaelen 1995 (buybacks); Michaely, Thaler & Womack 1995 (dividend initiations and omissions); Loughran & Ritter 1995 (issuance);
Brav, Jiang, Partnoy & Thomas 2008 (13D activism); Novy-Marx 2013 (gross profitability); Piotroski 2000 (F-score); Ang, Hodrick, Xing & Zhang
2006 and Baker, Bradley & Wurgler 2011 (low volatility); Barber & Odean 2008 (attention); Sloan 1996 (accruals, for a later metric).
