# Advisor v1: final specification of the "Generate Recommendations" step

Jacob Pyrett, CMU Agentic AI capstone, 2026-11-05. This document supersedes and combines `design/advisor_strategy_handoff.md`,
`design/advisor_summary_handoff.md`, `design/advisor_strategy_expansion.md` and `design/advisor_grok_expanded_summary.md`.
Where they disagree, this document wins. It is written for the coding model (Claude Sonnet 5.5) and it is a design, not code.

**Scope.** One step of the trading desk: the Advisor turns a work order whose data is ready into a ranked, top-k list of
recommendation records that carry everything a Critic needs. The Critic itself, the Trader, and the rest of the LangGraph wiring are
out of scope (Jacob is building them). Nothing in this document starts a live, paid call.

---

## 0. How the coding model must use this document

1. Read, in this order, before writing anything: `AGENTS.md`, `README.md`, `HowToRun.txt`, `design/jpyrett_capstone_tradingbot.pdf`,
   `docs/design_choices.md` entries 8, 14, 24, 25, 26, 38, 39, 40, 41, 42, then these files in full:
   `prometheus/common/work_order.py`, `prometheus/agents/advisor/advisor.py`, `prometheus/prompts/advisor.py`,
   `prometheus/agents/data_clerk/data_clerk_agent.py` (at least `request_data`, `request_quotes`), `prometheus/tools/data_clerk.py`,
   `prometheus/agents/data_clerk/filing_signals.py`, `prometheus/agents/data_clerk/data_toolbox.py` (`SCHEMAS`, `performance_summary`,
   `Financials`), `prometheus/common/portfolio.py`, `prometheus/models/chat.py`, `tests/test_advisor.py`, `tests/test_filing_signals.py`.
2. Follow the patterns of that code: plain Python functions for arithmetic, Pydantic models for shared objects, a `today: date | None`
   argument wherever a date matters, `check_symbol` for every ticker, `get_chat_model("advisor")` for the model, error messages that
   name a setting and never its value, `None` plus a `note` for a number that cannot be computed, and tests that use fakes.
3. **MUST / NEVER** in this document are rules. **ASK JACOB** means stop and ask; do not pick a default. **NEEDS-YES** marks a data
   fetch or download that may not be run without his yes. **PHASE n** marks the build step a thing belongs to (section 16).
4. When a value is not in the data (a Finnhub field missing, fewer bars than an indicator needs), return `None` with a `note`.
   NEVER substitute a guess, a default price, or a value from the model.
5. NEVER let a model write a number that is used anywhere. The model's outputs are listed in section 11; everything else is code.
6. Do not modify Advisor v0 (`advisor.py`, `prompts/advisor.py`) except where section 16 says so. The week 3 comparison
   (`labs/module3_rag/advisor_with_without.py`) must keep running unchanged.
7. Every phase ends with: tests passing (`~/miniconda3/envs/cmu/bin/python -m pytest -q --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo`),
   a mutation check (break one rule, see a test fail, restore), a new numbered entry at the end of `docs/design_choices.md`, the docs
   listed in AGENTS.md updated, and no `__pycache__` or `.pytest_cache` left behind.

---

## 1. The step in one paragraph

Input: a `WorkOrder` with `status == "data_ready"` (news, press releases, filings, filing evidence on `data.*`), the user's `Portfolio`,
the `case` (`funds`, `theme`, or `market`) and `today`. The Advisor (1) collects candidate symbols; (2) makes ONE model call that
proposes, per candidate, which strategy archetypes the text supports, with exact quotes (checked by code as v0 does) and the corporate
events it sees (checked by code against the event taxonomy); (3) asks the Data Clerk, with ONE `calculate_financials` work order, for the
metrics the eligible strategy families need; (4) runs the gates; (5) scores each family in code; (6) applies the quality multiplier and
the modifiers; (7) turns each family's score into a 20-trading-day excess return over SPY after cost, one vote per family; (8) reviews what
is already held (hold, add, reduce) and applies the cooldown; (9) ranks, with the sleeve rules of the case and a diversity rule; (10)
writes the trade plan (entry zone, stop, target, size, order style, invalidation rules); (11) makes a SECOND model call that writes a
three-sentence thesis and bull/bear bullets from the finished record, checked by code so that every number and every reference in it
exists in the record; (12) appends the records to `order.recommendations`, the losers to `order.candidates`, a run record to
`order.advisor_runs`, and the records to the ledger. Output: the same order with `status == "recommended"`.

---

## 2. Data that exists today, and what is missing

| Data | Where | On the order? | Notes |
|---|---|---|---|
| News chunks with Polygon per-ticker insights | `data.sources` (numbered), `data.results.news` | yes | `format_source` shows them to the model |
| Press releases, SEC filing sections | `data.sources`, `data.results.press_releases`, `data.results.filings` | yes | the company's own words, half weight (section 8.2) |
| Filing evidence | `data.filing_evidence[symbol]` (balance sheet both periods, capital actions, income headlines, E-items by topic, gaps, digest) | yes | written by `filing_signals.recommendation_evidence`, scores nothing |
| Quote | `data.quotes[symbol]` on a `fetch_quote` order (`DataClerk.request_quotes`) | only on that order | `price, change, high, low, open, previous_close, as_of_utc, source` |
| Ten fundamentals | `DataClerkTools.get_financials` → `Financials.get` (Finnhub `stock/metric`, `metric=all`) | no | the raw payload holds more fields than the ten kept; section 5.4 names the extra ones to keep |
| Daily bars | `data/historical/day_aggs_v1/2026/10/*.csv.gz`, columns `ticker, volume, open, close, high, low, window_start (ns epoch UTC), transactions`, one file per trading day, all US tickers | no | **Only October 2026 is on disk today (17 trading days).** Every indicator that needs more bars returns `None` until more months are synced (NEEDS-YES, section 2.1) |
| Minute bars, trades, NBBO quotes | `minute_aggs_v1`, `trades_v1`, `quotes_v1`, October 2026 | no | section 5.6, confirmation only |
| Theme discovery | `DataClerkTools.discover_tickers(phrase)` over the market news store | no | `{tickers: [{symbol, articles, positive, neutral, negative, providers, headlines}]}` |
| Portfolio | `common/portfolio.py`, `configs/portfolios/*.json` | passed to the Advisor as an argument | `cash_avail`, positions with `shares, cost_basis, acquired, recent_shares`, `sellable_shares(symbol, today)` |

### 2.1 Missing data (each NEEDS-YES; the code MUST work with it absent)

| Need | Source | Used by | Without it |
|---|---|---|---|
| Daily bars back at least 13 months (24 preferred), for all tickers | `aws s3 sync` of `day_aggs_v1` for earlier months (small: ~320 KB per day) | momentum, 52-week high, SMA 200, beta, breadth, backtest | long-window metrics are `None`; the trend family is `not_computable`; reversion and event families still run on 17+ bars for the short indicators that fit |
| Earnings history and calendar (actual, estimate, surprise, next date) | Finnhub earnings endpoints | PEAD, earnings-gap sizing, event-risk gate | `sue_4q` and `days_to_next_earnings` are `None`; the event-risk gate uses "last earnings press release date + 91 days" and says so in `note` |
| Company profile (sector/industry, market cap, shares outstanding) | Finnhub `stock/profile2`; market cap is also in the `stock/metric` payload | sector residuals, diversity rule, net payout yield | `sector` may be passed on the work order by the user; else residual metrics are `None` and the diversity rule falls back to correlation only |
| Index constituents (S&P 500, NASDAQ-100) | a hand-maintained file `configs/universe_sp500.json` (ASK JACOB for the source) | Case 3 universe, cross-sectional ranks, breadth | the universe is the candidate list; threshold scoring only |
| Insider transactions, EPS revisions, short interest, related companies | Finnhub / Polygon endpoints | the bench modifiers of section 9.4 | the modifiers are absent; nothing else changes |

---

## 3. Work order changes (`prometheus/common/work_order.py`) — PHASE 2

Add, without changing any existing behaviour or message:

| Field | Type | Meaning |
|---|---|---|
| `WorkOrder.metrics` | `list[dict] \| None` | on a `calculate_financials` order: the metric requests, each `{"name": str, "params": dict}` (params optional) |
| `WorkOrder.case` | `Literal["funds", "theme", "market"] \| None` | which use case the order serves (section 12). `None` → the Advisor infers: symbols as subject → `funds`; a term as subject → `theme` |
| `WorkOrder.sector` | `str \| None` | optional sector ETF ticker the user supplies (one of the keys of `configs/sector_etfs.json`) until the profile endpoint exists |
| `WorkOrder.candidates` | `list[dict]` | the symbols that reached scoring and were not returned as recommendations, with the gate or rank that stopped them (section 10.3) |
| `DataResult.metrics` | `dict[str, dict]` | per symbol, per metric name: the metric result of section 4.2 |
| `DataResult.market_brief` | `dict` | Case 3 only (section 13); empty otherwise |
| `METRIC_ACTION = "calculate_financials"`, `MAX_METRIC_SYMBOLS = 30`, `MAX_METRICS = 40` | constants | |
| `verify_metric_request(order) -> list[str]` | function | the reasons the Data Clerk may NOT answer a metric order; `[]` means it may. Same style as `verify_quote_request`: `action_type` must be `calculate_financials`; `subject` 1 to 30 checked symbols; `metrics` a non-empty list of at most 40 items whose `name` is in the catalogue (the message lists the unknown names and says "known metrics: …"); `status` not already `data_ready`/`recommended` |

`verify_request` keeps refusing `calculate_financials` (it is the check of a SEARCH, entry 25). Bump `schema_version` to 2 on an order that
carries any of the new fields; tests that build v1 orders keep passing.

---

## 4. The Data Clerk calculates metrics — PHASE 1 and PHASE 2

### 4.1 The call

`DataClerk.request_metrics(order, today=None) -> WorkOrder`, in `data_clerk_agent.py`, next to `request_quotes` and shaped like it:
no model call; `verify_metric_request`; then `self.tools.calculate_metrics(symbols, metrics, today)`; the result on `order.data.metrics`;
each symbol's failures named in `order.errors`; `status` `data_ready` when every requested symbol has a result object (a metric that is
`None` with a note is still a result), `data_failed` when a symbol could not be read at all. `tool_calls` records one `calculate_metrics`
entry with the symbol count and metric count.

`DataClerkTools.calculate_metrics(symbols, metrics, today=None) -> dict` in `tools/data_clerk.py`, added to `DATA_CLERK_TOOLS` so the
MCP server and the LangChain tools see it. Docstring for the model: what it computes, that it only reads stored bars and the Finnhub
metric payload, and that it rates nothing. Argument checks: `_symbols` (raise `MAX_SYMBOLS` for the model's own use to stay at 10; the
Advisor's code path passes up to `MAX_METRIC_SYMBOLS` through `request_metrics`, not through the model tool), metric names checked
against the catalogue with `ToolInputError`.

### 4.2 The result object, per symbol, per metric

```
{"value": float | bool | None, "unit": str, "as_of": "YYYY-MM-DD", "window": str, "source": str, "inputs_summary": dict, "note": str | None}
```

- `source` is one of `day_aggs:<first_date>..<last_date>`, `finnhub_metric:<fetch_date>`, `filing:<accession>`, `derived:<names of the inputs>`.
- `inputs_summary` holds what a Critic needs to recompute: for `rsi_14` `{"bars": 45, "last_close": 101.2}`; for `pe_ratio` `{"price": ..., "eps_ttm": ...}`.
- `note` is a gap in words, never a judgement.
- A test walks every metric result and refuses the keys `score, stance, sentiment, recommendation, rating, verdict` (the `filing_signals.FORBIDDEN_KEYS` test, reused).

### 4.3 Where the code lives

| Module | Content |
|---|---|
| `prometheus/agents/data_clerk/bars.py` | `Bar` dataclass `{date, open, high, low, close, volume, transactions}`; `BarStore(root)`: reads the `day_aggs_v1` files under `data/historical` ONCE per run for a requested set of tickers (always add `SPY` and the ETFs of `configs/sector_etfs.json`), keeps them in memory, exposes `bars(symbol, end=today, max_bars=300) -> list[Bar]` sorted by date; `window_start` (ns since epoch, UTC) → the trading date; a day file that is missing is skipped and counted in `BarStore.gaps`. No writes to `data/historical`, ever. Tests use a `tmp_path` with generated files in the same format (section 15.1) |
| `prometheus/agents/data_clerk/technicals.py` | pure functions over lists of bars: every indicator of section 5.1 to 5.3. Each declares `MIN_BARS` and returns `None` below it |
| `prometheus/agents/data_clerk/fundamentals.py` | ratios from the Finnhub payload and from `filing_evidence` (section 5.4); `filing_signals.py` is extended with the extra statement rows of section 5.5 (keep its no-score test) |
| `prometheus/agents/data_clerk/metrics_catalog.py` | `METRICS: dict[str, MetricSpec]`, `MetricSpec = {name, group, inputs: ("bars" \| "financials" \| "evidence" \| "universe"), unit, min_bars, description, fn}`; `compute(symbols, requests, bars_store, financials_by_symbol, evidence_by_symbol, universe, today) -> dict[symbol][name] -> result`. Adding a metric = one entry + one test |
| `prometheus/agents/data_clerk/market.py` | `market_brief(...)` for Case 3 (section 13), PHASE 8 |

`Financials.get` is called at most once per symbol per run (cache inside the run) and is NEEDS-YES in a live run; in tests it is a fake.
Keep the extra payload fields section 5.4 lists by adding them to `FINANCIAL_METRICS` (entry 8 allows Finnhub's precomputed fields).

---

## 5. The metric catalogue (version 1)

"Bars" means daily bars ending at `today` (the last stored session on or before it). N bars means N trading sessions, not calendar days.
`MIN_BARS` is the number below which the metric is `None` with the note `"needs N bars, M stored"`. Returns are simple, close-to-close,
unless stated. Every indicator is a FEATURE of a family (section 8); none is a vote by itself.

### 5.1 Price, return, trend

| Name | Definition | MIN_BARS |
|---|---|---|
| `ret_1d, ret_5d, ret_21d, ret_63d, ret_126d, ret_252d` | `close[-1] / close[-1-N] - 1` | N+1 |
| `mom_12_1` | `close[-21] / close[-252] - 1` (12 months, skipping the latest month) | 252 |
| `pct_of_52w_high` | `close[-1] / max(high over the last 252 bars)` | 252 |
| `sma_50, sma_200` | simple mean of the last N closes | N |
| `above_sma_50, above_sma_200` | `close[-1] > sma_N` (bool) | N |
| `breakout_20d` | `close[-1] >= max(high over bars -21..-2)` (bool) | 21 |
| `gap_1d` | `open[-1] / close[-2] - 1` | 2 |
| `candle_reversal_1d` | `close[-1] > open[-1]` and `low[-1] < low[-2]` (bool: sold off below yesterday's low and closed up) | 2 |

### 5.2 Oscillators and trend strength (the indicators Jacob asked for)

| Name | Definition | MIN_BARS | Enters as |
|---|---|---|---|
| `rsi_14` | Wilder RSI: first average gain/loss = simple mean of the first 14 changes, then `avg = (prev*13 + current)/14`; `RSI = 100 - 100/(1 + avg_gain/avg_loss)`; 100 when `avg_loss == 0` | 30 | reversion (≤ 30), anti-chase in trend (≥ 75 blocks) |
| `macd_hist` | `EMA12(close) - EMA26(close)` minus its `EMA9`; EMAs seeded with the simple mean of the first period | 60 | trend confirmation (sign) |
| `bollinger_pct_b` | `(close - lower) / (upper - lower)`, bands = `SMA20 ± 2 × std20` (population std) | 20 | reversion (≤ 0), trend (0.8 to 1.0 fine, > 1.05 is stretched) |
| `zscore_21d` | `(close[-1] - mean(close, 21)) / std(close, 21)` (sample std) | 21 | reversion (≤ −2) |
| `adx_14` | Wilder ADX from +DI/−DI with 14-period smoothing | 40 | trend confirmation (≥ 20 = trending) |
| `atr_14`, `atr_pct` | Wilder ATR of true range; `atr_pct = atr_14 / close[-1]` | 20 | stops, sizing, underreaction |
| `obv_20d_change` | `(OBV[-1] - OBV[-21]) / sum(volume, last 20)`, OBV = cumulative signed volume by close direction; result in [−1, 1] | 21 | trend confirmation (> 0) |
| `stoch_k_14` | `(close - min(low,14)) / (max(high,14) - min(low,14)) × 100` | 14 | reversion (≤ 20), recorded only |

### 5.3 Volatility, risk, liquidity, market

| Name | Definition | MIN_BARS |
|---|---|---|
| `vol_21d_ann, vol_63d_ann` | sample std of daily log returns × √252 | N+1 |
| `max_drawdown_63d` | min over t of `close[t] / max(close[..t]) - 1` within the last 63 bars (negative number) | 63 |
| `beta_63d` | OLS slope of the symbol's daily returns on SPY's, last 63 | 64 (both series) |
| `corr_63d_spy` | Pearson correlation of the same series | 64 |
| `adv_20d_dollars` | mean of `close × volume` over 20 bars | 20 |
| `volume_ratio_1d` | `volume[-1] / mean(volume, bars -21..-2)` | 21 |
| `sector_etf` | the ETF ticker from `configs/sector_etfs.json` for the order's `sector` (or the profile's industry when that endpoint exists); `None` when unknown | – |
| `sector_ret_5d, sector_ret_21d, sector_ret_63d` | the ETF's returns | N+1 |
| `residual_5d, residual_21d, residual_mom_12_1` | the symbol's return minus the ETF's over the same window; `None` without `sector_etf` | as the base |
| `spy_ret_5d, spy_ret_21d, spy_above_sma_200, spy_vol_21d_ann, spy_drawdown_63d` | the same metrics on SPY, computed once per run | as above |
| `breadth_200` | share of the universe's symbols with `above_sma_200 == True`, over the symbols whose bars suffice; `None` when fewer than 50 symbols can be computed | 200 per symbol |
| `earnings_move_abs` | median of `abs(ret_1d)` on the last four earnings dates (NEEDS-YES for the dates) | – |

### 5.4 Fundamentals (Finnhub `stock/metric`, fetched by `Financials.get`)

Keep the ten existing names. Add to `FINANCIAL_METRICS`, keeping Finnhub's own values: `market_cap` (`marketCapitalization`, in millions,
convert to dollars and say so in `unit`), `shares_outstanding` (if present), `beta_provider` (`beta`), `high_52w` (`52WeekHigh`),
`low_52w` (`52WeekLow`), `adv_10d_provider` (`10DayAverageTradingVolume`), `ps_ratio` (`psTTM`), `pb_ratio` (`pbQuarterly` or `pbAnnual`),
`ev_to_ebitda` if present, `payout_ratio_provider` (`payoutRatioTTM`) if present, `eps_growth_ttm` if present. A field Finnhub does not return is `None`.
Derived, in `fundamentals.py`: `earnings_yield = 1 / pe_ratio` when `pe_ratio > 0`, else `None`.

### 5.5 From the filing evidence (`filing_signals.py`, extended)

Add to `BALANCE_ITEMS`: `accounts_receivable` (`accounts receivable, net|receivables`), `inventories` (`inventories|inventory`).
Add to `CASH_FLOW_ITEMS`: `cash_from_operations` (`cash generated by operating activities|net cash provided by operating activities`), topic `None`.
Keep the module's rule: it stores figures and quotes, no score, no derived judgement. The derived numbers below live in `fundamentals.py`:

| Name | Definition | `None` when |
|---|---|---|
| `net_payout_yield_ttm` | `(abs(repurchases) + abs(dividends paid) − proceeds from issuing common stock) / market_cap`, using the current-period cash-flow values (which are year-to-date in a 10-Q: annualize by `× 12 / months in the period`, with `inputs_summary.months`) | any input missing |
| `share_count_change` | `balance_sheet.shares.change_pct / 100` (already there) | no shares row |
| `payout_coverage` | `abs(dividends paid) / net income` for the same period | either missing or net income ≤ 0 → `None` with note |
| `sbc_to_revenue` | `share-based compensation expense / revenue` same period | missing |
| `accrual_ratio` | `(net income − cash from operations) / total assets` | missing |
| `receivables_vs_sales` | `change_pct(receivables) − change_pct(revenue)` in percentage points, current vs prior period | missing |
| `inventory_vs_sales` | the same with inventories | missing |
| `cfo_negative_with_profit` | `cash_from_operations < 0 and net income > 0` (bool) | missing |
| `dilution_current` | `True` when any `dilution` E-item is `derived` with a non-zero current value, or a `dilution` passage comes from an 8-K or a 10-Q filed within 120 days | – |
| `risk_factor_change` | `1 − similarity` between the `risk_factors` sections of the two latest stored 10-Ks of the symbol; similarity = cosine of the RAG embedding of each whole section (the same `all-MiniLM-L6-v2`; chunk and average when the section is long), with token Jaccard as the fallback and `inputs_summary.method` saying which | fewer than two stored 10-Ks with that section |

### 5.6 Intraday confirmation (optional, PHASE 10, current month only)

`median_spread_bps_1d` from `quotes_v1` for the latest stored day (median of `(ask − bid) / mid × 10000` over the regular session),
`large_print_share_1d` from `trades_v1` (share of volume in prints of $100,000 or more). Both carry `inputs_summary.window = "current month only"`.
They may move `agreement` by at most ±0.10 (section 9.2) and the spread feeds the cost (section 9.5). They never add a symbol.

---

## 6. Advisor v1: module layout — PHASE 3 onward

```
prometheus/agents/advisor/
  advisor.py            v0, unchanged (the week 3 baseline)
  advisor_v1.py         AdvisorV1: recommend(order, portfolio, case, today, clerk) -> WorkOrder   (the step, sections 7 to 11)
  candidates.py         collect_candidates(order, portfolio, case) -> list[Candidate]
  events.py             EVENT_CLASSES, match_events(text) -> list[EventMatch], verify_event(signal, sources) (section 8.4)
  gates.py              run_gates(symbol, metrics, evidence, events, today, config) -> GateResult
  quality.py            quality_multiplier(metrics, evidence, config) -> (float, list[Haircut])
  families/__init__.py  FAMILIES registry: name -> module with score(...) -> FamilyResult | None
  families/event.py     families/trend.py  families/reversion.py  families/capital_return.py  families/theme.py
  ranking.py            family votes, composite, after-cost excess, holdings pass, cooldown, sleeves, diversity, top-k
  trade_plan.py         entry zone, stop, target, time stop, size, order style, invalidation rules
  record.py             Pydantic: Recommendation (schema 2), Candidate, FamilyResult, GateResult, TradePlan, ThesisText
  thesis.py             the second model call and its checks
  config.py             loads configs/advisor_v1.json (every threshold in this document), validated once
prometheus/prompts/advisor_v1.py     PLAN_SYSTEM, THESIS_SYSTEM, builders, RETRY texts
prometheus/memory/ledger.py          append(record), recent(symbol, family, since) over data/ledger/recommendations.jsonl
configs/advisor_v1.json              thresholds, priors, regime multipliers, family weights
configs/sector_etfs.json             {"XLK": {"name": "Technology", "industries": [...]}, ...} for the 11 SPDR sectors
configs/universe_sp500.json          ASK JACOB for the source; absent is allowed
```

`AdvisorV1.__init__(model=None, model_id=None, clerk=None, config_path=None, ledger=None)`: `model` as v0 (anything with `invoke`;
scripted in tests); `clerk` is the `DataClerk` whose `request_metrics` is called (a fake in tests); `ledger` a `Ledger` or a fake.

The desk node for this step (Jacob wires the rest): `advisor_v1 = AdvisorV1(...).recommend(order, portfolio, case, today)`; inside it the
metric request is a sub-step, not a separate graph node, so one node reads and returns one order. `order.advisor_runs` gets one entry per
model call with `{"stage": "plan" | "thesis", "model", "tokens_in", "tokens_out", "calls", "dropped", "error"}`.

---

## 7. Step by step: `AdvisorV1.recommend`

Each step names its function, its inputs, its output, and what happens when something is missing. No step calls the network except
the Data Clerk's `request_metrics` (which may call Finnhub's metric endpoint: NEEDS-YES in a live run).

### Step 1. Preconditions
`order.status == "data_ready"`, else append `"the Advisor needs an order whose data is ready (status X)"` to `order.errors` and return
(v0's wording). `case` from the argument, else `order.case`, else inferred (section 3). `today` from the argument, else `date.today()`.

### Step 2. Collect candidates (`candidates.collect_candidates`)
Union, in this order, de-duplicated, every symbol through `check_symbol`:
1. the order's subject symbols;
2. every symbol in `order.data.sources[*].tickers` that is ALSO named (ticker as a word, or a distinctive name word: reuse `names_company`)
   in the text of at least one source (a ticker list alone does not qualify: entry 26);
3. `case == "theme"`: `discover_tickers(order.query or subject term)` results with `articles >= 3` (the Advisor calls the Data Clerk's
   tool method directly; it reads stored market news; no fetch);
4. the portfolio's symbols, tagged `held: true` (for the holdings pass; they are not new ideas);
5. `case == "market"`: the symbols the market brief lists (section 13).
Cap new ideas at `max_candidates` (15) by: subject first, then by the number of sources naming them, then alphabetically; record the
cut in `advisor_runs[...].dropped`. Each `Candidate = {symbol, company (from the order or the source metadata), origin: "subject" | "sources" | "discover" | "held" | "market", held: bool}`.

### Step 3. Plan call (model call #1) — `prompts/advisor_v1.PLAN_SYSTEM`
Input text, built as v0 builds it: the work order head, `kinds_text`, the numbered sources (`format_source`, same as the Data Clerk's model saw),
`evidence_text`, plus the candidate list with their origins, the list of archetype names (section 8.1), the list of event class names
(section 8.4) and the rules. Output JSON, exactly:

```
{"companies": [{"symbol": "MU", "company": "Micron", "archetypes": ["trend", "event"],
                "signals": [{"source": 3, "direction": "bullish", "quote": "words copied exactly"}],
                "events":  [{"class": "guidance_raise", "source": 5, "quote": "words copied exactly"}],
                "drop_cause": {"source": 2, "quote": "...", "class": "sector_sympathy"} | null,
                "model_score": 65}]}
```

Code then: `parse_reply` as v0 (one retry with `RETRY`); each signal through v0's `verify_signal` / `verify_evidence_signal` and the
once-per-story rule (import them from `advisor.py`; do not copy); each event through `events.verify_event` (section 8.4); `archetypes`
filtered to known names; `drop_cause` kept only when its quote verifies. A symbol the model names that is not a candidate is added with
origin `"model"` only if it has at least one verified signal, else dropped with the reason. Nothing numeric from this call is used except
`model_score`, which is recorded.

### Step 4. Metric request (`AdvisorV1._metric_order`)
Build ONE `calculate_financials` work order (`new_work_order(query=f"metrics for {order.request_id}", action_type="calculate_financials",
subject=symbols, requester="advisor")`, `metrics` = the union of what every family that is eligible for any candidate needs, from
`families/*.REQUIRED` and `OPTIONAL`, plus the gate metrics and SPY/sector metrics). Call `clerk.request_metrics`. Copy
`metric_order.data.metrics` into `order.data.metrics` and the metric order's `request_id` into `advisor_runs[-1].metric_order`.
A `data_failed` metric order does not stop the step: the symbols that have results are scored, the others become candidates with
`gate: "no_metrics"`.

### Step 5. Gates (`gates.run_gates`, section 9.1)
Per candidate. A failed gate is a `Candidate` with `gate` and `gate_value`; held symbols that fail a gate are still reviewed in Step 8
(they may need a `reduce`).

### Step 6. Family scores (`families/*.score`)
Each family returns `FamilyResult` or `None` (not computable: the result says `missing: [...]`). Section 8 defines each family.

### Step 7. Quality and modifiers (`quality.quality_multiplier`, section 9.3; modifiers 9.4)

### Step 8. Votes, composite, cost, holdings, cooldown (`ranking.*`, sections 9.5 to 9.8)

### Step 9. Sleeve and top-k (`ranking.select`, section 12)

### Step 10. Trade plan (`trade_plan.build`, section 10)

### Step 11. Thesis call (model call #2) and checks (`thesis.write`, section 11)

### Step 12. Write out
`order.recommendations.extend(records as dicts)`, `order.candidates = the rest`, `order.status = "recommended"`, `ledger.append(record)` for
each record (section 14), return the order. Every record also carries `schema_version: 2`.

---

## 8. Strategy families

A family has ONE vote per symbol. Inside a family, correlated features are combined into one score. Every family declares
`REQUIRED` metrics (missing → `None` result with `missing`) and `OPTIONAL` ones (missing → lower `completeness`). All thresholds are in
`configs/advisor_v1.json`; the numbers below are the initial values.

Two scoring paths, set by the size of the comparable set (`scoring_basis` on the result):
- `threshold` (default; used when fewer than 30 symbols have the metric): fixed cut-offs as written below;
- `rank` (Case 3 with a universe of 30+): the percentile of the metric within the universe replaces the cut-off where the text says "(rank: …)".

### 8.1 Archetypes the model may propose
`event`, `trend`, `reversion`, `capital_return`, `theme`. The model proposing an archetype only makes that family RUN for the symbol; the
family's own rules decide the score. Families also run without a proposal when their REQUIRED metrics pass (the model cannot hide a family),
except `theme`, which needs the theme context.

### 8.2 Text evidence shared by all families
From Step 3's verified signals and events. `agreement = (bullish_weight + 1) / (bullish_weight + bearish_weight + 2)` where a counted news
signal weighs 1.0, a counted press-release or filing signal or E-item weighs 0.5. `evidence_n` = number of counted signals and events.
A family that is bearish by its own rule uses `1 − agreement`.

### 8.3 `trend` (momentum with a quality filter, anti-chase)
REQUIRED: `mom_12_1, pct_of_52w_high, above_sma_200, ret_21d, rsi_14, atr_pct`. OPTIONAL: `residual_mom_12_1, macd_hist, adx_14, obv_20d_change, vol_63d_ann, breakout_20d`.
Pass when ALL: `mom_12_1 ≥ 0.15` (rank: top 20%); `pct_of_52w_high ≥ 0.90`; `above_sma_200`; `ret_21d ≤ 0.15` (rank: not top 10%) — the anti-chase rule;
`rsi_14 < 75`; `residual_mom_12_1 > 0` when available; `bollinger_pct_b ≤ 1.05` when available.
`score = 0.40 + 0.20 × clip(mom_12_1 / 0.50, 0, 1) + 0.10 × [macd_hist > 0] + 0.10 × [adx_14 ≥ 20] + 0.10 × [obv_20d_change > 0] + 0.10 × [vol_63d_ann < 0.35]` (a missing optional term contributes 0 and lowers completeness).
Direction: bullish only. Horizon 63 bars. Prior 20-day excess: 0.6%. Off in regime `recovering`; halved in `narrow` (section 9.6).
Today, with 17 bars on disk, this family is `None` with `missing: ["mom_12_1", ...]` for every symbol. That is correct behaviour.

### 8.4 `event` (corporate events and post-earnings drift)
REQUIRED: `atr_pct, ret_1d, volume_ratio_1d` and at least one verified event (from Step 3) or `sue_4q` (NEEDS-YES). OPTIONAL: `gap_1d, sue_4q, eps_surprise_last, market_cap`.

Event taxonomy (`events.py`, modelled on `filing_signals.TOPIC_PATTERNS`: regex per class, exact-quote extraction, the model may only cite a
class whose pattern the code also matches in that quote's source text; direction is fixed by the class and checked):

| Class | Direction | Horizon (bars) | Ship now? |
|---|---|---|---|
| `guidance_raise`, `earnings_beat`, `buyback_authorization`, `dividend_increase`, `dividend_initiation`, `activist_stake_13d` | bullish | 40 | yes |
| `guidance_cut`, `earnings_miss`, `dividend_cut`, `dilution_offering`, `ceo_departure_unplanned`, `material_weakness`, `restatement`, `going_concern`, `form_144_sale_notice` | bearish | 40 | yes |
| `analyst_rating_change`, `sector_sympathy`, `macro`, `lawsuit_headline`, `index_flow` | neutral (used by `reversion` as a non-fundamental cause) | – | yes |
| `fda_approval`, `index_inclusion`, `major_contract_award`, `delisting_notice`, `acquisition_target`, `acquirer_large_deal` | as named | 40 | parked: add when a stored record shows the pattern |

`activist_stake_13d` and `form_144_sale_notice` come from the filings index rows (form `SC 13D`, `144`) with the filing date; no text needed.
Event age = bars since the source's `published_utc` (or filing date); only events with age ≤ 10 bars count. Reaction so far
`reaction = (close[-1] / close on the bar before the event − 1)`; `reaction_atr = abs(reaction) / atr_pct`.
Pass when a bullish class fired and `reaction_atr ≤ 3` (above 3 the move is priced: result `None` with note), or a bearish class fired (always
passes, as a bearish vote), or `abs(sue_4q) ≥ 1.0` with the announcement-day return of the same sign and age ≤ 10.
`score = 0.50 + 0.30 × (1 − clip(reaction_atr / 3, 0, 1)) + 0.20 × [volume_ratio_1d on the event bar ≥ 2]`.
PEAD haircuts: `× 0.5` when `market_cap > 200e9`; `× 0.7` when the last four surprises are all positive and each smaller than the last.
Direction by class (a bullish and a bearish class on the same symbol inside 10 bars → the family's vote is the one with the newer event, and
`contradictions` records both). Prior 20-day excess: 1.0% for a surprise, 0.8% for a taxonomy event; bearish classes use the same magnitudes negative.

### 8.5 `reversion` (quality on sale)
REQUIRED: `ret_5d, zscore_21d, rsi_14, atr_pct, roe, debt_to_equity, current_ratio, revenue_growth`. OPTIONAL: `residual_5d, sector_ret_5d, spy_ret_5d, above_sma_200, above_sma_50, bollinger_pct_b, volume_ratio_1d, gap_1d, candle_reversal_1d, stoch_k_14`.
Pass when ALL:
- oversold: `ret_5d ≤ −0.07` or `zscore_21d ≤ −2` or `rsi_14 ≤ 30`;
- idiosyncratic: `residual_5d ≤ −0.05 and sector_ret_5d > −0.03` when the sector is known; else `ret_5d ≤ −0.07 and spy_ret_5d > −0.03`;
- sound: `roe > 10`, `debt_to_equity < 1.5` *(waived for negative stockholders' equity: Amendment 1)*, `current_ratio > 1`, `revenue_growth > 0`;
- in an uptrend: `above_sma_200` when computable, else `above_sma_50` when computable, else no requirement and note `"trend filter skipped: N bars"`;
- cause: Step 3's `drop_cause.class` is one of the neutral classes, or no company-specific source in the last 5 bars (sympathy); a fundamental
  bearish class (`guidance_cut`, `dilution_offering`, `restatement`, `going_concern`, `material_weakness`, `earnings_miss`) → `None` with note `"fundamental cause"`;
- no haircut from section 9.3's accrual/receivables/CFO rules (they turn this family off).
`score = 0.40 + 0.20 × clip((−zscore_21d − 2) / 2, 0, 1) + 0.10 × [rsi_14 ≤ 30] + 0.10 × [bollinger_pct_b ≤ 0] + 0.10 × [volume_ratio_1d ≥ 2] + 0.10 × [candle_reversal_1d]`.
Direction bullish. Horizon 10 bars. Prior 20-day excess: 1.0% (the edge is realised inside 10 bars; ranking uses it as is).

### 8.6 `capital_return`
REQUIRED: `share_count_change` or `net_payout_yield_ttm` (at least one), `earnings_yield`, `debt_to_equity`, `dilution_current`. OPTIONAL: `payout_coverage, revenue_growth, net_payout_yield_ttm, sbc_to_revenue, market_cap`.
Pass when ALL: `net_payout_yield_ttm ≥ 0.03` or (`net_payout_yield_ttm is None` and `share_count_change ≤ −0.005`); `share_count_change ≤ 0`;
`dilution_current == False`; `debt_to_equity ≤ 2` *(waived for negative stockholders' equity: Amendment 1)*; `earnings_yield > 0`; `payout_coverage < 1.0` when available; `revenue_growth ≥ −5` when available.
`score = 0.40 + 0.30 × clip(net_payout_yield_ttm / 0.08, 0, 1) + 0.10 × [share_count_change < −0.01] + 0.10 × [a buyback_authorization or dividend_increase event within 60 bars] + 0.10 × [revenue_growth > 0]`
(with `net_payout_yield_ttm` missing, the 0.30 term becomes `0.15 × [share_count_change < −0.01]` and completeness drops).
Funded check: when a `buyback_authorization` event states a dollar amount (the regex captures `\$[\d.,]+ (billion|million)`) and the balance
sheet cash is below it, that event's 0.10 term is halved and `note` says `"authorization exceeds cash"`.
Direction bullish. Horizon 126 bars. Prior 20-day excess: 0.3%.
A `dilution` E-item that passes the Advisor's check, or `dilution_current`, is a BEARISH vote of this family with `score = 0.6`, prior −0.3%.

### 8.7 `theme` (Case 2 only)
REQUIRED: the theme's `discover` result; `ret_21d`; at least one verified signal naming the company in the theme's sources. OPTIONAL: `residual_21d, volume_ratio_1d`, quality inputs.
Theme basket = the discover tickers with `articles ≥ 3`; `basket_median_ret_21d` over the members whose bars suffice.
Pass when: `articles ≥ 3`; `positive / (positive + negative + neutral) ≥ 0.6`; `ret_21d ≤ basket_median_ret_21d` (the laggard rule);
quality multiplier ≥ 0.5. Members that fail only the laggard rule are kept as candidates with `theme_role: "leader"` and the reason
`"already moved {ret_21d:.0%} in 21 bars"`.
`score = 0.40 + 0.20 × clip(article_share / 0.30, 0, 1) + 0.20 × (1 − position of ret_21d within the basket's range) + 0.10 × [positive share ≥ 0.8] + 0.10 × [volume_ratio_1d between 1 and 2]`
(`article_share` = the symbol's articles / the basket's total).
Counterparties (PHASE 9): the plan call may propose up to five customers/suppliers named in the stored 10-K business section, each with a
quote; code verifies the quote and resolves the name with `symbols.resolve_company` (an ambiguous name is dropped); verified ones join
the basket with `theme_role: "counterparty"`. The model's memory is never a source.
Direction bullish. Horizon 42 bars. Prior 20-day excess: 0.5%.

---

## 9. Gates, quality, modifiers, votes, composite, regime, holdings, cooldown

### 9.1 Gates (`gates.py`); a failure is recorded, never silent

| Gate | Fails when | Applies to |
|---|---|---|
| `liquidity` | `adv_20d_dollars < 5e6` or `close < 5` or `median_spread_bps_1d > 20` (when available) | new ideas |
| `freshness` | the latest bar is more than 1 session before `today`; or fundamentals `as_of` older than 100 days; or any required metric `as_of` older than its window allows | all |
| `red_flag` | any of: `going_concern`, `material_weakness`, `restatement`, `delisting_notice` event ≤ 60 bars; `dilution_current` with a stated current-period amount; `debt_to_equity > 4 and current_ratio < 1` *(not applied when the equity is negative: Amendment 1)*; `cfo_negative_with_profit` | new ideas (held → `reduce`) |
| `event_risk` | `days_to_next_earnings ≤ 3` (NEEDS-YES) or, without the calendar, the last earnings press release is 85 to 95 days old (`note: "estimated from the last earnings release"`) | all families except `event` |
| `no_metrics` | the symbol has no entry in `data.metrics` | all |

### 9.2 Completeness and agreement (recorded separately, never folded into one number)
`completeness` = present-and-fresh OPTIONAL inputs / all OPTIONAL inputs of the family. `agreement`, `evidence_n` from section 8.2.
`confidence = (evidence_n × agreement + 4 × 0.5) / (evidence_n + 4)` (a prior of four neutral observations). Intraday confirmation
(section 5.6) may add or subtract at most 0.10 from `agreement` before this formula.

### 9.3 Quality multiplier (`quality.py`), start 1.0, multiply per failing check, floor 0.2

| Check | Factor | Also |
|---|---|---|
| `operating_margin ≤ 0` | 0.6 | |
| `roe < 8` | 0.85 | |
| `revenue_growth < 0` | 0.85 | |
| `debt_to_equity > 2` | 0.7 | skipped for negative equity (Amendment 1) |
| `current_ratio < 1` | 0.85 | |
| `share_count_change > 0.01` | 0.8 | |
| `dilution_current` | 0.6 | |
| `sbc_to_revenue > 0.15` | 0.7 | |
| `accrual_ratio > 0.10` or `receivables_vs_sales > 15` or `inventory_vs_sales > 15` | 0.5 | `reversion` family off for this symbol |
| `cfo_negative_with_profit` | 0.5 | `reversion` off |
| `risk_factor_change > 0.35` | 0.75 | the thesis call must quote one sentence present in the new 10-K and absent from the old (code checks both) |

A missing input applies no factor and is listed in `haircuts_skipped`. Statements older than 120 days skip the accrual/receivables/CFO checks with a note.

### 9.4 Modifiers (bench, NEEDS-YES, PHASE 10): each a named factor on the family's edge, recorded with its inputs
`insider_cluster` (+25% on trend and capital_return; turns reversion on after a drop): at least two distinct insiders with open-market
purchases in 30 days and no larger open-market sale. `estimate_revision` (+15% on event and trend when the current-year EPS estimate rose over
30 days; a cut vetoes reversion). `short_squeeze_risk` (days-to-cover > 5: volatility in the sizer × 1.5, `squeeze_risk: true`; with a bearish
event it strengthens `reduce`). None of these is a family or a vote.

### 9.5 From a family score to a 20-day excess after cost
```
edge_20d(family)    = prior_20d[family] × regime_multiplier[family][regime] × score × quality × Π modifiers        (signed by direction)
composite           = best bullish edge − largest bearish magnitude, when both exist (conflict: true)
                    = best bullish edge, when only bullish; = most negative edge, when only bearish
cost_pct            = half of median_spread_bps_1d / 10000 (0.0010 when unknown) + 0.0010 slippage
excess_20d_after_cost = composite × (0.5 + confidence) − cost_pct          (confidence 0.5 → ×1.0; 1.0 → ×1.5)   *(confidence is the blend of Amendment 2, not the text agreement)*
stance              = bullish if excess_20d_after_cost > 0; bearish if composite < 0; else neutral (reason "edge_below_cost")
```
`basis = "prior"` until the ledger has 30 outcomes for that family and regime; then `measured` with shrinkage `(n × measured + 30 × prior) / (n + 30)`
and `n` on the record (PHASE 7). The thesis must call a prior-based number "unmeasured".

### 9.6 Regime (computed once per run from SPY and, when possible, breadth)

| State | Rule | Effect |
|---|---|---|
| `trend` | `spy_above_sma_200` and (`breadth_200 ≥ 0.5` or `breadth_200 is None`) | all families ×1.0 |
| `narrow` | `spy_above_sma_200` and `breadth_200 < 0.4` | trend ×0.5 |
| `stress` | `spy_vol_21d_ann > 0.25` or not `spy_above_sma_200` | sizes halved (risk budget 0.5%), `cost_pct` threshold doubled for `stance` |
| `recovering` | `spy_drawdown_63d < −0.10` and `spy_ret_21d > 0.05` | trend ×0 (off), reversion ×1.0, capital_return ×1.0, event ×0.8 |
| `unknown` | SPY bars insufficient (today) | ×1.0, `regime_note: "N SPY bars; regime not computable"`; sizes as `stress` (conservative) |

### 9.7 Holdings pass (every run, before new ideas)
For each portfolio symbol, after gates and families:

| Action | Rule (first that matches) |
|---|---|
| `reduce` | `red_flag` gate failed, or a bearish event class ≤ 10 bars, or an earlier recommendation's invalidation rule fired (ledger), or the family that originally recommended it now returns `None`/bearish |
| `add` | stance bullish, not `oversized`, excess in the top of this run's new ideas, position after adding ≤ 20% of equity |
| `hold` | everything else |

`reduce` outranks every `open` in the returned list. Jacob's RaiseFunds should prefer `reduce` names when it must sell (a hook; do not change `raise_funds.py`).

### 9.8 Cooldown
Skip a new `open` for a symbol when the ledger has a record for the same symbol and family whose `time_stop_date ≥ today` and whose
invalidation rules have not fired; the candidate is returned with `cooldown_of: <record id>`.

---

## 10. Trade plan (`trade_plan.py`)

| Field | Rule |
|---|---|
| `entry_zone` | `[last − 0.5 × atr_14, last + 0.5 × atr_14]` around the quote (or the last close when no quote; `entry_basis` says which) |
| `stop_distance` | `2 × atr_14` for reversion and event; `3 × atr_14` for trend and capital_return, or the distance to `sma_50` when that is nearer and below; then `max(stop_distance, earnings_move_abs × last)` when the next report is inside the horizon (`gap_risk: true`) |
| `stop_price` | `entry_mid − stop_distance` |
| `target_price` | `entry_mid × (1 + 1.5 × edge at the family's horizon)` for event, reversion, trend; `None` for capital_return |
| `time_stop_date` | the trading day `horizon_bars` after today (count bar dates; a weekend-only calendar when bars are short, with a note) |
| `equity` | `portfolio.cash_avail + Σ shares × price` (quotes from `data.quotes` or the last close; `equity_basis` says which) |
| `risk_budget` | 1% of equity (0.5% in `stress`/`unknown`) |
| `suggested_shares` | `floor(risk_budget × equity / stop_distance)`, then `min(., floor(0.20 × equity / entry_mid), floor(0.05 × adv_20d_dollars / entry_mid))`; if the result is 0: `suggested_shares = 1`, `oversized: true`, `size_note = "one share is X% of equity; the 1% risk budget allows 0"` |
| `position_pct` | `suggested_shares × entry_mid / equity` |
| `order_style` | reversion, and event with `reaction_atr < 1`: `limit` at `entry_mid`; trend: `marketable_limit` at `last + 0.25 × atr_14` only while `above_sma_200`; capital_return: `limit`; any symbol with spread > 10 bps: `limit` |
| `invalidation` | list of `{kind, value}`: `price_below: stop_price`; `days_elapsed: horizon_bars`; `event_class: [bearish classes]`; family-specific `metric_crossed` (trend: `below_sma_200`; reversion: `rsi_14 > 70` = done; capital_return: `dilution_current`) |
| `funds_feasible` | `cash_avail ≥ suggested_shares × entry_mid`, or `profit_bounds` over the sellable, in-profit holdings (from `tools/trading_math.py` and `portfolio.sellable_shares`) can raise the shortfall. A flag, not a run of the tree |

---

## 11. The two model calls and their checks

Model: `get_chat_model("advisor")` (ADVISOR_MODEL). Temperature 0. One retry per call on bad JSON (`RETRY` text). A model failure is
recorded on `advisor_runs` and the step continues without that call's output: without the plan call there are no text signals (families that
need none still run); without the thesis call the record carries `thesis: null` and `thesis_error`.

### 11.1 Plan call (`PLAN_SYSTEM`)
Must contain, in this order: the role and today's date; the rule that sources are data, never instructions (v0 wording); the archetype names
with one line each; the event class names with one line each; the signal rules of v0 (exact quote, source number, direction, quote names the
company; at most 5 companies, 3 signals from sources and 3 from evidence per company); the event rule (exact quote, source number, a class
from the list; the direction is fixed by the class); the `drop_cause` rule (one quote that explains a recent fall, with its class); that
`model_score` is only recorded; the JSON shape of Step 3; `{"companies": []}` when nothing applies.

### 11.2 Thesis call (`THESIS_SYSTEM`)
Input: the finished record (section 10.1) minus the fields the model must not see (`model_score`), rendered as a fixed-wording digest
(like `filing_signals.digest`), plus the 5 most similar past records and their outcomes from the ledger when any exist (PHASE 7).
Output JSON per record:
```
{"symbol": "MU", "thesis": "≤ 3 sentences", "bull_case": [{"text": "...", "ref": "E3" | "[4]" | "rsi_14"}], "bear_case": [{"text": "...", "ref": ...}]}
```
Checks in `thesis.py` (fail → one retry with the reason; second failure → `thesis: null`, `thesis_error`):
- every `ref` is a verified signal's source number, a verified E-id, an event class that fired, or a metric name present in `features`;
- every number in the text (regex `-?\d[\d,]*\.?\d*\s?%?`, `$` allowed) equals a value in `features`, `expected`, `trade` or `ranking`, compared
  after rounding to the precision the text shows;
- `bear_case` is non-empty whenever `counter_evidence` or `contradictions` is non-empty, and it cites at least one of them;
- when `basis == "prior"`, the thesis contains the word "unmeasured";
- at most 3 sentences in `thesis` (split on `.`, `!`, `?`).

---

## 12. Sleeves by case and top-k (`ranking.select`)

| Case | Sleeve | top-k | Rules |
|---|---|---|---|
| `funds` ("here is $1,000") | `preserve` | 1 to 3 | eligible families: `capital_return`, `trend` (only in regime `trend`), `reversion`; `event` may VETO (bearish) but may not open a name unless no other family qualifies; prefer lower `vol_63d_ann` on ties; the holdings pass runs first and a `reduce` is returned above any `open` |
| `theme` | `theme` | 3 to 5 | `theme` family required; others as support; leaders shown ranked down; counterparties allowed |
| `market` ("what's happening") | `tape` | 5 | the market brief first (section 13); then `event` and `trend` inside the top three sectors by `sector_ret_63d`, plus any symbol from the brief's unusual-volume and fresh-event lists that passes the gates |

Ordering: by `excess_20d_after_cost` descending; ties by fewer haircuts, then by lower `vol_63d_ann`, then alphabetically (deterministic).
Diversity (only when it changes the pick): if the next pick shares the `sector_etf` with one already picked, take the best candidate of a
different sector whose excess is within 80% of it; record `diversity_adjustment` and `displaced_by`. Every record in the top-k of a `funds`
sleeve carries `concentrated: true` and the list of picked sectors and betas, because three names are one bet. Never more than `k` from one
family, except `capital_return` in `preserve`.

---

## 13. Case 3 market brief (`market.py`, `data.market_brief`) — PHASE 8

Computed by the Data Clerk's code (no model) over the universe when one exists, else over the candidates and the ETFs:
1. `sectors`: the 11 ETFs with `ret_21d`, `ret_63d`, `above_sma_200`, ranked by `ret_63d`;
2. `unusual_volume`: symbols with `volume_ratio_1d ≥ 3` and `abs(ret_1d) ≥ atr_pct`, with the newest stored headline naming them (from `keyword_search` or the market store), at most 20;
3. `fresh_events`: event classes fired in the last 5 bars across the stored press releases and filings of the universe, with the underreaction flag;
4. `regime`: section 9.6's state and inputs.
The Advisor's thesis for Case 3 opens with three sentences from the brief; code checks that every ticker and every percentage in them is in the brief.

---

## 14. The recommendation record (schema 2) and the ledger

Keep every v0 field with its v0 meaning: `symbol, company, stance, model_score, score, supported, consistent, bullish_signals, bearish_signals,
signals_claimed, signals_verified, signals_counted, signals[], advisor`. In v1, `score` = `round(50 + 50 × clip(excess_20d_after_cost / 0.03, −1, 1))`
so that `top_recommendation` in `account_manager.py` (score > 50 and `supported`) keeps working without a change; `supported` = at least one family
produced a result. Add:

| Section | Fields |
|---|---|
| `identity` | `schema_version: 2, record_id, request_id, case, sleeve, as_of_utc, sector_etf, market_cap, regime, regime_note` |
| `action` | `open \| add \| hold \| reduce \| avoid`, `held_shares, cost_basis, unrealized_pct, cooldown_of` |
| `families` | list of `FamilyResult`: `family, computable, missing[], scoring_basis, score, direction, horizon_bars, prior_20d, basis, n, regime_multiplier, edge_20d, completeness, features_used{name: value}, evidence_ids[], counter_evidence_ids[], notes[]` |
| `quality` | `multiplier, haircuts[{check, value, factor}], haircuts_skipped[]` |
| `modifiers` | `[{name, factor, inputs}]` |
| `evidence` | verified signals (v0 shape), E-items cited, `events[{class, direction, source, quote, age_bars, reaction_atr}]`, `counter_evidence[]`, `contradictions[[id, id]]` (two verified quotes the model flagged as opposed; code checks both exist; one unverified side drops the pair) |
| `features` | every metric used: `{name: {value, unit, as_of, source}}` (copied from `data.metrics`) |
| `agreement` | `agreement, evidence_n, completeness, confidence` |
| `ranking` | `composite, conflict, cost_pct, excess_20d_after_cost, stance, rank, diversity_adjustment, displaced_by, concentrated` |
| `trade` | section 10's fields |
| `risk` | `vol_63d_ann, atr_pct, beta_63d, max_drawdown_63d, adv_20d_dollars, median_spread_bps_1d, days_to_next_earnings, earnings_move_abs, gap_risk, red_flags[]` |
| `portfolio_fit` | `sector_weight_after, correlation_to_portfolio_63d, funds_feasible` |
| `gates` | `[{gate, passed, value, threshold}]` |
| `thesis` | `thesis, bull_case[], bear_case[], thesis_error` |
| `audit` | `metric_order, plan_run, thesis_run, config_version, family_versions{}` |

`Candidate` (on `order.candidates`): `symbol, origin, held, gate, gate_value, families (as above, may be empty), excess_20d_after_cost, rank, reason, cooldown_of, theme_role`. Cap 30.

Ledger (`prometheus/memory/ledger.py`, PHASE 7): `data/ledger/recommendations.jsonl`, one record per line, append-only, `record_id = uuid4 hex[:12]`;
`recent(symbol, family, since_date)`; outcomes (`ret_5, ret_20, ret_60` vs SPY, `stop_hit_first`, `invalidation_fired`) are filled by a separate
script from the bars later; the generate step only appends. Create the directory on first write; it is not `data/processed`.

A test asserts that no field in the record except `model_score, thesis, bull_case, bear_case, signals[*].quote, events[*].quote, companies[*].company`
originates from a model reply.

---

## 15. Tests

### 15.1 Fixtures
- `tests/fixtures/bars/make_bars.py` (imported by `conftest.py`, not run from the repo): writes, into a `tmp_path`, day-aggregate files in
  the real format (`ticker,volume,open,close,high,low,window_start,transactions`, gzip, one file per weekday, 300 sessions) for invented tickers:
  `UPUP` (steady uptrend, low vol), `DOWN` (downtrend), `FLAT`, `DROP` (flat then −12% over 5 sessions on high volume, then a reversal candle),
  `SPIKE` (+25% in the last 21 sessions: the anti-chase case), `SPY`, `XLK` (flat), with a fixed seed. Deterministic: tests assert exact numbers.
- Financials fakes: dicts in the shape of `Financials.get`. Evidence fakes: build with `filing_signals.recommendation_evidence` on the existing
  filing fixtures (`tests/test_filing_signals.py` shows how).
- A scripted model (as `tests/test_advisor.py` does) that returns fixed JSON for the plan and thesis calls, including one bad reply to exercise the retry.

### 15.2 Test files and the tests each MUST contain
| File | Must cover |
|---|---|
| `tests/test_bars.py` | reads the fixture files; `window_start` → date; missing day counted in `gaps`; never writes |
| `tests/test_technicals.py` | each indicator against a hand-computed value on a short series (RSI on the classic 14-period example; MACD; ADX; ATR; %B; OBV; drawdown; beta); `None` below `MIN_BARS` with the note; `mom_12_1` skips the last month |
| `tests/test_fundamentals.py` | each derived ratio with and without inputs; annualization of year-to-date cash-flow rows; `risk_factor_change` with two sections, with one, with identical sections; the no-score-keys walk |
| `tests/test_metrics_catalog.py` | every catalogue entry has `fn`, `unit`, `min_bars`, a test; `compute` returns one result object per symbol per requested metric; unknown names refused |
| `tests/test_work_order.py` (extend) | `verify_metric_request` cases (unknown metric lists the known names; 31 symbols; empty list; already `data_ready`); v1 orders still validate |
| `tests/test_data_clerk_metrics.py` | `request_metrics` with a fake toolbox: status, `tool_calls`, a symbol without bars → named in errors and the others still answered; `calculate_metrics` visible as a tool and on the MCP server |
| `tests/test_events.py` | every shipped class matches at least one real stored press-release/8-K sentence (copy the sentence into the test) and does not match a near miss; direction fixed; `verify_event` refuses a class the text does not match; age and reaction |
| `tests/test_gates.py` | each gate at the threshold ±ε; held symbols pass through to the holdings pass |
| `tests/test_quality.py` | each haircut; floor 0.2; missing inputs listed in `haircuts_skipped`; reversion turned off by the accrual rule |
| `tests/test_families.py` | per family: pass, each required-rule failure, `None` with `missing` when bars are short (the 17-bar case), the anti-chase case on `SPIKE`, the idiosyncratic-drop test on `DROP` with and without a sector ETF, the "fundamental cause" refusal, PEAD haircuts, the funded-buyback halving, the laggard rule on a theme basket |
| `tests/test_ranking.py` | one vote per family (two correlated trend features do not add); conflict → min rule; cost; stance thresholds; regime multipliers; holdings pass (`reduce` beats `open`); cooldown; sleeves (Case 1 does not open an event name when a capital-return name qualifies; Case 2 ranks the leader down; Case 3 leads with the brief); diversity displacement recorded; deterministic ties |
| `tests/test_trade_plan.py` | ATR stops by family; earnings-gap override; `oversized` on a $1,000 account; liquidity cap; order style by family and spread; invalidation list; `funds_feasible` |
| `tests/test_thesis.py` | number check (a number not in the record fails; a rounded one passes); ref check; bear case required; "unmeasured" required; retry once then `thesis: null` |
| `tests/test_advisor_v1.py` | the whole step on the fixture universe with fakes: status `recommended`, records validate, `candidates` holds the rejects with reasons, `advisor_runs` has `plan` and `thesis`, the ledger got one line per record, v0's `advise` still runs on the same order, `top_recommendation` picks the right symbol from the v1 `score` |
| `tests/test_ledger.py` | append-only; `recent`; a corrupt line is reported, not swallowed |

Mutation checks per phase: break one threshold, one formula, one check; each must be caught; restore and compare byte for byte (the project's rule).

---

## 16. Build order (do them in this order; stop at the end of each for Jacob's review)

| Phase | Deliverable | Acceptance |
|---|---|---|
| 1 | `bars.py`, `technicals.py`, fixture generator, `configs/sector_etfs.json` | `test_bars`, `test_technicals` green; `python -c` on the real October files for one symbol prints `ret_5d`, `rsi_14 = None` (17 bars) with the note |
| 2 | work-order fields, `verify_metric_request`, `fundamentals.py`, `filing_signals` rows, `metrics_catalog.py`, `calculate_metrics` tool, `request_metrics`, MCP exposure | `test_work_order`, `test_fundamentals`, `test_metrics_catalog`, `test_data_clerk_metrics` green; `python -m prometheus.tools.data_clerk` lists the new tool |
| 3 | `record.py`, `events.py`, `gates.py`, `quality.py`, `config.py` + `configs/advisor_v1.json` | `test_events`, `test_gates`, `test_quality` green |
| 4 | families (`capital_return` first, then `event`, `reversion`, `trend`, `theme`) | `test_families` green, including the 17-bar `None` cases |
| 5 | `ranking.py`, `trade_plan.py`, holdings pass, cooldown (ledger read stubbed), sleeves | `test_ranking`, `test_trade_plan` green |
| 6 | `prompts/advisor_v1.py`, `candidates.py`, `thesis.py`, `advisor_v1.py`, `desk.report` extension (print family, excess, trade, action per record) | `test_thesis`, `test_advisor_v1` green; a no-network run with the scripted model prints a report |
| 7 | `memory/ledger.py`, similar-past-records in the thesis prompt, `basis: measured` with shrinkage | `test_ledger` green |
| 8 | `market.py`, Case 3 brief | brief tests on the fixture universe |
| 9 | theme counterparties from the 10-K business section | tests on the stored AAPL/META 10-K text |
| 10 | bench: intraday confirmation, insider/estimate/short-interest modifiers (each behind a fake until NEEDS-YES) | tests with fakes; nothing live |

Before any live run (ASK JACOB, every time): the Finnhub metric call for a list of symbols, the sync of earlier day-aggregate months,
any new endpoint. The first live target, when approved: `./test_prometheus.sh` on one symbol with the v1 Advisor and the sample portfolio,
then the same with `--term` for a theme.

---

## 17. Things the coding model must not do

- Do not compute `mom_12_1`, `sma_200`, `pct_of_52w_high`, `beta_63d` or `breadth_200` from fewer bars than `MIN_BARS`, and do not shorten a
  window to make a number appear. Return `None` with the note.
- Do not call Finnhub, Massive, EDGAR or OpenRouter from a test or from a module import.
- Do not read or print `.env`; do not write under `data/historical` or `data/processed`.
- Do not change `raise_funds.py`, `account_manager.py`, `advisor.py` (v0) or `prompts/advisor.py`, except: `account_manager.top_recommendation`
  is kept working by the v1 `score` mapping, with a test, not by editing it.
- Do not let the model choose a threshold, a weight, a price, a size or a rank. Do not let it see `model_score` again in the thesis call.
- Do not tune a threshold from the ledger. The ledger may change `basis` to `measured` and the family weights; thresholds stay in the config file for Jacob.
- Do not add a per-event-class weight. All classes share the event family's prior until a class has 30 outcomes (and even then only the report changes).
- Do not invent an endpoint field. If `Financials.get`'s payload lacks a key, the metric is `None`.
- Do not create documentation files beyond the design-choices entries and the doc updates AGENTS.md names.

---

## 18. Open questions for Jacob (the build does not wait on them; defaults are stated)

1. Universe file source for Case 3 (default: none; the universe is the candidate list).
2. Permission to sync day aggregates back 24 months and to add the Finnhub earnings and profile endpoints (default: not yet; long-window metrics stay `None`).
3. Long-only confirmed (default: yes; bearish = `reduce`/`avoid` for held names only).
4. Whole shares only (default: yes).
5. Ledger location (default: `data/ledger/recommendations.jsonl`).
6. Should `calculate_metrics` be offered to the Data Clerk's MODEL as a tool, or only to the Advisor's code path (default: both, with `MAX_SYMBOLS = 10` for the model).

---

## 19. References for the priors (cite the key in `families[*].basis_ref`)

`bernard_thomas_1989` PEAD; `jegadeesh_titman_1993` momentum; `jegadeesh_1990`, `lehmann_1990` short-term reversal; `george_hwang_2004` 52-week high;
`daniel_moskowitz_2016` momentum crashes; `blitz_huij_martens_2011` residual momentum; `boudoukh_et_al_2007` net payout yield; `ikenberry_et_al_1995` buybacks;
`michaely_thaler_womack_1995` dividend changes; `loughran_ritter_1995` issuance; `brav_et_al_2008` 13D activism; `novy_marx_2013` gross profitability;
`sloan_1996` accruals; `cohen_malloy_nguyen_2020` 10-K language change; `lakonishok_lee_2001` insider buying; `chan_jegadeesh_lakonishok_1996` estimate revisions;
`ang_et_al_2006`, `baker_bradley_wurgler_2011` low volatility; `frazzini_pedersen_2014` betting against beta; `barber_odean_2008` attention.


## Amendment 1 (2026-11-06, Jacob): debt to equity is not tested for a company with negative stockholders' equity

A company whose balance sheet shows a stockholders' DEFICIT (total stockholders' equity below zero: Domino's, Papa John's, Yum, McDonald's after years of buybacks) has a debt-to-equity ratio that means nothing (Finnhub gives none, or a large positive number). The four rules that read `debt_to_equity` are therefore not applied to it:

1. the `red_flag` gate leg `debt_to_equity > 4 and current_ratio < 1`;
2. reversion's `sound` rule `debt_to_equity < 1.5` (the other three tests of `sound` still apply);
3. capital_return's rule `debt_to_equity <= 2`;
4. the quality haircut `debt_to_equity > 2` (factor 0.7).

`debt_to_equity` is also no longer a REQUIRED input of reversion or capital_return for such a company (a missing value no longer makes the family not computable). Every other rule, gate and haircut applies unchanged. The deficit is read from the latest stored balance sheet by the metric `negative_equity` (true when `total_equity` is below zero); with no stored balance sheet it is unknown, and the rules apply as before. The result says it everywhere: the record and the candidate carry `negative_equity: true`, each affected family lists `waived: ["debt_to_equity"]`, the `red_flag` gate and the quality `haircuts_skipped` carry the note "debt_to_equity rule waived: negative stockholders' equity (the ratio has no meaning)", and the report marks the company with `*`, "waived" in its debt/equity cell and DEFICIT in its equity cell. Design choices entry 54.

## Amendment 2 (2026-11-06, Jacob): the score is blended: 40% news and press releases, 30% performance, 20% historical metrics, 10% filings; the newer the news, the heavier

Replaces the text-agreement `confidence` of section 10. The blend is `0.40 x news + 0.30 x performance + 0.20 x historical + 0.10 x filings`, each part from -1 to +1. News and press releases: every verified signal and event
that cites an article or a press release counts `kind x 0.5 ** (age_days / 7)` (an article 1.0, a press release 0.5) and the part is `sum(direction x weight) / (sum(weight) + 2)`. Performance, historical and filings: the mean of
a table of measures, each `clip((x - neutral) / span, -1, 1)` (the table is `blend.measures` of the config); filings also count each verified filing-evidence item and filing-index event as one vote. A part with nothing to say is
unavailable and counts as 0. The strategy's edge is scaled: `excess = composite x (1 + 0.5 x direction x blend) - cost` (so `confidence = (1 + direction x blend) / 2` and the old formula `composite x (0.5 + confidence)` holds);
when no strategy passed, `excess = blend x 0.003 - cost` and the stance stays neutral. Design choices entry 55; the walk-through is `design/week5_advisor_implementation.md` section 10.


## Amendment 3 (2026-11-12, Jacob): the dilution red flag counts only equity issuance

Section 5.5, `dilution_current`: a share count that rose by less than 1.0% (vesting), an 8-K passage about debt (senior or convertible notes, debentures, capped calls, anti-dilution adjustments) and the "not a public offering" boilerplate of a private issuance no longer make the metric true. A rise of 1.0% or more, proceeds from issuing common stock, and an 8-K passage about an equity offering or an at-the-market programme still do. Why and what it costs: `docs/design_choices.md` entry 60.


## Amendment 4 (2026-11-12, Jacob): a breakout path for the trend family

Section 8.3. `above_sma_50` joins the OPTIONAL inputs. A stock above BOTH its 50- and 200-day means is in a confirmed uptrend, and then: (a) the anti-chase limit is `ret_21d ≤ 0.25` (`max_ret_21d_uptrend`) instead of 0.15; (b) when `mom_12_1 ≥ 0.50` (`breakout_min_mom_12_1`) the proximity rule needs `pct_of_52w_high ≥ 0.80` (`breakout_min_pct_of_52w_high`) instead of 0.90. RSI under 75, `%B ≤ 1.05`, residual momentum and `mom_12_1 ≥ 0.15` apply as before; a missing `above_sma_50` leaves the old rules. The notes of a pass say when a relaxed rule was used. The score formula is unchanged. Why: `docs/design_choices.md` entry 61.


## Amendment 5 (2026-11-12, Jacob): the `conviction` family, so a high blend can be a recommendation by itself

Section 8 gets a sixth family, `conviction`: the blend of news, performance, historical metrics and filings (Amendment 2) as a strategy of its own. Until now a name needed one of the five strategies to pass before it could be recommended, and the blend alone was worth at most 0.30% against a cost of about 0.20% (`no_strategy_edge_scale`), so a company with good news, a good chart and sound numbers but no strategy trigger (Yum! Brands, 2026-11-12: blend 70, `trend` failing on 12-1 momentum of -2.7%) was never one.

`conviction` (`families.py`, `configs/advisor_v1.json` `families.conviction`). REQUIRED: none (it reads the blend, computed before the families now). OPTIONAL: `rsi_14`. Pass when ALL: `blend.value >= 0.30` (a score of 65); at least 3 of the 4 blend parts have a value (`min_parts`); the news part, when it has a value, is at least 0 (`min_news`); `rsi_14 < 75` when known (`max_rsi`). Score fixed at 0.5 (`score`): the blend already scales every edge, so it is not counted twice. Bullish only. Horizon 40 bars, prior 0.6% (borrowed from `trend`: unmeasured, `basis_ref` `borrowed_from_trend`), regime multipliers trend 1, narrow, stress and recovering 0.5, unknown 1, stop 3 ATR, no family-specific invalidation (the stop, the time stop and a bearish event end it). Eligible in all four sleeves (in the funds sleeve it is not limited to the `trend` regime).

Edge = `0.006 x regime x 0.5 x quality` = 0.30% in a normal regime; the expected excess is that times the blend's scale factor (`1 + 0.5 x blend`) less the cost: 0.14% at a blend of 0.30, 0.20% at 0.60. It is a family like the others: one vote, the best edge wins the composite and the open family, the cooldown is per family. The Critic's counterfactual (Critic spec 8.2) drops the family when the blend, worked again without the quotes the review did not accept, is under `min_blend`. Why and the alternatives: `docs/design_choices.md` entry 63.


## Amendment 6 (2026-11-12, Jacob): no cost hurdle; the blend ranks; no profit number is shown

The expected 20-day excess return after cost (sections 9.5 to 9.7) was the Advisor's gate and its ranking key. It is a product of priors nobody has measured, and a few hundredths of a percent decided whether a company such as Procter & Gamble was a recommendation. Amendment 6:

- **Stance** (9.5): bullish when a bullish strategy passed (the composite edge is above zero), bearish when the composite is below zero, else neutral (`stance_reason` `no edge` for an edge of exactly zero; `no family passed` when none did). The `edge_below_cost` reason no longer exists. A new position or an addition still needs a bullish stance.
- **Score**: `round(50 + 50 x the blend)` (the v0 field `top_recommendation` reads), not the excess. A company with no metrics has none.
- **Ranking** (9.6): `sort_key` is the blend (highest first), then the larger strategy edge, the confidence, fewer haircuts, lower volatility, the symbol. The sleeve's diversity rule (the next pick is taken from another sector when its blend is at least 80% of the same-sector one's) and the holdings' `add_floor` (the lowest blend among the new ideas picked) use the blend.
- **Kept, internal**: `ranking.excess_20d_after_cost`, `cost_pct` and `scale_factor` are still worked out and recorded (the Critic checks their arithmetic); nothing decides on them and no report, thesis or card shows them.
- **Thesis**: the record the model reads says "Score: N of 100 (the blend ...)" and "No return is promised"; the "unmeasured" word rule (section 11) is removed with the number it labelled.
- **Report**: the ranking table has no "Excess 20d" column and is "ranked by the blended score"; a recommendation's line has no expected return; the empty case reads "no strategy passed for any company".
- **Critic** (its Amendment 3 removed the matching rule): `stance.action` needs a bullish stance with a composite above zero, and `recompute.score` is `round(50 + 50 x the blend)`.

Reasons and what it costs: `docs/design_choices.md` entry 66.


## Amendment 7 (2026-11-12, Jacob): the expected return after cost is removed from the record

The expected 20-day excess return after cost was a product of fixed priors (a 0.3% to 0.6% edge per strategy, a 0.2% cost) that this system has not measured, and it is not a reliable evaluator. Amendment 6 stopped using it; Amendment 7 removes it.

- `Ranking.cost_pct`, `Ranking.scale_factor`, `Ranking.excess_20d_after_cost` and `Candidate.excess_20d_after_cost` are gone from the records; `Scored.excess`, `cost_pct` and `scale_factor`, `ranking.cost_of` and `ranking.v1_score` are gone from the code; `rank_symbol(s)` takes no regime and no config.
- Config keys removed: `ranking.edge_scale`, `ranking.default_half_spread`, `ranking.slippage`, `ranking.stress_cost_multiplier`, `blend.scale_strength`, `blend.no_strategy_edge_scale`.
- The Critic's checks `recompute.scale` and `recompute.excess` are removed. What the Critic still works again: the composite, the blend and its parts, the confidence `(1 + direction x blend) / 2` and the score `round(50 + 50 x blend)`.
- The strategy edge (`FamilyResult.edge_20d`, the prior x regime x score x quality) stays: it chooses the strongest strategy and the composite, and still sizes the target price of the trade plan. It is a ranking of strategies, not a promised return, and no report shows it as one.
- Not built, by decision: a tracker of the realized outcomes of past recommendations. Past outcomes do not predict the future.

Reasons: `docs/design_choices.md` entry 68.


## Amendment 8 (2026-11-12, Jacob): `dilution_current` is not evaluated

Section 5.5's metric `dilution_current` is removed from the Data Clerk, and every rule that read it is removed from the Advisor: the `red_flag` leg, the quality haircut (factor 0.6), the filings blend measure, the REQUIRED input of `capital_return`, that family's bearish vote (score 0.6, prior −0.3%, including a cited 8-K or derived dilution item) and its `metric_crossed` invalidation. `families.capital_return.prior_bearish` and `bearish_score` are removed with the vote. `share_count_change` is unchanged. A news class `dilution_offering` is unchanged. Why: `docs/design_choices.md` entry 70.


## Amendment 9 (2026-11-14, Jacob): the Advisor recommends no amount and does not use the user's money

"The Advisor only generates recommendations. The Account Manager decides how much to buy." How much to buy or sell is the Account Manager's, from the user's money, the user's limits and what the user said (`prometheus/agents/account_manager/sizing.py`, `configs/account_manager_v1.json`; `docs/design_choices.md` entry 80; `design/sizing_in_account_manager_spec.md`). This amendment SUPERSEDES the sizing text of the sections below; they are left as they were written.

- **Sections 1, 6 and 7 (Step 10)**: the trade plan is the entry zone, the stop, the target, the time stop, the order style and the invalidation rules: prices and dates. It has no size.
- **Section 10 (trade plan)**: the rows `equity`, `risk_budget`, `suggested_shares`, `position_pct` and `funds_feasible` are gone, with the stress and unknown regime's halving of the risk budget, `oversized`, `size_note`, `equity_basis` and `funds_note`. `trade_plan.build(family, metrics, quote, today, config, symbol, reaction_atr)` is given no portfolio, no equity and no regime; `equity_of` and `funds_check` moved to `account_manager/sizing.py`. Every other row is unchanged, to the last digit (208 real plans of the old code, `tests/fixtures/advisor_real/plans_before_size_removal.json`, `tests/test_trade_plan_real.py`). The stop DISTANCE stays: it is a price fact the Account Manager's loss limit divides by.
- **Section 9.7 (holdings pass)**: `add` is a bullish stance whose blend is as good as this run's new ideas. The conditions "not `oversized`" and "position after adding <= 20% of equity" are removed (they needed a size and the user's money): the user's own position limit is applied by the Account Manager's sizing, which counts the shares already held, and an `add` that no limit leaves room for is `no_funds`. `reduce` is unchanged and carries no amount.
- **Section 2 (the Portfolio row) and the `portfolio` argument**: the Advisor reads the portfolio only for the symbols held, their shares and their cost basis (the holdings pass and the candidates). It never reads `cash_avail` or the equity (`tests/test_advisor_no_size.py` reads the package's source and fails on it).
- **Section 11.2 (thesis)**: the digest has no size line; the thesis prompt says the record has no amount to buy or sell; the number check allows only numbers the record has, so none is a size.
- **Section 14 (record, schema 2)**: `trade` loses `equity`, `equity_basis`, `risk_budget`, `suggested_shares`, `position_pct`, `oversized`, `size_note`, `funds_feasible` and `funds_note`; `portfolio_fit` loses `position_pct_after` and `funds_feasible` and keeps `sector_weight_after`, `correlation_to_portfolio_63d` and `note`. The schema version stays 2, as it did for Amendment 7 (older ledger lines still hold the fields; nothing reads them back).
- **Configuration**: `trade.risk_pct`, `trade.risk_pct_stress`, `trade.max_position_pct`, `trade.max_adv_pct`, `ranking.add_max_position_pct` and `ranking.oversized_pct` are removed from `configs/advisor_v1.json` (the 5% share of the day's dollar volume is `max_adv_pct` in `configs/account_manager_v1.json`).
- **Section 15.2**: `tests/test_trade_plan.py` no longer tests `oversized`, the liquidity cap or `funds_feasible` (the Account Manager's sizing tests do); the new tests are `tests/test_trade_plan_real.py` and `tests/test_advisor_no_size.py`.

Reasons and what it costs: `docs/design_choices.md` entry 80.
