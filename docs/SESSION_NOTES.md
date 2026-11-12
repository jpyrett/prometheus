# Session notes: how the recommendation system works

Last updated 2026-11-11 (the Critic was added: see section 9 and design/how_to_code_critic.md); the rest is as of 2026-11-06, end of the Week 5 session. Written for Jacob and for the next chat. It is a map, not the spec: the build guide is
`design/week5_advisor_implementation.md`, the reasons are the numbered entries 51 to 55 at the end of `docs/design_choices.md`, the spec is
`design/advisor_v1_final_spec.md` (with Amendments 1 and 2 at its end), the rules for working in this repo are `AGENTS.md`, the commands are `HowToRun.txt`.
(The earlier DataToolbox notes of October are in git history: `git log -- docs/SESSION_NOTES.md`; `docs/DataToolbox.md` still describes the toolbox.)

## 1. What it does, in one paragraph

You type a request: a ticker, a list of tickers, a company name, a list that mixes tickers with names or brands (`MCD, YUM, "Burger King"`), or a phrase (`Pizza`).
The system finds the listed companies, gathers their data (prices, news, press releases, SEC filings), scores each one with a blend of news, price behaviour, history and filings,
ranks them best to worst, and for the best ones writes a trade plan (entry, stop, target, size) and a three-sentence thesis. It never buys or sells. Numbers are calculated by code;
the model only proposes company names, picks quotes from the text, and writes the thesis, and code checks every quote, number and reference it produces.

```
./test_recommendations.sh 'MCD, YUM, "Burger King"' --find 3 --no-fetch --no-ledger
        |
prometheus/application/recommendations.py        classify -> find -> refresh -> order -> recommend -> report
        |
   1 classify        symbols | list | company | phrase                                  (code)
   2 find            DataClerk.find_companies: the MODEL names brand + owner, CODE confirms the ticker in Finnhub;
                     a private company (Little Caesars) is "not scored"
   3 refresh         DataClerk.refresh_data: bars, news, press releases, filings          (only with --fetch: PAID, writes files)
   4 order           DataClerk.request_data: one work order, the stored text searched (ChromaDB)
   5 recommend       Advisor.generate_recommendations(order, portfolio, case, today, top_k, only_subject=True)
   6 report          ranking, key numbers side by side, why not recommended, plans, theses, holdings
```

## 2. Inside `generate_recommendations` (`prometheus/agents/advisor/recommend.py`, twelve steps)

1. preconditions (the order must be `data_ready`) 2. candidates (`candidates.py`: the named symbols, plus the holdings) 3. **PLAN call** (model: signals, events and exact quotes; `calls.py` gives one retry and a 300 s limit)
4. metrics: one order to the Data Clerk for ~80 calculated numbers per symbol (`request_metrics`, `metrics_catalog.py`) 5. gates 6. strategy families 7. quality multiplier 8. blend, ranking, stance, score
9. holdings pass and the cooldown (`ranking.py`, the ledger) 10. trade plans (`trade_plan.py`) 11. **THESIS call** (model, then `thesis.py` checks it) 12. write the records to the order and the ledger.

**Gates** (`gates.py`; a failed gate stops a new idea, a held stock becomes `reduce`): `no_metrics`, `liquidity` (average daily dollars at least $5M), `freshness` (bars and fundamentals not stale),
`red_flag` (going concern, restatement, dilution now, debt/equity above 4 with a current ratio under 1 (not for negative equity), cash flow negative with a profit), `event_risk` (earnings too close).

**Strategy families** (`families.py`; each answers "is there an entry, which way, how big is the prior edge over 20 days"): `trend` (momentum), `event` (a verified corporate event: earnings beat, guidance cut, buyback...),
`reversion` (a sound company that fell with no news of its own), `capital_return` (high net payout yield, shares shrinking), `theme`. The edges are priors from published papers (0.3% to 1% over 20 days), marked
`basis: "prior"`: not measured yet.

**Quality multiplier** (`quality.py`): weak numbers shrink an edge (floor 0.2). **Regime** (from SPY): scales the families and halves the risk budget when stressed or unknown.

## 3. The score: the blend (entry 55, `advisor/blend.py`)

```
blend = 0.40 x news + 0.30 x performance + 0.20 x historical + 0.10 x filings          each part -1 to +1;  blend score = round(50 + 50 x blend)
scale = 1 + 0.5 x direction x blend                                                      0.5 to 1.5 (direction -1 for a bearish edge)
excess_20d_after_cost = composite_edge x scale - cost          (no strategy fired: blend x 0.003 - cost, stance neutral)
stance bullish if excess > 0;   score = round(50 + 50 x clip(excess / 0.03))
```

- **News + press releases (40%)**: the verified, counted quotes. Weight = kind x 0.5 ** (age_days / 7): a story loses half its weight every 7 days; an article is 1.0, a press release 0.5.
  Value = sum(direction x weight) / (sum(weight) + 2) (the 2 is a prior of two neutral voices).
- **Performance (30%)**: 12-1 month momentum, one-month return, distance from the 52-week high, above the 200-day mean. **Historical (20%)**: earnings yield, ROE, revenue growth, margin, current ratio, debt/equity.
  **Filings (10%)**: net payout yield, share-count change, payout coverage, dilution, cash flow vs profit, and the verified filing-evidence items.
- A part with nothing to say is **unavailable**: it counts as 0 and is shown `n/a`. Every weight, the half-life and the table of measures are in `configs/advisor_v1.json` under `blend`.
- **Negative equity (entry 54)**: a company with a stockholders' deficit (Domino's, Papa John's, Yum, McDonald's) has no meaningful debt/equity, so the four debt/equity rules are not applied
  (red-flag gate, reversion's soundness test, capital_return's limit, the quality haircut). The result says so everywhere: `negative_equity`, `waived: ["debt_to_equity"]`, a `*` in the report.
- Ranking = expected excess return after cost, highest first (so when all are negative the least negative is first); ties go to the blend, then fewer haircuts, lower volatility, the symbol.

**Trade plan** (`trade_plan.py`, code only): an entry zone from the ATR, a stop a few ATR below the entry (the number is per family in the config; the 50-day mean is used when nearer), a target from the edge (none for capital_return), whole-share sizing so a stop-out loses
about 1% of equity (0.5% in stress/unknown), liquidity and position caps, invalidation rules, a time stop. **Thesis** (`thesis.py`): at most three sentences, every number must exist in the record, every ref must be
in the record, and it must say "unmeasured" while the basis is a prior.

## 4. The record (what the Critic reads)

`order.recommendations` (the picks) and `order.candidates` (everything else, with the reason) hold dicts shaped by `record.py`: identity, action, families (with `waived`), quality, evidence (verified quotes),
features (every number with unit, date and source), agreement, ranking (`scale_factor`), `blend` (the four parts and their pieces), trade, risk, gates, thesis, audit. `tests/simple_critic.py` was the working example
(now ported into `prometheus/agents/critic/integrity.py`) of an auditor: it recomputes the stop, the position, the composite, the whole blend (with the recency weights), the scale, the excess and the score, and looks every quote up in the order's own sources.
The ledger (`data/ledger/recommendations.jsonl`) is append-only; its cooldown hides a company already recommended by the same strategy for a while.

## 5. Where things are

```
prometheus/application/recommendations.py   the runner + report          ./test_recommendations.sh   the front (prompts, flags, asks before any paid fetch)
prometheus/agents/advisor/                  advisor.py (the method) recommend.py (12 steps) blend.py families.py gates.py quality.py ranking.py trade_plan.py events.py thesis.py candidates.py calls.py config.py record.py
prometheus/agents/data_clerk/               data_clerk_agent.py (request_data, request_metrics, find_companies, refresh_data) metrics_catalog.py bars.py technicals.py fundamentals.py filing_signals.py data_toolbox.py
prometheus/memory/ledger.py   configs/advisor_v1.json (every threshold and weight)   prompts/advisor_v1.py
tests/  test_blend.py test_negative_equity.py test_recommendations.py test_recommendations_script.py test_advisor_sndk_pg.py (real captured data + the simple Critic) ...
```

## 6. How to run, and what costs money (ask Jacob first; see AGENTS.md)

- Tests (free, always safe): `~/miniconda3/envs/cmu/bin/python -m pytest -q --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo` (2,248 pass).
- `./test_recommendations.sh "..." --no-fetch --no-ledger`: uses the stored data; still calls the models and Finnhub's metrics (cents). Without `--no-fetch` it also refreshes bars, news, press releases and
  filings with the paid APIs and writes files: the script asks first, and never fetches when nobody can answer. Never read or print `.env` or `~/.aws`; never touch `data/historical/` or `data/processed/`.
- Every real run appends to the ledger and its cooldown then hides that company (`cooldown: an earlier ... recommendation still stands`). `--no-ledger` explores without reading or writing it.
  The real ledger holds four entries from this session's test runs (not deleted: deleting needs Jacob's yes).

## 7. What this session built (2026-11-05 and 06), in order

1. Advisor v1 (entry 51): twelve steps, five families, gates, quality, ranking, trade plans, thesis, ledger; tests on real SNDK and PG data with a simple Critic.
2. Data Clerk growth (entry 52): daily bars from Massive REST, 80 metrics, parsers fixed on real filings (and more on Domino's, Papa John's, Yum, McDonald's, Restaurant Brands: split negative numbers, title-case headings,
   year-to-date months from balance-sheet dates, four-column cash flows, "Part I. Financial Information").
3. The runner and script (entry 53): classify, find companies (brand to owner), only the named companies, report; Advisor on gpt-oss-120b via Cerebras (plan call about 2 s); a time limit on model calls.
4. Negative equity (entry 54, spec Amendment 1). 5. The blend (entry 55, spec Amendment 2). 6. Test clock fixed: tests no longer break as the days pass.

## 8. What the real runs showed (2026-11-06)

- Pizza leaders (DPZ, PZZA, YUM) and burger leaders (MCD, YUM, QSR = Burger King): Yum and McDonald's recommended on `capital_return`; Domino's passes it with an edge just under the cost; Papa John's fails on its own
  dividend coverage (1.97); Restaurant Brands neutral. Before entries 54 and 55 nothing passed for any of them.
- Procter & Gamble's real capture (edge 0.16% vs cost 0.20%) became a thin recommendation once the blend (78) scaled it by 1.28.

## 9. Open decisions and next steps (yours to choose)

- **The edges are the same size as the cost** (0.1% to 0.3% vs 0.2%): recommendations sit near the line. The priors, `ranking.default_half_spread` and `ranking.slippage` are the settings; none was tuned.
  The blend weights, the 7-day half-life and the measure table are choices too: priors, not measurements. Measure them once there are outcomes.
- **News depends on the model's quotes.** The news part (40%) is `n/a` when the model offered no verified quote for a company (it happened for 3 of 4 in one run). Scoring the stored articles in code would fix it.
- **No leverage test for deficit companies** now. A rule that does not use equity (debt over annualised operating cash flow) would still catch over-borrowed ones; it needs a threshold nobody has measured.
  Return on equity and the current ratio are also distorted for these companies and still used.
- `capital_return` recommends a stock that fell by half (SNDK): a "not far under the 52-week high" rule would catch it earlier. Bars are not refreshed for candidates discovered in the news.
- Not built: the refresh node before the Advisor in the desk graph, Case 3 `market`, theme counterparties, measured-basis shrinkage, the Trader.
- **The Critic (2026-11-11, entries 56 to 59) is built**: it approves or denies each recommendation (a score 0 to 100), the Account Manager decides what may trade (Ask or Auto), and `./test_recommendations.sh` and `desk --critic` run it. See `design/how_to_code_critic.md` for how it works, what it found and what is open. It has not had a live run: the first one needs your yes (Finnhub and Z.ai GLM 5.3), and `.env` needs `CRITIC_MODEL=z-ai/glm-5.3`.

## 10. Gotchas

- Tools count windows from the real clock; tests use `NOW` = yesterday at noon (`tests/test_rag_retriever.py`). Do not pin a test date.
- A new dependency goes in `requirements.txt` (direct ones only). Leave no `__pycache__` or `.pytest_cache`. A design decision gets a new numbered entry; never rewrite an old one.
- Keep an LLM out of anything a program can check (entries 24, 26). When in doubt, simpler.
