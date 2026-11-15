# Session notes: how the recommendation system works

Last updated 2026-11-15 (END OF THE LONG SESSION OF 2026-11-14 TO 15: read "START HERE" below first; the Trader was added 2026-11-14: see the end of section 9 and design/how_to_code_trader.md), 2026-11-11 (the Critic was added: see section 9 and design/how_to_code_critic.md); the rest is as of 2026-11-06, end of the Week 5 session. Written for Jacob and for the next chat. It is a map, not the spec: the build guide is
`design/week5_advisor_implementation.md`, the reasons are the numbered entries 51 to 55 at the end of `docs/design_choices.md`, the spec is
`design/advisor_v1_final_spec.md` (with Amendments 1 and 2 at its end), the rules for working in this repo are `AGENTS.md`, the commands are `HowToRun.txt`.
(The earlier DataToolbox notes of October are in git history: `git log -- docs/SESSION_NOTES.md`; `docs/DataToolbox.md` still describes the toolbox.)

## START HERE (the handoff of 2026-11-15, written at the end of a very long chat; the next chat starts cold)

**State.** Branch `dev`, NOTHING IS COMMITTED (170+ changed or new files, and the new `data/`, `news/` files from real runs). `design/accountmanager_postgres_notes.md` is untracked and holds plaintext local passwords: do not `git add .` without checking it. The local Postgres (`scripts/setup_postgres.sh`, container up) holds only the three sample accounts (1, 2, 3) and a leftover `Bob Jones` (account 123) from an early real run; every test deletes the accounts it makes.

**What was built in this stretch (all with real-data tests; read the numbered entries 79 to 85 of `docs/design_choices.md`, and `design/week6_updated_design.md`, the updated design):** Ask mode shows and approves the final order (79); the Account Manager alone sizes (`sizing.py`; Auto spends round-robin, "split evenly" is equal dollars, no cash reserved) and the Advisor and Critic no longer size (80); the Trader holds no cash and shakes hands with the Account Manager on every trade (81); a booking is a SELECT, one transaction under row-level locks on the whole portfolio, and a SELECT again, a failed trade is shown to the user, never retried, and nothing halts the run (82); a news article or press release that fails is discarded completely (83); RaiseFunds does not enlarge the sizing cash, a funding sale is the minimum, and paid test runs are allowed and preferred (84); the Account Manager asks how much to buy in Ask mode (85).

**Tests.** `~/miniconda3/envs/cmu/bin/python -m pytest -q --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo -p no:cacheprovider`. Two tests fail and are NOT caused by this work, and are left failing on purpose: (1) `tests/test_rag_loaders.py::test_an_article_in_two_real_files_differs_only_in_its_tickers_or_a_corrected_time` (a real Finnhub article, CVX `finnhub:143654406`, stored with two times 7.8 hours apart); (2) `tests/test_data_clerk_verification.py::test_recent_sources_respects_the_window_the_symbol_and_k` (its sample dates are built from "noon UTC yesterday" and the window is 72 hours, so it passes only before 12:00 UTC: anchor `NOW` in `tests/test_rag_retriever.py` to fix it). Paid tests (Jacob said yes, 2026-11-15): `PROMETHEUS_LIVE_TESTS=1 pytest tests/test_end_to_end_live.py -q` (the real agents on the stored data, no fetch, a few minutes) and the other `live` files; the live Critic and Advisor vary from run to run (one wording assertion and one "nobody was to be asked" failure came from that and were fixed or made to say what was asked). DO NOT EDIT FILES WHILE A FULL SUITE RUNS: a config edit during a run once made 179 tests fail for nothing. LAST FULL RUN WITH THE PAID TESTS ON (2026-11-15, after entry 85): 4403 passed, 38 skipped before the live files were fixed; the failures were the two above plus four in the older live files, of which three were real test bugs and are fixed (their module-scoped fixtures ran before conftest's per-test `live_clock_on`, so they inherited the dev-mode `.env` and got simulated closes instead of Finnhub quotes: both fixtures now set `USE_LIVE_TRADING=true` themselves; and Jacob then decided that NO test measures how long a session takes (entry 86): the five timing tests of `test_realtime_live.py` and the two `first_plan_seconds` tests of `test_raise_funds.py` were deleted; time limits are kept only where a limit changes WHAT IS SOLD). The FINAL full run with the paid tests on (2026-11-15, after entry 86): 4395 passed, 7 failed in 18 minutes. Two are the known ones above. Five were the SMALL model of `tests/test_raise_funds_complex_live.py` (the `fast` role, qwen3-30b-a3b): in that run it found no plan for the first purchase (`infeasible`); run again alone, 16 of that file's 17 passed. The one that keeps failing is `test_the_tree_is_wide_and_deep_and_the_search_dropped_repeats[small]` (the small model's search is 1 level deep where the test wants at least 3: it failed in all three runs since the live files were fixed). It asserts a quality of the small model, not of this work; Jacob's call whether to relax it or drop the large-against-small comparison.

**Real data.** `tests/fixtures/session_real/bob_jones_oil.json` (the first real session, migrated), `bob_jones_oil_round_robin.json` (2026-11-15, the whole new flow: CVX 6, EOG 9, PSX denied) and `clarify_ask_mode.json` (the real model's replies to the Ask-mode questions). Tests replay them; none invents stock data.

**Models.** Every chat model is on OpenRouter (Advisor, Account Manager, Trader gpt-oss-120b; Critic GLM 5.3; the Data Clerk's own). The one local model is the RAG embedder (`sentence-transformers/all-MiniLM-L6-v2`, "Loading weights" in the logs): Jacob's choice, it stays local. `prometheus/models/llm.py` (`LocalLLM`, a transformers causal LM from a course lab) is used only by the MCP host demo, never by the desk.

**Open, and Jacob's to say.** (1) Commit (nothing is committed). (2) The loss and order limits cannot be raised by an answer (only position and exposure are offered). (3) A total in shares that a limit cuts is not moved to another company. (4) The Critic's model bake-off (GLM 5.3 against Qwen), persisting WorkOrders, a PDF of `design/week6_updated_design.md` with `design/make_submission_pdf.py` (not made). (5) Filings keep the strict loader (a bad filing stops its load); only news and press releases discard a failed record. (6) Older tests still use stand-in doubles (a separate task). (7) Live trading is out of scope and refused in code.

**First things to do in the next chat.** Run the suite and read the two known failures; read `design/week6_updated_design.md` and entries 79 to 85; ask Jacob what is next.

## 1. What it does, in one paragraph

You type a request: a ticker, a list of tickers, a company name, a list that mixes tickers with names or brands (`MCD, YUM, "Burger King"`), or a phrase (`Pizza`).
The system finds the listed companies, gathers their data (prices, news, press releases, SEC filings), scores each one with a blend of news, price behaviour, history and filings,
ranks them best to worst, and for the best ones writes a trade plan (entry, stop, target, size) and a three-sentence thesis. It never buys or sells. Numbers are calculated by code;
the model only proposes company names, picks quotes from the text, and writes the thesis, and code checks every quote, number and reference it produces.

*Correction, 2026-11-14 (entry 80): the trade plan has no size. The Advisor writes the entry, stop, target and time stop (prices and dates) and never says how much to buy; that is the Account Manager's.*

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

*Correction, 2026-11-14 (entry 80): the whole-share sizing, the 1% / 0.5% risk budget, the liquidity and position caps, `oversized` and `funds_feasible` described above are gone from the Advisor; the same arithmetic, reshaped, is `prometheus/agents/account_manager/sizing.py`, from the user's limits and cash. The prices of the plan are unchanged. The holdings pass adds a held name when it is bullish and as good as the new ideas, with no 20% limit (the Account Manager applies the user's own limit). `tests/simple_critic.py` no longer checks a position.*

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
- **The Trader (2026-11-14, entries 73 to 75) is built**: it executes what the Account Manager decided to trade against a SIMULATED broker (paper at the current quote, or a backtest from the stored bars; no live trading), through a model with four guarded tools and a code sweep, and reports the fills back to the Account Manager. `desk --critic --trade` and `./test_recommendations.sh --trade` run it; `python -m prometheus.agents.trader.replay --order FILE --mode backtest --no-model` replays a saved order for free. The spec is `design/trader_v1_final_spec.md` and the build notes (three phases, the deviations, the first-live-run checklist) are `design/how_to_code_trader.md`. It has not had a live run: the first one needs your yes (OpenRouter, and Finnhub in paper mode).

## 10. Gotchas

- Tools count windows from the real clock; tests use `NOW` = yesterday at noon (`tests/test_rag_retriever.py`). Do not pin a test date.
- A new dependency goes in `requirements.txt` (direct ones only). Leave no `__pycache__` or `.pytest_cache`. A design decision gets a new numbered entry; never rewrite an old one.
- Keep an LLM out of anything a program can check (entries 24, 26). When in doubt, simpler.

- **2026-11-14 (late): the user session.** `./test_prometheus.sh --session` creates a user and an account in Postgres, reads a request in the user's own words, runs the LangGraph desk with the real agents and prints the recommendations, the trades and the portfolio (design choices entry 78, `HowToRun.txt` part 14). Real run: Bob Jones, oil, EOG, XOM and CVX bought and booked. Open: the Advisor only recommends, and how much to buy should be the Account Manager's, with ReAct questions to the user in Ask mode (not built; Advisor v1's trade plan still carries a suggested size that the Critic uses as a bound).
- **2026-11-14 (late): Ask mode asks about the order that will be placed** (design choices entry 79). The session sizes and splits first, then asks the user about each purchase with its FINAL shares and cost (the whole proposal is shown once, before the first question); a "no" changes none of the others; no cash is reserved; the script's default mode is Ask; with nobody at the terminal nothing trades and the report says why. Still open: sizing belongs to the Account Manager (with ReAct questions: a percentage, dollars or shares), see `design/sizing_in_account_manager_spec.md`.
- **2026-11-14 (late): how much to buy is the Account Manager's** (design choices entry 80, `design/sizing_in_account_manager_spec.md`). Correcting the two lines above: the Advisor's trade plan no longer carries a size and the Critic no longer sizes or receives the portfolio (`review_recommendations(order, risk, now)`; `RiskView` holds no size limit; the risk weights were scaled to total 100 and `BLOCKER_CODES` is 27). `account_manager/sizing.py` sizes every approved purchase from the user's cash and limits; Auto mode spends the funds round-robin when no split was asked, "split evenly" is equal dollars, no cash is reserved. The Advisor's records keep schema version 2 with the fields gone, and `tests/fixtures/session_real/bob_jones_oil.json` was migrated once (33 fields stripped). Open: sizing cash is the available cash, so RaiseFunds no longer changes a decision (Jacob's $51.23 sample sizes PG to 0 and is `no_funds`): whether the sizing cash should include what RaiseFunds could raise is Jacob's to decide; the Account Manager's ReAct questions in Ask mode (a percentage, dollars or shares) and the per-trade Trader handshake (entry 81) are not built.
- **2026-11-14 (late): the Trader and the Account Manager shake hands on every trade** (design choices entry 81, `design/trader_handshake_spec.md`, amendment at the end of `design/trader_v1_final_spec.md`, notes at the end of `design/how_to_code_trader.md`). The Trader holds no cash and no portfolio: for each trade the Account Manager approves the shares against the funds as they are now (`approve_trade`) and records the fill at once (`record_fill`: one Postgres transaction with `--book`), and only then the next trade starts; a purchase cannot overdraw the account and two runs cannot spend the same cash, and the handshake never waits for a next trade (no ticket, no handshake; after the last answer the run ends). Proved on the real Postgres (throwaway accounts) with the real session order and the stored closes, including two threads racing for one account. RaiseFunds is still open (Jacob's to decide, entry 80). Not run live: the paid session (`tests/test_end_to_end_live.py`) needs your yes.
- **2026-11-15: the booking is three reads around one locked commit, and nothing halts the run** (design choices entry 82, amendment 7 of `design/trader_handshake_spec.md`). The Account Manager SELECTs the account, books the trade in ONE transaction under row-level locks on the account row and every position row (a re-read inside the transaction rolls it back on any difference), SELECTs again to confirm the trade is in and the totals changed, and only then lets the Trader go on. A trade that fails is not committed, is shown to the user by the Account Manager (its `screen`) with the account as it stands, is never retried, and the Trader places the next trade; `PaperBroker.halted` is gone and a run with failed trades is `traded`. Proved on the real Postgres (`tests/test_account_booking_real.py`, with a mutation check on the lock). Not run live: the paid session.
- **2026-11-15: a news article that fails is discarded, RaiseFunds is decided, paid runs are allowed** (design choices entries 83 and 84). A news article or press release the loader cannot process (formatting, a bad time, a cut-off record) is skipped completely: it is never chunked, stored, found, scored or sent to the Advisor, and the records around it load as before (`load_news(path, skipped)`; the ingest prints what it discarded; the toolbox discards a malformed API item or a corrupt market-news line and counts it as `failed`). Filings keep the strict rule. RaiseFunds does not enlarge the sizing cash and a funding sale is the minimum number of shares, never a full liquidation. Paid test runs are preferred.
- **2026-11-15: the paid runs were made.** A real Auto session through the whole new flow is saved as `tests/fixtures/session_real/bob_jones_oil_round_robin.json` (CVX 6 and EOG 9 shares, round-robin, each limited by the 10% position limit, PSX denied for risk, both fills booked by the handshake) and replayed by `tests/test_session_replay_round_robin.py`. `PROMETHEUS_LIVE_TESTS=1 pytest tests/test_end_to_end_live.py` passes (8). The live Critic and Advisor vary from run to run: one live test asserted the exact wording of what the Account Manager understood and failed when the model gave no count for "the three" tickers (it now accepts both), and the test's failure message now prints what was asked when it fails because the session asked the user a question (one run did, once; the repeat did not). The embedder (MiniLM, `sentence-transformers`) is the one local model, by Jacob's choice; every chat model is on OpenRouter.
- **2026-11-15 (end): the Account Manager asks how much to buy in Ask mode** (design choices entry 85, `agents/account_manager/clarify.py`, `AccountManager.clarify_sizing`, `design/sizing_in_account_manager_spec.md` Phase 2). When Ask mode, an approved purchase, somebody at the keyboard and a request that did not say how much (`TradeIntent.said_how_much()`) all hold, a bounded ReAct loop with the Account Manager's own model asks up to three questions (a percentage of the account, dollars, shares, or "as much as I can", for each company or in total) BEFORE anything is sized. The model only words the question and reads the answer; code checks the answer against what the user typed and does the arithmetic (`check_choice`, `plan_choice`); the shares asked for are one more limit of the purchase (`sizing.size_purchases(..., wanted=)`, binding `asked`); the user's position or exposure limit that cut the request may be raised once for this purchase only (`offer_raise`; the profile is never changed). Real model recorded on six scenarios (`tests/fixtures/session_real/clarify_ask_mode.json`), 87 offline tests (`tests/test_account_manager_clarify.py`) and one opt-in live test.
