# The Proving Ground: backtesting, paper testing and the track record

Status: spec for the coding workers. Written 2026-11-20 by the design advisor for Jacob. It becomes design choices entry 126 (and 127 for the as-of clock, 128 for the venue seam). Nothing here is built yet.

## 1. What Jacob said

> *"I want to turn this into the jaw-dropping Trading desk that can do stock paper testing with real historical (and live) data to see how the recommended stocks actually performed. ... add this test simulation to the existing GUI and works with the Trader agent and the AccountManager. ... more graphical than too much text. We will be adding ability to trade live using a stock broker in a later version of this app, so make sure the new simulation paper testing code is clean and can hook up to live (real) trading. Focus on adding the needed LangSmith tracing also. The main goal is to show the user that their recommendations worked or did not work well. This will test the overall quality of this prometheus trading desk."*

His three decisions of 2026-11-20:

| Question | Answer |
|---|---|
| How far back does the history go? | **6 months.** News, press releases and market news are backfilled to about 2026-05-20; day bars are synced back to 2025-05 for the 252-session indicators. |
| What does the live monitor do when a stop or a target is hit? | **Watch and alert.** Nothing is sold. Entry 105 stays as it is. |
| Fundamentals are today's snapshot from Finnhub. | **Keep and disclose.** Every backtest carries the caveat. |

## 2. What it is

One new part of the desk, the **Proving Ground**, with three views of one question: *did the desk's recommendations work?*

| View | What it does | Cost |
|---|---|---|
| **Time Machine** (the backtest) | Replays the REAL desk at past moments: at each decision day D the Data Clerk, the Advisor and the Critic see only what was known on D, the Account Manager sizes with the test's own money, the Trader fills at the next open of the real stored bars, and the positions are then followed day by day on real bars by the Advisor's own plan (stop, target, time stop). The result is an equity curve against SPY and an outcome for every recommendation, the denied ones too. | Paid models, one desk run per decision day (cached). No fetch. |
| **Track Record** | Scores every recommendation the desk already made in real runs (the saved sessions, and from now on every run of the page) against the bars since. | Free. |
| **Live** | Marks the paper account to live quotes and draws every open position against its plan: stop, entry, price now, target. Flags a stop or a target that was hit. Sells nothing. | Finnhub quotes. |

It is a new package, `prometheus/proving/`, a new set of routes in the web bridge and a new tab of the page. The agents are not copied or faked: the Time Machine runs `build_desk` itself.

## 3. The rules

| # | Rule |
|---|---|
| P1 | **The same desk.** The Time Machine runs the real `build_desk` graph with the real agents and the real models, in Auto mode. No stand-in agent, no shortcut scoring, no second copy of a rule. What it tests is what runs live. |
| P2 | **Nothing from the future.** At decision day D every input is as of D: the bars, the retrieved text, the filings, the ledger, the quote, the calendar. One clock (section 4) says so, and each data door enforces it itself. A test proves each door closed. The ONE exception is P3. |
| P3 | **Said plainly where it is not point-in-time.** Finnhub's fundamentals are today's snapshot (Jacob: keep and disclose), the symbol directory is today's, and the models' own knowledge has its own cutoff. Every result carries `caveats` (closed list, section 8.4) and the page shows them. Never hidden, never in small print only. |
| P4 | **Real fills only.** A purchase fills at the next regular open after D from the stored bars (`simulator.fill_backtest`, slippage 0: entry 111), or not at all. No bar, no fill: a price is never invented, carried from another day or interpolated. |
| P5 | **The plan decides the exit.** A position is followed by the Advisor's own stop, target and time stop with the rules that exist (`simulator.follow`: stop before target on a day that touches both, only the stop on the entry bar). The Proving Ground adds no exit rule and no risk rule of its own. A position with no rule fired is OPEN and is marked at the last stored close. |
| P6 | **A test never touches a real account.** The Time Machine's money is its own portfolio in memory and its results are files. It never writes the Postgres book, the real ledgers (`data/ledger/*.jsonl`), the news files or the vector store. It never fetches: the Data Clerk is built with `allow_fetch=False`. |
| P7 | **A test never spends more than its money** (entry 111 applies unchanged: the pot, orders in dollars, the handshake). Money, prices and share counts are `Decimal` through the doors of `common/money.py` and `common/shares.py` (entry 112); a return, a drawdown and a hit rate are measurements and stay `float`. `prometheus/proving/` joins `tests/test_no_float_in_the_account.py` for its portfolio module, with returns on the allow-list by name. |
| P8 | **Every recommendation gets an outcome, the denied ones too.** For a recommendation that was NOT traded (denied, neutral without the setting, beyond the count, no cash) the outcome is HYPOTHETICAL: one share-equivalent bought at the next open and followed by its plan. It is marked `hypothetical` and is never in the equity curve. This is how the Critic is judged: did its denials avoid losers? |
| P9 | **Honest about the sample.** Every statistic carries its `n`. Under `min_sample` (config, 20) outcomes a rate or an average is shown with a "small sample" mark and no colour of success or failure. Open positions are counted as open, never as wins. |
| P10 | **Splits are handled, dividends are not.** The stored bars are unadjusted. A position that crosses a split has its shares and its plan prices adjusted on the split's date from a stored split table; a symbol whose table could not be read gets no outcome across a suspected split (a close-to-open move past `split_suspect_pct`, 40%) and says why. Returns are price returns for the stock and for SPY alike; the caveat says so. |
| P11 | **The live monitor watches.** It reads the book and the quotes, it compares, it tells. It never sells, never places, never writes a trade or a decision (Jacob). Entry 105 is unchanged. |
| P12 | **A program computes, a picture shows.** Every number on the page is computed in Python (`proving/`), exactly once. The page draws; it does no statistics of its own. No LLM takes part in an outcome, a score or a statistic (entries 24 and 26). |
| P14 | **A test takes minutes, not half an hour** (Jacob, 2026-11-20: *"this backtesting does not need to be so precise. To take a half hour + to run. Will need to make it just a few mins by adding any indexing and/or caching of previous data to speed up the scoring."*). The target: a 6-month weekly test (about 26 decision days) finishes in under 5 minutes cold and in seconds from cache. Section 7.6 is how: the decision days are researched IN PARALLEL, the bars come from an index, what does not change within a run is fetched once, and every research result is cached. Speed never buys a look at the future (P2) or an invented price (P4): those two stay exact. |
| P13 | **One seam for the broker.** The Trader's funnel talks to a `Venue` (section 6). Paper and backtest are two venues. A real broker later is a third. Nothing else in the desk learns which one it is. `PAPER_TRADING` still gates everything and no live venue is built now. |

No size limit, liquidity rule, minimum order or risk rule is added anywhere.

## 4. The as-of clock (entry 127)

### 4.1 The mechanism

Entry 76 already gives the desk a simulated moment (the previous business day, half an hour before its close) with stored-bar quotes marked `simulated`. The Time Machine is the same idea for ANY past trading day. `prometheus/common/trading_clock.py` gains:

    with trading_clock.as_of(day):            # a context manager over a contextvars.ContextVar
        ...                                   # inside: trading_now() == simulated_moment(day), market_day() == day, use_live_trading() is False
    trading_clock.bound_day()                 # the day bound by as_of, or None
    trading_clock.horizon()                   # the last instant the desk may know: end of `bound_day()` in Eastern time as an aware UTC datetime; None when nothing is bound

- `as_of` refuses a day that is not a trading day, a day in the future, and nesting with a different day (ValueError).
- `use_live_trading()` returns False while a day is bound, whatever the setting, so everything entry 76 built applies: `DataClerk.request_quotes` answers from the stored bar of THAT day (it calls `market_day()`: no change needed beyond the clock), the quote is `simulated`, the Critic skips its second-provider check.
- `clock_note()` says `CLOCK    AS OF <day> (the Proving Ground): ...` while a day is bound.
- It is a ContextVar, so it follows `contextvars.copy_context()` into the threads that already copy the context for tracing (advisor `calls.invoke_with_limit`, RaiseFunds). LangGraph copies the context into its node threads. A test runs a real graph node in a thread and reads `bound_day()` there. Any `ThreadPoolExecutor` under `prometheus/` that calls agent code without copying the context is fixed to copy it (the worker greps for them and lists them in its report).
- The desk is ALSO passed `clock=lambda: simulated_moment(day)` explicitly. Belt and braces: the explicit clock for the nodes, the bound day for the leaves.

`horizon()` is the end of day D, not the simulated moment (15:30), on purpose and stated: the stored daily bar of D is the close, entry 76 already works "with the market close of that day", and an article of D evening is the same information set as the close. The fill is the NEXT open, so nothing after the decision is used.

### 4.2 The doors (each closes itself; "bound" means `trading_clock.horizon()` is not None)

| Door | File | Change |
|---|---|---|
| Text retrieval | `prometheus/rag/retriever.py` | `search_news(..., until: datetime \| None = None)` and the same for press releases and filings. `_where` adds `{"published_ts": {"$lte": until_ts}}` beside the `$gte`. When `until` is None and a day is bound, `until = horizon()` and `now = horizon()` (so `since_days` counts back from D). Never a chunk with no `published_ts` when bound. |
| The clerk's tools | `prometheus/tools/data_clerk.py` | `DataClerkTools.today` is a property: the explicit value if set, else `bound_day()`, else the real today. `_retrieve`, `recent_sources`, `keyword_search`, `discover_tickers` pass `now`/`until` from it. `fetch_*` tools refuse while a day is bound (`"the Proving Ground never fetches"`). |
| Filing evidence | `prometheus/agents/data_clerk/filing_signals.py` | `recommendation_evidence(..., today)`: adds `f["filed"] <= today` to the window. `today` defaults to the bound day. |
| Stored text search | `prometheus/agents/data_clerk/data_toolbox.py` | `discover` and `search` take `until: date \| None` (default: the bound day) and never refresh while bound. `Financials.get` is unchanged (P3) and its payload gets `"point_in_time": False` when a day is bound so the caveat is computed, not assumed. |
| Bars | `prometheus/agents/data_clerk/bars.py` and the tools | Already bounded by `end=today`. `request_bars` with an explicit range is the Trader's and the follower's door to the FUTURE of D and is the only one: it is called by `proving/` and by the Trader's backtest venue, never by the Clerk's model, the Advisor or the Critic while bound. A test reads the calls. |
| The ledger | `prometheus/memory/ledger.py` | `recent(symbol, family, since, until: date \| None = None)`; the cooldown (`advisor/ranking.cooldown_of` caller in `recommend.py`) passes `until=today`. The Time Machine also gives the Advisor and the Critic its OWN run ledger (section 7.3), so the real ledgers are neither read nor written. |
| The Advisor's fallbacks | `advisor/recommend.py:70, 215` | `today or trading_clock.market_day()` instead of `date.today()`. |
| The portfolio's date | `common/portfolio.py:92, 184` | `today or trading_clock.market_day()`. |
| The Critic | `critic/` | No rule changes. With the clock bound its quote is D's stored close, `session_at(now)` is D's regular session, `record_expired`, `bars_stale` and `quote_stale` are judged against D. A test runs a real stored record through `review_order` as of a past D with a captured model reply and shows it is NOT denied for timing. If one of the timing checks still reads the real clock, that is a leak to fix at its door, not a check to switch off: the Proving Ground adds no `ignore_*` flag. |

`tests/test_as_of_clock.py` (real stored data): for a D in the stored range, every door above returns nothing dated after D (retrieval on a real stored symbol whose newest chunk is after D; filing evidence for a symbol with a filing after D; the ledger with a later line; a quote equal to D's stored close and marked simulated); the context reaches a thread; unbound, every door behaves exactly as before (the existing tests are the proof and must pass unchanged).

### 4.3 The history (the backfill)

`scripts/backfill_history.py` (new; it prints the plan and stops without `--yes`): for a list of symbols (default: every symbol with stored news) and `--from YYYY-MM-DD`, fetches news, press releases, filings and market news in WINDOWS (30 days, oldest first), because the providers cut a long range short. It needs `DataToolbox.fetch_news(..., start: date | None, end: date | None)` and the same on `fetch_press_releases`, `fetch_filings`, `sync_market_news` (the `from_date` form stays and means what it means today). Rules that exist stay: a broken article is discarded whole (entry 83), the security guard screens at ingest (entry 116), duplicates are dropped by id. It also fetches the split table: `DataToolbox.fetch_splits(symbols) -> data/historical/splits_v1/<SYMBOL>.json` (Massive reference splits; the worker reads the provider's documentation for the route). Then it prints the ingest commands; it does not ingest.

The design advisor runs the backfill and the ingest, not a worker (paid, hours, writes `data/` and `data/processed/vectorstore`: Jacob's yes of 2026-11-20 covers this one backfill). Day bars: `scripts/sync_day_aggs.py --from 2025-05 --to 2025-08 --yes` and the current month (done 2026-11-20).

`GET /api/proving/coverage` (section 9) reports what is stored per month (bars, news, press releases, filings, market news) so the page can show the usable range and never offer a decision day with no news behind it. `proving/coverage.py` computes it from the files and the Chroma metadata, cached by the files' mtimes.

## 5. The package `prometheus/proving/`

Plain Python. No model anywhere in it. Only `engine.py` and `live.py` reach an agent.

| Module | Job |
|---|---|
| `config.py` | Loads `configs/proving_v1.json` (every threshold; a missing or unknown key is an error, as the other configs). |
| `record.py` | The pydantic models of section 8: `PlanView`, `Path`, `RecOutcome`, `Position`, `EquityPoint`, `DecisionDay`, `Stats`, `ProvingRun`. `Decimal` money through `common/money` types. |
| `splits.py` | Reads the split table; `adjust(shares, prices, day)`. |
| `outcomes.py` | PURE. `score(plan: PlanView, bars, benchmark_bars, sector_bars, entry_after: date, horizons) -> RecOutcome`: the entry (next stored open after D; inside the plan's limit range or `not_filled` with the Trader's own reason code, by calling `simulator.fill_backtest`), the exit by `simulator.follow` (reused, not rewritten), and the day-by-day `Path` (date, close, return since entry, benchmark return since entry), the return at each horizon (5, 10, 20 sessions) with the benchmark's, the best and the worst close-to-entry on the way (`max_favorable_pct`, `max_adverse_pct`), sessions held, and `result`: `won` (closed with a gain), `lost` (closed with a loss), `open`, `not_filled`, `no_data`. |
| `portfolio.py` | The Time Machine's own money. `SimPortfolio(cash)`; `apply_fills(trades)` from the Trader's filled trades (the fill's shares, price, amount, date: never recomputed); `step(day, bars) -> list[Exit]` applies P5 for that day to each open position (stop, target, time stop; a sale at the rule's price by `simulator.follow`'s own arithmetic; proceeds to cash; realized gain by `trade_amount`); `mark(day, bars) -> EquityPoint` (cash + shares at D's close; a held symbol with no bar that day keeps its previous close and the point says `stale: [symbols]`). `as_desk_portfolio()` gives the desk the portfolio shape it reads (`common/portfolio.py`). Every identity is checked each step: cash never negative, equity = cash + marked positions, the sum of realized gains = proceeds - cost of the closed lots. |
| `stats.py` | PURE. From outcomes and the equity curve: total return, benchmark return, excess; max drawdown (depth, from, to); win rate over CLOSED outcomes; average win, average loss, profit factor; average sessions held; exit mix (stop, target, time stop, open); `by_verdict` (approved / denied / neutral / not traded: n, mean return, mean excess, win rate) which is the Critic's value; `by_score` (the Advisor's score in the buckets of the config: n, mean excess) which is the calibration; `by_family`; `by_symbol`. Each with `n` and `small_sample` (P9). |
| `store.py` | Results as files under `data/proving/` (gitignored; created on first write): `runs/<run_id>.json` (one `ProvingRun`, written through `money.dumps`), `runs/index.json`, `cache/<key>.json` (section 7.4), `sessions/<request_id>.json` (section 10). Atomic writes (temp file, rename). Never under `data/historical/` or `data/processed/`. |
| `coverage.py` | Section 4.3. |
| `engine.py` | The Time Machine, section 7. |
| `memo.py` | The per-run memo of section 7.6. |
| `track.py` | The Track Record, section 10. |
| `live.py` | The live monitor, section 11. |
| `tracing.py` | Section 12. |

`configs/proving_v1.json`:

    {"config_version": "1",
     "horizons": [5, 10, 20], "benchmark": "SPY", "min_sample": 20, "split_suspect_pct": 0.40,
     "score_buckets": [0, 40, 50, 60, 70, 80, 101],
     "time_machine": {"max_decision_days": 60, "cadences": {"daily": 1, "twice_weekly": [0, 3], "weekly": [0], "every_two_weeks": 10}, "default_cadence": "weekly",
                      "min_history_sessions": 252, "min_news_days": 14, "follow_after_end": true, "use_trader_model": false, "cost_note_per_day_usd": null,
                      "parallel_days": 8, "retry_429": 4},
     "live": {"poll_seconds": 60, "only_in_session": true},
     "store": {"root": "data/proving"}}

`cost_note_per_day_usd` is filled by the worker from the MEASURED token cost of its own real run (section 14), not guessed; null shows no estimate.

## 6. The venue seam (entry 128)

Today `PaperBroker` (`agents/trader/broker.py`) is one concrete class that branches on a `mode` string in `_simulate`, and `execute._prices` branches on it again. That is two places a real broker would have to be cut into. It becomes one.

    class Venue(Protocol):                       # prometheus/agents/trader/venue.py
        name: str                                # "paper" | "backtest" | later "alpaca"
        simulated: bool                          # True for both venues built now
        def prices(self, account, symbols: list[str], now: datetime) -> Prices: ...
        def execute(self, trade: Trade, shares: Decimal, prices: Prices) -> Execution: ...   # filled | active | rejected, by the venue's own rules
        def follow(self, ...) -> Outcome | None: ...                                          # the backtest venue's exits; None elsewhere

    PaperVenue      # the quote: today's paper branch (simulator.fill_paper), unchanged arithmetic
    BacktestVenue   # the stored bars: today's backtest branch (entry_date, fill_backtest, follow), unchanged arithmetic
    venue_for(mode, config) -> Venue             # the ONE place that maps a mode to a venue; an unknown mode is an error; a venue with simulated=False is refused unless PAPER_TRADING is off AND the venue is registered (none is)

- `PaperBroker` keeps its name and every guard (permission, the handshake `approve_trade` -> fill -> `verify_fill` -> `record_fill`, one trade at a time, never retried) and asks its venue for the price and the fill. It is the funnel; the venue is where the order goes.
- `execute.py` asks the venue for `prices` and, at step 9, for `follow`. No `if mode ==` remains in `broker.py` or `execute.py` outside `venue_for` and the precondition that refuses `book` on a venue that cannot book (a venue attribute, `books: bool`).
- A RENAME-FREE, BEHAVIOUR-FREE change: every existing Trader test passes unchanged and no recorded fixture changes. The Trader still reaches the world only through the Account Manager (the venue is given the `account`; the import test still passes).
- `design/trader_v1_final_spec.md` gets an amendment at its end: what a live venue must implement (`prices` from the broker's quote, `execute` = submit and wait for the fill report, `simulated = False`, `books = True`), that the handshake already orders the calls correctly for a real fill (approve, fill, verify, record), and what is still missing for it (order ids and idempotency, partial fills, an order that stays open past the run, cancel, the broker's own cash as a second source of truth to reconcile with the book). Nothing of it is built.

## 7. The Time Machine (`proving/engine.py`)

### 7.1 The request

    TimeMachineRequest: prompt (the user's own words, as on the request card), cash (Money), start (date), end (date), cadence (a key of the config),
                        risk (conservative|moderate|aggressive), neutral (bool: ALLOW_NEUTRAL_RECOMMENDATIONS for the run, through web.settings.run_settings), strict (bool)

Checked by code before anything is paid for: the dates are trading days inside the coverage (bars reach `min_history_sessions` before `start`; news reaches `min_news_days` before it), `start < end <= the last stored bar day`, at most `max_decision_days` decision days, the cash is positive. A refusal names the reason and the usable range.

The request must be a request for RECOMMENDATIONS (a theme, a list, a company). A direct order ("Buy 50 shares of MSFT") is refused with a sentence: there is no recommendation in it to test. The request is read ONCE by the real `AccountManager.understand_request` (one model call) and the security door screens it once, as a live run.

### 7.2 The walk

    decision_days = the trading days of [start, end] the cadence picks
    portfolio = SimPortfolio(cash);  day = start
    for each trading day from start to the last stored bar day (or `end` when follow_after_end is false):
        1. exits = portfolio.step(day, bars)                      # P5, on the day's bar, BEFORE any new decision
        2. if day is a decision day and day <= end:
               with trading_clock.as_of(day):
                   research = the result of phase 1 for this day (7.6), less what the cooldown takes (7.3)
                   the Account Manager decides and sizes in Auto mode with portfolio.as_desk_portfolio()   # the user's words: budget, count, split
                   the Trader executes on the BacktestVenue, book=False, use_model=False                   # fills at the next open (P4)
               portfolio.apply_fills(filled trades)               # dated the fill's own date; the cash leaves on that date, not on D
        3. point = portfolio.mark(day, bars);  emit               # the curve grows a point
    outcomes = one RecOutcome per recommendation of every decision day (traded: from the portfolio's own lot; not traded: hypothetical, P8)
    stats = stats.compute(...)

- Step 2's fill is on the NEXT trading day's bar. The engine applies it when the walk reaches that day (a pending fill list keyed by date), so cash and equity are right on every day.
- Exits come first on a day, and a position bought at a day's open can be stopped on that same bar only by the stop (the existing rule of `follow`).
- The benchmark curve: the same starting cash put into SPY at the open of the first fill day (or the first walk day when nothing ever filled), fractional, marked at each close.
- The desk's own run of the Trader in backtest mode also computes `Trade.outcome` by `follow`; `portfolio.step` must agree with it for every position (same exit date, price and reason). The engine asserts it per position and a disagreement fails the run loudly: there are not two exit rules.
- A decision day whose desk run fails (a model error, a refusal) is recorded as `failed` with its reason, trades nothing, and the walk goes on. Nothing halts a run (the Trader's own rule, entry 82).
- Stop: `Run.stop` is checked between days; a stopped run is saved as `stopped` with what it has.
- The sizing each decision day meets the cash the portfolio has THAT day (P7): a request "use as much as I can" spends it once and then waits for exits; "Invest $2,000 in the 3 best oil stocks" spends $2,000 each decision day while the cash lasts. That is the user's words, and the page says which one it read before the run starts.

### 7.3 The desk as of D

`build_desk(clerk_factory, advisor_factory, critic_factory, trader_factory, with_critic=True, with_trader=True, clock=lambda: simulated_moment(D), guard=guard.without_rate_limit(), ...)` with:

- `DataClerk(allow_fetch=False, guard=guard)`; `Advisor(clerk=clerk, ledger=an empty throwaway ledger)`; `Critic(clerk=clerk, ledger=a throwaway ledger, history=the same empty ledger)`; `Trader(mode="backtest", use_model=False, ledger=NullLedger())`.
- An in-memory account in Auto mode with the test's profile (risk from the request), no Postgres store (`store_factory` unset). The worker follows what `application/recommendations.py` already does for `--cash --mode auto --trade backtest --now` and reuses it rather than writing a second way.
- The Advisor and the Critic of a decision day get an EMPTY throwaway ledger: a day's research does not depend on any other day's, which is what lets the days run in parallel (7.6). The Advisor's COOLDOWN (do not open the same symbol for the same family again while the first plan is alive) is applied by the engine during the walk, in date order, by calling the Advisor's OWN function (`advisor/ranking.cooldown_of`) with the earlier recommendations of this run and D: a recommendation it catches is taken off the order before the Account Manager decides, gets `not_traded_reason: "cooldown"` and a hypothetical outcome. One rule, one function, applied at a different moment. Said difference from a live run: the Advisor does not put another company in the place of a cooled-down one, so that day may trade fewer than the number asked for.

### 7.4 The research cache

The research of a decision day (Data Clerk -> Advisor -> Critic) does not depend on the portfolio (the Advisor is given none, entry 105; the Critic gets only the `RiskView`) and, with 7.3, not on any other day. So the cache key is

    sha256(prompt as understood (the TradeIntent, canonical JSON), D, risk, neutral, strict, advisor/critic/account_manager config versions, the model ids, the store's coverage stamp)

and the value is the work order as the Critic left it. A rerun with another amount of cash, another range or another cadence reuses every day it shares and is identical up to sizing; a changed prompt, model or backfill misses. The page's quick prompts (13.2) are there so that the common tests are warm. `proving/store.py` keeps it; the page shows "N of M decision days from cache" before a run starts. `--no-cache` on the command.

### 7.6 Fast (P14)

The run has two phases, and the page shows both.

**Phase 1, research, in parallel.** Every decision day that is not in the cache is researched at once by a pool of `parallel_days` threads (config, 8). Each task runs in its own `contextvars.copy_context()` and binds its own `trading_clock.as_of(D)` inside it, builds its own `DataClerk`, `Advisor`, `Critic` and graph (through the Critic: `with_trader=False`, no decision node), and nests its span under the run's root. Shared and read-only across the tasks: the embedder, the Chroma client, the bar index, the run's memo. A provider's rate limit (429) backs the pool off: the task waits and retries up to `retry_429` times with a growing pause and the pool drops one thread; a day that still fails is `failed` and the run goes on. Results are emitted as they finish (`pv_research`), in any order.

**Phase 2, the walk, in order.** Section 7.2 with step 2 reduced to what is cheap: the cooldown (7.3), `AccountManager.decide_trades` in Auto mode, the Trader on the BacktestVenue with `use_model=False`. No model is called in phase 2. With the bars indexed a 130-day walk is well under a second per day; the page paces the drawing, not the engine (`pv_day` events are sent as fast as they are made and the page animates at its own speed, with a "skip to the end" control).

**What is fetched once per run, not once per day** (`proving/memo.py`, a memo that lives for one run, keyed by the call's arguments, thread-safe):
- the request's reading (`understand_request`) and its security screening: once;
- Finnhub's fundamentals per symbol (`Financials.get`): the snapshot is today's whatever D is (P3), so one call per symbol serves every day, and Finnhub's 60-a-minute limit no longer paces the run;
- the symbol directory lookups (`request_symbols`, the company search): today's directory (P3), once per name.
The memo wraps these at the Data Clerk's toolbox through ONE seam (`DataToolbox(memo=...)`, default None: unchanged behaviour). It never memoises anything that depends on D: retrieval, filing evidence, bars and quotes are asked for each day.

**The bar index** (`prometheus/agents/data_clerk/bar_index.py`). `BarStore.load` reads up to 300 gzip day files of every US ticker for each call, and the Advisor, the Critic and the Trader each call it for each day: that is most of the non-model time. The index is one Arrow/Parquet file per symbol under `data/index/bars_v1/` (derived data, gitignored, safe to delete: it is rebuilt), holding exactly the bars `BarStore` would give for that symbol from the day files and the REST files with the same precedence (the day file wins), plus a manifest of the source files' names, sizes and mtimes. `BarStore.load` uses it when the manifest matches the sources and a symbol is in it, builds a missing symbol's file on first use (one pass over the day files for ALL the symbols asked for), and falls back to the files otherwise. `python -m prometheus.proving index` builds it for every symbol with stored news ahead of time. The proof is equality: a test loads real symbols both ways, for several `end`/`start`/`max_bars`, and the bars and the `gaps` are identical; a changed source file invalidates it. The live desk gets the same speed-up with the same numbers.

**The scoring** (`outcomes.score`, `stats`) is pure and reads the bars already in memory: the whole Track Record and a finished run's statistics are computed in well under a second, and `track.build()` is cached by the mtimes of the session files and the bar manifest.

**Measured, not promised.** The real 6-month run of section 14 records its wall time per phase (`ProvingRun.timing: {research_seconds, walk_seconds, cached_days, parallel_days}`) and the report states it. If the cold run is over 5 minutes the worker says where the time went (per-day spans) before tuning anything.

### 7.5 The command

`./test_proving.sh` -> `python -m prometheus.proving` (`__main__.py`):

    time-machine --prompt "..." --cash 10000 --from 2026-06-01 --to 2026-11-13 --cadence weekly [--risk moderate] [--neutral] [--no-cache] [--save FILE] [--yes]     PAID (models), prints the plan and the cache hits and asks first
    track [--save FILE]                                                                                                                                             free
    live --account N                                                                                                                                                Finnhub
    coverage                                                                                                                                                        free
    show RUN_ID                                                                                                                                                     free

`langsmith_tracing.configure()` is its first call.

## 8. The result (`proving/record.py`)

### 8.1 `RecOutcome` (one per recommendation, traded or not)

    key (run-unique), decision_day, symbol, company, record_id, review_id,
    advisor: {score, stance, family, rank}, critic: {approved, score, grade, denial_reasons[]} | None, trust_score | None,
    verdict: approved | denied | neutral | unreviewed,
    traded: bool, hypothetical: bool, not_traded_reason: str | None        # denied | neutral_off | beyond_count | no_cash | not_filled | failed
    plan: {entry_low, entry_high, stop, target, time_stop_date, horizon_bars, order_style}      # Price | None each
    entry: {date, price, shares, amount} | None,   exit: {date, price, reason: stop|target|time_stop, sessions_held} | None,
    result: won | lost | open | not_filled | no_data,
    return_pct, benchmark_return_pct, excess_return_pct, sector_return_pct,                    # float | None
    gain (Money | None: realized, or unrealized at the last close; None when hypothetical),
    horizons: [{sessions, return_pct, benchmark_return_pct}],
    max_favorable_pct, max_adverse_pct,
    path: [{date, close, return_pct, benchmark_return_pct}],                                   # the entry day to the exit or the last bar; the page's small charts
    notes: [str]                                                                               # a split adjusted, a stale bar, why no data

### 8.2 `ProvingRun`

    run_id, kind: time_machine | track | live, status: finished | stopped | failed, created_utc, seconds,
    request: the TimeMachineRequest | None, understood: str (the Account Manager's own words for what it read),
    range: {first_day, last_day, last_bar_day}, decision_days: [{day, status: traded|no_trade|failed|cached, recommended, approved, denied, traded, spent (Money), reason, trace_id}],
    equity: [{date, equity, cash, invested, benchmark, drawdown_pct, stale[]}],
    events: [{date, kind: buy|exit, symbol, shares, price, amount, reason, key}],              # the markers on the curve
    outcomes: [RecOutcome], stats: Stats, caveats: [code], coverage: {...}, models: {...}, config_versions: {...}, langsmith: {trace_id, url | None}

### 8.3 `Stats`

    portfolio: {start, end, return_pct, benchmark_return_pct, excess_return_pct, max_drawdown_pct, drawdown_from, drawdown_to, best_day_pct, worst_day_pct, days, invested_avg_pct}
    outcomes:  {n, closed, open, won, lost, win_rate, avg_win_pct, avg_loss_pct, profit_factor, avg_sessions_held, exits: {stop, target, time_stop, open}, small_sample}
    by_verdict: [{verdict, n, mean_return_pct, mean_excess_pct, win_rate, small_sample}]
    by_score:   [{low, high, n, mean_excess_pct, win_rate, small_sample}]
    by_family:  [...same shape...],   by_symbol: [{symbol, n, gain, return_pct}]
    critic_value: {denied_n, denied_mean_excess_pct, approved_n, approved_mean_excess_pct, sentence}     # the sentence is code's own, from a closed set of templates
    headline: {tone: pos|neg|neutral, sentence}                                                           # code's own: "Beat SPY by 3.2 points over 24 weeks: 31 recommendations, 58% of the closed ones won." / "small sample" wording under min_sample

### 8.4 Caveats (closed list; code decides which apply)

`fundamentals_not_point_in_time`, `symbol_directory_is_todays`, `price_returns_no_dividends`, `fills_at_next_open_no_slippage`, `daily_bars_stop_before_target`, `news_coverage_thin` (a decision day with fewer than the config's stored articles for its companies), `small_sample`, `open_positions_marked`, `split_adjusted`, `model_knowledge_cutoff`. Each has ONE sentence in `proving/record.py`, shown by the page as written.

## 9. The bridge (`prometheus/application/web/`)

`proving_run.py` (the worker thread of a Time Machine run, modelled on `desk_run.run_live`), `proving_views.py` (pure: the page's shapes from a `ProvingRun`; `views.py` stays the only place that knows a recommendation's shape and is called for the per-day work-order view), and routes in `app.py`. One run at a time stays true: a proving run goes through the same `RunManager` (`kind="proving"`), so it and a live desk run exclude each other, and the existing events and stop routes serve it.

| Route | Does |
|---|---|
| `GET /api/proving/coverage` | The usable range, per-month counts, the last bar day, the symbols with a split table. Free. |
| `POST /api/proving/plan` | Body: a TimeMachineRequest. Validates, reads NOTHING with a model, returns the decision days, cache hits, the caveats that will apply, the cost estimate (or null). Free. |
| `POST /api/proving/runs` | Body: a TimeMachineRequest. 202 `{run_id}`. Rate limit, 409 when a run is active. PAID. |
| `GET /api/proving/runs` | The index of saved runs (id, kind, created, prompt, range, headline, return, excess). |
| `GET /api/proving/runs/{id}` | One saved `ProvingRun` as the page's view. |
| `POST /api/proving/runs/{id}/replay` | Plays a SAVED run through the same events at a pace (free; the page's "watch it again" and the tests' and `--replay` mode's path). |
| `GET /api/proving/track` | The Track Record (section 10), computed on request, cached by file mtimes. |
| `GET /api/proving/live/{account_id}` | One live snapshot (section 11). `GET /api/proving/live/{account_id}/events` streams one every `poll_seconds`. |
| `DELETE /api/proving/runs/{id}` | Removes a saved run (the user's own result file; never a cache entry or a session). |

New event types on the run's stream (added to `EVENT_TYPES` in the page): `pv_research` (a decision day's research finished, failed or came from the cache: day, status, recommended, approved, denied, seconds, cached), `pv_started` (the plan: range, decision days, starting cash, caveats), `pv_day` (one walk day: the `EquityPoint`, the day's `events`, and when it was a decision day its `DecisionDay` row), `pv_decision` (a decision day's desk result as `views.work_order_view` so the page can show the SAME work-order table and recommendation panel for any day), `pv_outcome` (a `RecOutcome` when it closes or is first known), `pv_finished` (stats, headline, caveats, the saved id). The days are researched in parallel, so the desk's own `stage` events are NOT replayed for them: each day reports through `pv_research`, and a `log` line per day goes to the drawer. `run_finished` ends the stream as ever.

In `--replay` mode of the server the Proving Ground serves the saved real runs under `tests/fixtures/proving_real/` (section 14) and refuses a new paid run with the replay's own sentence.

## 10. The Track Record (`proving/track.py`)

- From now on every live desk run of the page and of `session.py` that reached the Advisor saves its finished work order to `data/proving/sessions/<request_id>.json` (the `--save` capture format, through the one encoder; the same redaction as the page's records; `store.save_session`). One call at the end of `desk_run._stream` and `session.run_session`; a failure to save is a log line, never a failed run. Not in `--replay` mode, not in tests (a conftest fixture points the store at a temp folder).
- `track.build()` reads those and, clearly labelled `source: "fixture"`, the real captures in `tests/fixtures/session_real/` that have recommendations (so the page is not empty on day one), dedupes by `record_id`, and scores each recommendation with `outcomes.score` against the stored bars: the REAL fill when the capture has one (its price, shares and date), hypothetical otherwise (P8). Result: a `ProvingRun` of kind `track` with no equity curve of its own but a "$1 in every traded recommendation" index against SPY.
- Free: stored bars only. It says `last_bar_day` so a stale store is visible; refreshing bars is the paid fetch the page already has.

## 11. The live monitor (`proving/live.py`)

`snapshot(manager, account_id) -> LiveSnapshot`: the account's positions from the book (`AccountManager.list_portfolio`), a quote for each through the Account Manager's own door (`request_prices`: Finnhub, or the stored close when the clock is simulated), and for each position the plan of the recommendation that bought it (`trading.trades.record_id` -> the saved session's record; a position bought by a direct order has no plan and shows none):

    positions: [{symbol, name, shares, cost_basis, price, price_time, simulated, value, gain, gain_pct, day_change_pct,
                 plan: {stop, target, time_stop_date} | None, status: in_plan | stop_hit | target_hit | time_stop_passed | no_plan, since: date}]
    cash, invested_at_cost, market_value, equity, gain, gain_pct, as_of_utc, alerts: [{symbol, kind, sentence}], session: open | closed (+ next open)

- It reads. It writes nothing (P11). `tests/test_proving_live.py` reads its source: no `book_trade`, `record_fill`, `approve_trade`, `decide_`, `perform_trades`, no `INSERT`/`UPDATE`.
- "stop_hit" is judged on the quote now and, for the days since the purchase, on the stored bars' lows (a stop touched last Tuesday is still a stop that was hit), and says which.
- With `only_in_session` it polls only while `session_at(now).is_open`; outside it serves the last snapshot and says the market is closed.
- An account-equity history: rebuilt from the book's own dated trades and cash ledger and the stored closes (free), so the Live view opens with a curve from the account's first day, and the live point extends it.
- `PortfolioCard` on the Desk tab stays as it is (value at cost, entry 99 D11). The market value lives on the Live view.

## 12. LangSmith (the tracing this feature needs)

Tracing rules of entries 114 and 115 hold: the agent address, never `LANGSMITH_PROJECT`, `summarize()` for what a span carries (ids and figures, never article text), tests force it off.

- **A root for a test.** `prometheus/common/trace.py` gains `root(name, run_type="chain", metadata=None, tags=None)`: a context manager that STARTS a trace (the only thing in `trace.py` that may; `traced` still only nests). The Time Machine opens ONE root, `proving.time_machine`, with metadata `{proving_run_id, kind, start, end, cadence, decision_days, cash (a number), risk, neutral, strict}` and tags `["proving", "time_machine"]`. The request text goes through `summarize` like every other.
- **One span per decision day**, `proving.decision_day` with `{as_of: D, cached: bool}`; the desk's whole LangGraph run of that day nests under it (so the agents' own spans of entry 115 appear inside, per day), then `proving.fill` and, at the end, `proving.outcomes` and `proving.stats` with the headline figures. A cached day is a span with `cached: true` and no children. The walk itself (`portfolio.step` per day) is NOT a span per day: one `proving.walk` span with the counts.
- **The as-of day on every nested run**: `as_of` is put in the run metadata (LangChain `config["metadata"]`) so any LLM call inside filters by it on the LangSmith page.
- **Outcomes as feedback.** When an outcome is known, `proving/tracing.py` posts feedback on the DECISION DAY's span (its run id is kept on `DecisionDay.trace_id`): `outcome_excess_return` (score = the mean excess of the day's traded recommendations), `outcome_win_rate`, and per recommendation a feedback row keyed `rec_outcome` with `value: won|lost|open` and the symbol in the comment. For the Track Record the same feedback goes on the ORIGINAL live run's trace when the saved session has its trace id (saved from now on in the session file: `langsmith_trace_id`). So on the LangSmith page a run that recommended a stock shows how the stock did. Feedback is written only when tracing is on, never fails a run, and is idempotent by a deterministic feedback id (uuid5 of run id + key + record id) so a second scoring updates, not duplicates.
- **A dataset for the offline evaluation**: `prometheus/evals/datasets.py` gains `prometheus-outcomes` (one example per scored recommendation: inputs = the record's ids, scores, verdict, plan; outputs = the outcome) and `evaluators.py` gains code evaluators `beat_benchmark`, `stop_before_target`, `critic_denial_avoided_loss`, `approved_outperformed_denied`. `scripts/run_offline_evals.py --dry-run` covers it for free; `--upload` and `--run` stay ask-first.
- **The page**: a "Trace" link on a finished run when `langsmith.url` is set.
- Still open and NOT part of this work: masking a refused request's raw text in the trace (Jacob's open item).

## 13. The page (`ui/`)

House rules of `design/ui_v1_spec.md` hold: React 19 + Vite + TypeScript, no UI kit, no router, no state library, plain CSS with the tokens, inline SVG. **No chart library is added**: the charts are hand-written SVG in `ui/src/charts/` on a few shared helpers (`scale.ts`: linear and time scales, nice ticks; `path.ts`: line, area, step; `Axis.tsx`, `Crosshair.tsx`, `Tooltip.tsx`). The reason is the same as entry 99's: six chart kinds, full control of the look, nothing to audit. The worker loads the `dataviz` and `frontend-design` skills before the first chart if its session offers them and follows them where they do not contradict this spec.

Colour, extended and still strict: green = gain and target, red = loss and stop, blue = the benchmark and information, grey = neutral and hypothetical, yellow = only "this waits for you" (so NOT used for alerts of the Live view: a stop hit is red, a target hit is green), ember = only Go! and what is busy (the Time Machine's Launch button and the moving playhead while it runs). Both themes pass AA. Motion: the curve DRAWS as days arrive and the playhead moves; a closed outcome's card flashes once (the existing 600 ms wash); nothing else moves; `prefers-reduced-motion` turns all of it off and the charts are complete without it. Every chart has `role="img"`, an `aria-label` with its headline numbers, and a "Show as table" toggle that renders the same data as a `.tbl` (also the tests' handle).

### 13.1 Navigation

The header gains two tabs beside the logo: **Desk** (the page as it is, untouched) and **Proving Ground**. Tab state in `App.tsx` and the URL hash (`#proving`, `#proving/track`, `#proving/live`, `#proving/run/<id>`): no router. The header pills, theme and account chip are shared. While a run of either kind is active the other tab shows it is busy and its Go!/Launch is disabled with the reason.

### 13.2 The Proving Ground tab

Sub-tabs: **Time Machine**, **Track Record**, **Live**. Desktop first at 1440x900. Unlike the Desk tab this one SCROLLS (it is a report), with the hero filling the first screen.

**Time Machine, before a run (the launch pad).** One card, full width:
- The request (the same textarea rules as the request card), starting cash, risk, the Neutral and Strict checks.
- **The range picker is a chart**: a strip over the last 18 months with one row per data kind (bars, news, press releases, filings) drawn as coverage bars from `/coverage` (thin where thin), and a draggable two-handle range on top, snapping to trading days and unable to leave the usable range; the decision days appear as ticks on it as the cadence changes. Keyboard: the handles are sliders with arrow keys. Quick ranges: 1M, 3M, 6M, Max.
- Cadence as a segmented control.
- A plan line from `/plan`, live as the inputs change: "24 decision days, 9 from cache, 15 paid desk runs" and the estimate when there is one, and the caveats that will apply.
- Quick prompts: three chips with tests that are kept warm in the cache (the oil request of the sample sessions among them), so a first click shows a full result in seconds.
- **Launch** (ember). Under it the list of saved runs as small cards with their own sparkline, return and excess: click opens one, a "Watch it again" button replays it.

**Time Machine, during and after a run (the hero).**
1. **The headline strip**: the code's own `headline.sentence` in large type with its tone, and five KPI tiles: Return, vs SPY (excess, the big one), Win rate (with `n` and the small-sample mark), Max drawdown, Recommendations (traded / denied / open). Tiles count up only as real data arrives.
2. **The equity chart** (the centrepiece, about 420 px tall, full width): the desk's equity as a line with a soft area, SPY as a blue line, the starting cash as a dashed baseline; a cash-versus-invested band along the bottom; **markers on the curve**: a buy is an up-triangle, an exit a dot coloured by its reason (green target, red stop, grey time stop); decision days are ticks on the x axis, a failed one hollow. An **underwater strip** (drawdown) under it on the same x scale. A crosshair with a tooltip: the date, equity, SPY, the day's events. Clicking a decision-day tick opens that day's desk (5). While running: in phase 1 the chart's x axis is already drawn with every decision day as a tile on a **research strip** above it, and the tiles fill in as the days come back in parallel (busy: ember ring; done: the count approved over recommended; cached: instantly, with a small mark; failed: hollow red), with a line "Researching 26 decision days, 8 at a time: 11 done"; in phase 2 the playhead (ember) sweeps the curve as it draws, with a "Skip to the end" control. The log lines go to a collapsed drawer, not the centre of the page.
3. **The outcome wall**: one small card per recommendation, in decision-day order, filterable by verdict and result: the ticker, the Advisor's score dial (the existing `ScoreDial`), the Critic's verdict badge, and a **plan chart**: the price path from the entry as a line against SPY's (blue, thin), the entry zone as a band, the stop as a red line, the target as a green line, the time stop as a vertical mark, the exit as a dot; the return and the excess underneath; WON / LOST / OPEN / NOT FILLED as a badge. A hypothetical one (P8) is drawn in grey with a dashed outline and the words "not traded: denied by the Critic" (or its reason). Click: the full recommendation in the existing `RecommendationPanel` in a side sheet.
4. **The quality row**, three cards:
   - **Was the Advisor's score right?** (calibration): a dot for every outcome, x = the Advisor's score, y = excess return, coloured by result, with the bucket means of `by_score` as a stepped line and each bucket's `n`.
   - **Was the Critic right?**: approved against denied (and neutral): a strip plot of excess returns per group with the mean as a bar, and the code's `critic_value.sentence` under it.
   - **How did the trades end?**: the exit mix as one stacked bar (target, stop, time stop, open) and average win against average loss as two bars; under it `by_family` as small horizontal bars.
5. **The day's desk** (opens from a tick or an outcome): the existing `WorkOrderTable` and `RecommendationPanel` fed by that day's `pv_decision` view, under a title "The desk on 2026-08-10". The same components as the Desk tab, not copies.
6. **The caveats**: a quiet, always visible strip of the `caveats` sentences, and the models and config versions the run used. Not collapsed.

**Track Record.** The same outcome wall, quality row and KPI tiles from `/track`, with the "$1 in every traded recommendation" index against SPY as its chart, a source filter (your runs / recorded sessions), and the last bar day. When there are fewer than `min_sample` outcomes the headline says so in plain words and suggests the Time Machine.

**Live.** The account's equity curve from its first day (book + stored closes) with the live point; KPI tiles (market value, gain, today's change, cash); and **the plan ladder**: one row per position, a horizontal track from stop (red end) to target (green end) with the cost basis as a tick and the price now as a marker that moves on each snapshot; the row's status badge; an alerts list on top when a stop or a target was hit. A position with no plan shows price against cost only. The "PAPER TRADING · SIMULATED BROKER" pill stays; the view says when the quote is simulated (entry 76) and when the market is closed. No control on it can sell: there is none to build.

### 13.3 State and tests

`ui/src/proving/`: `types.ts` (the shapes of section 8 as the views send them), `provingReducer.ts` (pure; folds the `pv_*` events; reuses `runReducer` for the embedded desk day), `useProving.ts`, the components, `charts/`. The API calls join `api/client.ts`.

Fixtures are RECORDED, never typed: `ui/src/fixtures/proving_*.json` are made by the existing recording script from the server replaying the real saved runs of section 14. Tests (vitest): the reducer over a recorded stream ends in the recorded stats; each chart renders its table twin with the recorded numbers; the range picker cannot leave the usable range; hypothetical outcomes are not in the equity table; small-sample marks appear under `min_sample`; the Launch button is disabled while a desk run is active; the scale helpers on exact inputs. `?states` gains the Proving Ground's states from the recorded fixtures. Screenshots of both themes go to `docs/ui_screenshots/`.

## 14. Tests and proof (real data only)

- `tests/test_as_of_clock.py` (section 4.2).
- `tests/test_proving_outcomes.py`: `outcomes.score` on the real stored bars for the real plans of `tests/fixtures/session_real/` (their fills of 2026-11-13), each rule at a real case found in the stored bars (a stop, a target, a time stop, an open position, a limit never reached, a symbol with no bars), agreement with `simulator.follow` for every one, and a real split from the fetched table.
- `tests/test_proving_portfolio.py`: the identities of `SimPortfolio` on those fills day by day (Decimal, exact).
- `tests/test_proving_stats.py`: the statistics over a real saved run, recomputed independently in the test from its outcomes (sums and counts, not the same function).
- `tests/test_bar_index.py`: the index equals the files (section 7.6), and a changed source invalidates it.
- `tests/test_venue.py`: the seam; plus every existing Trader test unchanged.
- `tests/test_proving_engine.py`: the walk over a real saved run's CACHED research (free: no model): the same result twice, and the same whatever `parallel_days` is (1 against 8); the cooldown taken by the engine equals the Advisor's own function on the same records; the per-run memo is never asked for anything dated; another amount of cash changes sizes and not recommendations; a stop mid-run; a failed day does not halt; exits before decisions; the engine's exits equal the Trader's `Trade.outcome`; nothing written outside the temp store (the real ledgers, the news files and the vector store have the same mtimes before and after); `PAPER_TRADING` off refuses.
- `tests/test_proving_live.py` (throwaway Postgres account, skipped when the book is not reachable; the simulated clock, so no Finnhub call) and the source read of P11.
- `tests/test_web_proving.py`: the routes over the saved real runs in replay; the plan refusals; one run at a time across kinds.
- `tests/test_proving_tracing.py`: with tracing forced off nothing is posted and nothing fails; the feedback ids are deterministic; `summarize` keeps article text out of the proving spans.
- **The real runs (paid; the standing yes for runs that test the desk covers them; no fetch):** one Time Machine run over the backfilled 6 months, weekly, on the oil request of the sample sessions, saved to `tests/fixtures/proving_real/oil_weekly_6m.json` with its cache, and one short run (4 decision days) for the fast tests. Their token cost is measured and written into the config's `cost_note_per_day_usd` and the report.
- The purge rule (entry 90) holds: one real-data test of a behaviour, no ladder of boundary values.

## 15. Documents

Entries 126 (the Proving Ground: the rules of section 3), 127 (the as-of clock) and 128 (the venue seam) at the end of `docs/design_choices.md`, in the form of the others. `AGENTS.md` (one paragraph: where it is, the rules P2, P6, P11, what is paid: a Time Machine run and the backfill are ask-first, `track`, `coverage`, `show` and a replay are free). `README.md` layout, `HowToRun.txt` (a new PART), `docs/SESSION_NOTES.md` checkpoint, `design/trader_v1_final_spec.md` (the venue amendment), `design/ui_v1_spec.md` (a pointer to this spec), `.gitignore` (`data/proving/`, `data/index/`).

## 16. Order of work

| Step | Who | What | Depends on |
|---|---|---|---|
| A | advisor | Day bars synced to 2025-05 and refreshed to today | done 2026-11-20 |
| B | worker 1 | Section 4: the as-of clock, every door, `scripts/backfill_history.py` and the windowed fetchers, the split table fetch; `tests/test_as_of_clock.py` | - |
| C | worker 2 | Sections 5 (not `engine`, `track`, `live`, `tracing`), 6 and 8: `proving/` pure core, the venue seam; their tests | - (parallel to B; disjoint files) |
| D | advisor | Review B; run the backfill and the ingest | B |
| E | worker 3 | Sections 7 (with 7.6: the bar index, the memo, the parallel research), 9, 10, 11, 12: the engine, the command, the bridge, the track record, the live monitor, the tracing; the real runs of section 14 | B, C, D |
| F | worker 4 | Section 13: the page, on the recorded fixtures of E | the contract of 8 and 9 from C; the recordings from E |
| G | advisor | Review everything: both suites, the build, a real run from the page, screenshots; the documents; restart the page on port 8000 | all |

## 17. Open, for Jacob

1. **A real broker.** The seam is ready (section 6); the amendment lists what a live venue still needs. Which broker (Alpaca was the maybe of week 5) is his.
2. **Paper stop-loss orders.** The monitor watches (his answer). Selling at the stop on paper is the stop-loss TODO of 2026-11-15 and would add a third origin of a sale to entry 105.
3. **Point-in-time fundamentals.** Disclosed, not fixed (his answer). Massive's statements by filing date would fix it.
4. **Intraday.** Fills are the next open and exits are judged on daily bars. The minute bars are stored for October only and no code reads them.
5. **Survivorship.** A theme's companies are found with today's directory and today's model: a company that was delisted since is not found. Said in the caveats (`symbol_directory_is_todays`).


## 18. As built (2026-11-20): where the code differs from the sections above

The sections above are the plan the workers built from. What changed while building, each with its design choices entry:

| Section | As built |
|---|---|
| 7.6 | Two speeds (entry 132). FAST is the default: the Advisor's reasoning off, the Critic's low, the fastest provider, a 90 second call limit; FAITHFUL is the live settings. `parallel_days` is 24: every day of a run at once. The cache key holds the speed. Measured: 23 days in about 90 s fast, 361 s faithful, 2.4 s from the cache. The 429 backoff retries and pauses; "the pool drops one thread" was not built (no 429 ever occurred). |
| 7.3 | The cooldown is applied in the walk and its reason is `not_traded_reason: "cooldown"`. In the real six-month test it held back about half of the recommendations. |
| 8 | `ProvingRun` also has `speed` and `timing`; `RecOutcome` has `source`, `reject_code`, `split_adjusted`; `PlanView` has `entry_mid`, `limit_price`; `Stats.portfolio` has `start_equity`/`end_equity` and its `start`/`end` are dates. The caveats are 11: `fast_models` was added, and it says the Advisor's reasoning was switched OFF. |
| 8.3 | The Critic's sentence says right or wrong only beyond two standard errors of the gap (`stats.CHANCE_BAND`); else "No clear difference". |
| 9 | Also `GET /api/proving/runs/{id}/days/{day}` and `.../records/{key}`; `/api/health` has `active_kind`. `pv_day` carries `decision_day`; `pv_finished` carries `saved_id`, `speed`, `timing`. |
| 10 | The Track Record's curve is an index of 100 ("$1 in every traded recommendation"), and `stats.portfolio` is null for it. |
| 11 | A position bought as a neutral company has `status: "no_plan"` and a note that says why. The Live view asks again by itself after an error. |
| 4.1 | The last usable bar day is `trading_clock.last_closed_day` (entry 129): today after its close. |
| 14 | The real runs are `tests/fixtures/proving_real/`: `oil_weekly_6m_fast.json` (the page's showcase), `oil_weekly_6m.json` and `oil_weekly_6m_cash100k.json` (faithful), `oil_weekly_short.json`, `track.json`, the two live snapshots and the live session. |
