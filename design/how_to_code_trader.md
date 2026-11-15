# How the Trader is coded: the guide and the build notes

The design is `design/trader_v1_final_spec.md`. This file is the record of the build: how it is run, what goes where, and, phase by phase, what was actually made. Each phase adds its own notes at the end when it finishes. Where the build differs from the spec, the difference is listed in that phase's notes with the reason.

## Status

Last updated 2026-11-14, 00:45 Eastern. **The build is finished.**

| phase | what it builds | state |
|---|---|---|
| spec | `design/trader_v1_final_spec.md` | written (Fable 5.2); tightened after phase 1 and after phase 2 (see the two "Changes to the spec" lists) |
| 1 | the foundations: work order, Data Clerk bars, Account Manager methods, records, settings | **done** 22:49 to 23:10 (Sonnet 5.5, extra-high effort): 3,274 tests pass, notes below |
| 2 | the Trader itself: `prometheus/agents/trader/`, its tools and its prompt | **done** 23:12 to 00:04 (Sonnet 5.5, extra-high effort): 3,725 tests pass (run again by Fable), notes below |
| 3 | the desk graph, the runner, the docs | **done** 00:08 to 00:45 (Sonnet 5.5, extra-high effort): 3,823 tests pass (run again by Fable), notes below |

**The Trader agent is in `prometheus/agents/trader/`** and in the desk graph (`desk --critic --trade`), the runner (`./test_recommendations.sh --trade`) and its own replay command. **First live run (2026-11-14): done, except what needs a trading day.** The day files are refreshed too (AWS CLI sync, 287 files). After the paid data refresh, the real Advisor, the real Critic (GLM 5.3), the Trader's real model (gpt-oss-120b) and real Finnhub quotes were run end to end: a purchase filled in paper mode, a funded purchase (a sale plus a purchase) filled after the user's yes, a fill was booked in a real Postgres account, and the free replay ran on a real saved decision. Not yet run: the desk itself (`desk --critic --trade`), a backtest that actually fills (it needs a stored bar from the next session, Monday 2026-11-16 or later), the Ask-mode prompt in a terminal. See the two sections at the end of this file.

### Changes to the spec after phase 1

Phase 1 found one hole and several loose ends. The spec now says:

- **A funding sell can no longer fill for nothing.** Before any holding is sold, code checks that the cash plus what the sells would bring pays for at least one share of the purchase. If not, nothing is sold. The output guard also reports a filled funding sell whose purchase did not fill (spec sections 7 and 11).
- **Amounts are rounded the way the account book rounds** (`portfolio.trade_amount`, half up), not with Python's `round` (spec section 6).
- **One trade date per run**, and in a backtest one entry date for the whole run (spec section 6).
- **The model's tool calls run one after the other**, in the order it wrote them. The prebuilt `ToolNode` runs them at the same time, and two buys in one turn would race for the same cash (spec section 8).
- **A second run starts from the first run's fills** when they were kept in memory only (spec section 4, step 2).
- **The replay command is its own module**, `prometheus/agents/trader/replay.py`: it is the only Trader module that may import the Data Clerk (spec section 16).
- **A funding sell carries its purchase's `record_id`** (spec section 7), as phase 1 assumed.

### Changes to the spec after phase 2

The review of phase 2 found no broken guard. Four things are tightened in phase 3:

- **Tool results carry codes and numbers, not sentences.** A rejected trade's `reason` is built from `verify_trade`, which repeats two values of the order (`will_trade`, `valid_until_utc`). On an order read from a file those could be any text, and the model would read it. The model now gets the `reject_code`; the sentence stays on the record and in the report (spec section 8, rule T9).
- **A trade's symbol must be a ticker.** `Trade` checks it with `check_symbol`, for the same reason (spec section 14).
- **The commission must be 0 in this version.** The cash check counted it and the fill recorded it, but `apply_fill` and the account book do not charge one, so any other value would overstate the cash (spec section 15).
- **Prices are asked in groups of at most 30 symbols**, the limit of a quote order and of a bars order (spec section 6).

To see where the build is: `git status --short`, `ls prometheus/agents/trader`, and the notes at the end of this file.

## How the build is run

The same method as the Critic (`design/how_to_code_critic.md`): one model writes the spec, another codes it.

1. **Fable 5.2 wrote the spec** from three sources: the week 5 architecture (`design/week5_submission_jpyrett.pdf`), the week 2 Trader loop (`design/jpyrett_capstone_tradingbot.md`: `verify_request`, `check_trade_status`, `AccountManager.notify`, "the Trader does not call the Data Clerk"), and the code on `dev` (the Critic's reviews, `decisions.py`, `work_order.verify_trade`, the Postgres account book).
2. **Sonnet 5.5 codes it at extra-high effort, in three phases.** Each phase is a new agent with no memory of the one before. It reads `AGENTS.md`, the spec and this file, builds its part with the tests, runs the whole suite, and writes its notes here.
3. **Fable reviews between phases**: reads the diff, runs the suite, checks the guards of spec section 2 against the code, and only then briefs the next phase.

The phases run one after the other, not side by side, because they change the same files (`work_order.py`, `account_manager.py`) and each needs the one before to be green.

Rules every phase works under (from `AGENTS.md`):

- No paid call. Tests use fakes. No desk run with a symbol, no real model, no quote, no fetch.
- `.env`, `secrets/`, `data/historical/` and `data/processed/` are not read into the build and not written.
- No file deleted, no dependency added, no commit.
- Tests run with caches off, so no `__pycache__` or `.pytest_cache` is left:
  `PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/cmu/bin/python -m pytest -q -p no:cacheprovider --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo`

## What the Trader does (one page; the spec has the detail)

The Account Manager hands the Trader a work order that the Critic reviewed and the Account Manager decided on. The Trader executes the permitted trades against a simulated broker and reports the fills back to the Account Manager, which updates the portfolio. There is no live trading: the mode is `paper` (fills at the current quote) or `backtest` (fills from the stored daily bars, then the trade is followed to its stop, target or time stop).

```
START -> account_manager -> data_clerk -> advisor -> critic -> account_manager_decision -> [account_manager_funds] -> trader -> END
```

The ten steps of `Trader.execute_trades(order, account, portfolio, now, funding, approver, book)`:

```
 1 preconditions      PAPER_TRADING is true, the mode is paper or backtest, the order was decided          code
 2 account            AccountManager.trading_portfolio                                                      code
 3 tickets            one per decision that says yes_auto or yes_user; work_order.verify_trade; the limits  code
 4 prices             ONE request through the Account Manager: quotes (paper) or bars (backtest)            code
 5 funding consent    the user's yes for the sells that pay for a purchase                                  code + the user
 6 the model's loop   gpt-oss-120b, temperature 0, four tools                                               MODEL
 7 the sweep          code places what the model left, and notifies the Account Manager                     code
 8 output guard       work_order.verify_trades, then AccountManager.notify_trades                           code
 9 outcomes           backtest only: follow each purchase through the later bars                            code
10 records            order.trades, order.trader_runs, the trades ledger, status "traded"                   code
```

The model's four tools, and what keeps it inside its permission:

| tool | what it does | the guard |
|---|---|---|
| `verify_request()` | lists the tickets | everything in it is made by code |
| `place_order(trade_id, order_type, limit_price, note)` | places one ticket | no argument names a symbol, a side or a share count; `verify_trade` runs again; the limit must be inside the range code computed; once per `trade_id`; sells before buys |
| `check_trade_status(trade_id)` | active, filled or rejected | reads only |
| `notify_account_manager()` | reports the fills | refused while a ticket is unplaced; the output guard runs first |

Temperature 0 makes the model repeatable. It is the tools that stop a made-up trade.

## Where each piece goes, and which phase builds it

| file | what it holds | phase |
|---|---|---|
| `prometheus/common/work_order.py` | status `traded`, `trades`, `trader_runs`, the `fetch_bars` action, `verify_bar_request`, `verify_trades` | 1 |
| `prometheus/common/portfolio.py` | `apply_fill` | 1 |
| `prometheus/tools/data_clerk.py`, `prometheus/agents/data_clerk/data_clerk_agent.py` | `daily_bars`, `request_bars` (no model) | 1 |
| `prometheus/agents/account_manager/account_manager.py` | `connect`, `trading_portfolio`, `request_bars`, `perform_trades`, `notify_trades` | 1 |
| `prometheus/agents/account_manager/decisions.py` | `terminal_approver` asks a card's own `question` | 1 |
| `prometheus/models/chat.py`, `.env.example` | the `trader` role, `TRADER_MODEL` | 1 |
| `prometheus/agents/trader/record.py` | `Trade`, `Fill`, `Outcome`, `Run`, the closed list of rejection codes | 1 |
| `prometheus/agents/trader/config.py`, `configs/trader_v1.json` | every threshold | 1 |
| `prometheus/agents/trader/simulator.py` | pure fill rules and `follow` | 2 |
| `prometheus/agents/trader/tickets.py` | decisions to tickets, the limits, the funding sells | 2 |
| `prometheus/agents/trader/broker.py` | `PaperBroker` | 2 |
| `prometheus/tools/trader.py` | the four tools, the budget | 2 |
| `prometheus/prompts/trader.py` | the system prompt | 2 |
| `prometheus/agents/trader/trader.py`, `execute.py` | `class Trader`, the loop, the ten steps, the replay command | 2 |
| `prometheus/agents/trader/report.py` | the TRADER section of a report | 2 |
| `prometheus/langgraph/desk.py` | the `trader` node, `--trade`, `--no-trader-model`, `--account`, `--book` | 3 |
| `prometheus/application/recommendations.py`, `test_recommendations.sh` | `--trade` in the runner | 3 |
| `docs/design_choices.md` (entries from 73), `HowToRun.txt`, `README.md`, `AGENTS.md`, `docs/SESSION_NOTES.md` | the docs | 3 |

## The order of work inside each phase

**Phase 1.** A baseline run of the whole suite first, so an old failure is not blamed on the build. Then the work order (everything else reads it), `apply_fill`, the Data Clerk's bars, the Account Manager's methods, the records and the settings, and last the tests that pin lists this phase extended (statuses, action types, model roles).

**Phase 2.** From the inside out: the simulator (pure functions, tested alone), the tickets, the broker, the tools, the prompt, then `Trader` and its steps, the report, and the replay command. Last, each rule T1 to T10 of the spec is broken on purpose once to see its test fail.

**Phase 3.** The desk node and its flags, a saved account in the desk (`--account`, `--book`), the runner and its script, then the docs.

## How to run it once it is built

Free (stored bars, no model): replay a saved, decided order as a backtest.

```
python -m prometheus.agents.trader.replay --order FILE --mode backtest --no-model
```

This one works now (phase 2). `--cash N` or `--portfolio FILE` gives it the account to trade.

Paid, so Jacob's to run: anything with the model (OpenRouter) or in paper mode (Finnhub quotes).

```
python -m prometheus.langgraph.desk --symbol PG --critic --trade                       # paper, in memory
python -m prometheus.langgraph.desk --symbol PG --critic --raise-funds --trade         # with the funding sells (it asks)
python -m prometheus.langgraph.desk --symbol PG --critic --trade --account jacob_sample --book   # saved in the account book
```

These commands work now (phase 3). None of them has been run for real yet: see "First live run" at the end of this file. `--account` writes the Account Manager's decisions to that Postgres account even without `--book`, so use a throwaway account, not one of the three samples.

---

The notes of each phase follow, added by that phase when it finishes.

## Phase 1: what was built

Done 2026-11-13 by Sonnet 5.5 (phase 1 of 3). Scope: spec sections 11 to 15 and the tests of section 18 that belong to them. No Trader class, no tools, no prompt, no simulator, no broker, no desk, no docs.

**The suite.** Command: `PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/cmu/bin/python -m pytest -q -p no:cacheprovider --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo`.

| run | passed | skipped | errors |
|---|---|---|---|
| baseline, before any change | 2926 | 30 | 5 |
| after phase 1 | 3274 | 30 | 0 |

The 5 errors of the baseline were all in `tests/test_account_book.py`: Postgres was not running. It was running again (started by somebody else, not by me) when the last full run was made, so that run has no errors, and the new Postgres test of this phase passed against the real database (below). Nothing fails.

### What was built, and where

| part | where |
|---|---|
| work order (section 11) | `prometheus/common/work_order.py`: status `traded` (in `STATUSES`, the `Literal`, `HAS_DATA`), `trades` and `trader_runs`, `bars` request field, `DataResult.bars`, action `fetch_bars`, `TRADE_STATUSES`, `TRADE_KINDS`, `TRADE_SIDES`, `iso_date`, `verify_bar_request`, `trade_of`, `trades_of`, `verify_trades`, the docstrings |
| Data Clerk (section 12) | `DataClerkTools.daily_bars` in `prometheus/tools/data_clerk.py`; `DataClerk.request_bars` in `prometheus/agents/data_clerk/data_clerk_agent.py`; `start=` added to `BarStore.load` in `prometheus/agents/data_clerk/bars.py` |
| Account Manager (section 13) | `connect`, `trading_portfolio`, `request_bars`, `perform_trades`, `notify_trades` in `prometheus/agents/account_manager/account_manager.py`; `apply_fill` and `trade_amount` in `prometheus/common/portfolio.py`; the one-line `question` change in `terminal_approver` (`decisions.py`) |
| records (section 14) | `prometheus/agents/trader/record.py` |
| settings (section 15) | `configs/trader_v1.json`, `prometheus/agents/trader/config.py`, `ROLES["trader"]` in `prometheus/models/chat.py`, `TRADER_MODEL` (and `PAPER_TRADING`) in `.env.example`, the autouse `paper_trading_on` fixture in `tests/conftest.py` |

`prometheus/agents/trader/` has no `__init__.py` (a namespace package, as the other agents). It holds `record.py` and `config.py` only.

Tests, all new unless said otherwise (counts are pytest items): `tests/test_work_order_trader.py` (74), `tests/test_data_clerk_bars.py` (33), `tests/test_account_manager_trades.py` (54), `tests/test_trader_config.py` (102), `tests/test_trader_record.py` (45), `apply_fill` and `trade_amount` tests appended to `tests/test_portfolio.py` (14 new), the Postgres test appended to `tests/test_account_book.py` (1 new). Updated: `tests/test_chat_model.py` (the roles pin, "trader" is no longer the example of an unknown role, `TRADER_MODEL` and `ACCOUNT_MANAGER_MODEL` join the settings it clears, three new tests), `tests/test_work_order_critic.py` (the statuses, `HAS_DATA`, the "already has its data" loop now includes `traded` and the bars order).

### Signatures phase 2 needs (exact)

```python
# prometheus/common/portfolio.py
apply_fill(portfolio, symbol, side, shares, price, today, name=None) -> Portfolio       # ValueError with the reason; NEW portfolio
trade_amount(shares, price) -> float                                                      # round(shares x price, 2), half up, as the account book rounds
# prometheus/common/work_order.py
verify_bar_request(order) -> list[str];  iso_date(text) -> date | None
trade_of(order, trade_id) -> dict | None;  trades_of(order, record_id) -> list[dict]       # trades_of returns BOTH kinds (a funding sale carries its purchase's record_id)
verify_trades(order) -> list[str]
TRADE_STATUSES = ("new", "active", "filled", "rejected");  TRADE_KINDS;  TRADE_SIDES;  TRADING = ("yes_auto", "yes_user")  (already there)
# prometheus/agents/account_manager/account_manager.py
AccountManager.connect(store, account_id) -> None
AccountManager.trading_portfolio(portfolio=None) -> Portfolio
AccountManager.request_bars(symbols, start: date, end: date) -> WorkOrder                 # data.bars {symbol: [bar...]}; start and end must be date (not datetime or text): ValueError otherwise
AccountManager.perform_trades(order, portfolio, trader, now=None, funding=None, approver=None, book=False) -> WorkOrder
AccountManager.notify_trades(portfolio, trades: list[dict], traded_on: date, book=False) -> tuple[Portfolio, list[str]]
# prometheus/agents/trader/config.py
load_config(path=None) -> TraderConfig (frozen, extra=forbid);  paper_trading_enabled() -> bool   # T4: true, yes, on, 1 pass
# prometheus/agents/trader/record.py
Trade, Fill, Outcome, Snapshot, FundingRun, Run, REJECT_CODES, MODEL_FIELDS, MODES, ORDER_TYPES, PLACED_BY, PRICE_SOURCES, EXIT_REASONS, SUMMARY_BY, KINDS, SIDES, STATUSES
# prometheus/agents/data_clerk: DataClerkTools.daily_bars(symbols, start: date, end: date) -> {"bars": {symbol: [bar...]}, "gaps": [...]};  DataClerk.request_bars(order) -> order
```

How the pieces behave, where the spec left it open:

- **`notify_trades` applies the filled trades in the order given.** The Trader must hand them over in the order they were filled: a funding sale before the purchase it pays for (a purchase applied first is refused for cash). It ignores every trade that is not `filled`. A problem about one trade starts with its trade id and a colon (`"T1: buy 13 PG was not applied: ..."`, `"T1: buy 3 PG was not booked: ..."`); the one general problem is `"book: the fills were not booked: ..."` (`book` true, no connected account: the fills are still applied in memory). A fill that is not applied or not booked is **left out of the portfolio that is returned**, so that portfolio is what the book holds. To set a trade's `booked`: it is true when `book` was true, an account was connected, the trade was `filled`, and no problem starts with its trade id. The portfolio is not read from the book again. The price goes to the book as `Decimal(str(price))`; a book refusal (`ValueError`) shows its own message, any other error only its class name (never connection details).
- **`apply_fill` does not check the T+1 lock of a sale** (the tickets are cut to `sellable_shares`). It checks the symbol, side, shares (whole, at least 1), price (finite, above zero, a number), `today` (a `date`, not a `datetime`) and the cash. The lock rules are one code, `_buy_into`, which `after_purchase` now calls too; `after_purchase` is unchanged in behaviour (the tests of `test_portfolio.py`, `test_raise_funds.py` and `test_two_purchases.py` that existed pass untouched).
- **Use `trade_amount` for every amount** (`Fill.amount`, the cash a buy needs, a commission total). Python's `round(x, 2)` goes to the even cent (`round(500.125, 2) == 500.12`) and the account book rounds half up (500.13): with plain `round` the broker's cash, the fill and the booked cash could differ by a cent. See deviation 2.
- **`request_bars`** never raises for a data problem. A symbol with no bar in the range is `[]` plus a note (`"ZZZZ: no stored bar from 2026-11-05 to 2026-11-30"`); a range after the newest stored day gives `[]` for every symbol and a note that starts `bars: no day_aggs_v1 files from ...`. The order is `data_failed` only when refused or when the folder cannot be read. Bars are `{"date", "open", "high", "low", "close", "volume"}`, oldest first, both ends included.
- **`verify_trades` holds a `filled` recommendation trade to its decision** (a decision that says `yes_auto` or `yes_user`, the same symbol, the side the decision's action means, at most the decision's shares, no second filled trade for the record). A funding sale is held to its shape only (a sale, `funds_for` names a buy of the order). `active` and `rejected` trades are held to the shape only. The Trader sets `schema_version` 4 itself when it writes trades (there is no helper for it).
- **`paper_trading_enabled()`** is the T4 check for step 1. Do not read `PAPER_TRADING` any other way.
- **`Run.mode` is a `str`, not a `Literal`**, so a refused run records the mode it was asked for (`"live"`). `Trade.mode` is a `Literal["paper", "backtest"]`.
- A `Trade` accepts `status="rejected"` only with a code of `REJECT_CODES` (the validator the spec asks for). It does not force `reject_code` to be empty on the other statuses, and does not force a `fill` on a `filled` trade: that is `verify_trades`'s job, so a bug in the Trader's code is caught by the guard and does not raise in the middle of a run (rule T7).

### Test helpers in this phase (reuse them, or copy the pattern)

- `tests/test_account_manager_trades.py`: `FakeStore` (the account book without Postgres: records calls, `fail=` makes a booking refuse with a `ValueError`, `crash=` raises a connection error), `manager(**options)` (a real `AccountManager(model=object(), model_id="account-book", ...)`), `holder(cash, **positions)`, `filled(trade_id, symbol, side, shares, price, record_id="r1", **changes)` (the dict of a filled `Trade`), `FakeTrader`, `BarClerk`. `tests/test_account_book.py` imports `filled` from it.
- `tests/test_work_order_trader.py`: `good_order()` (two decisions, three trades: a purchase, its funding sale, a rejected one), `trade(...)`, `bars_order(...)`.
- `tests/test_data_clerk_bars.py`: the `bars_root` fixture (40 weekdays ending 2026-10-23, symbols AAPL, SPY, PG, in the real day-file format), `ForbiddenModel` (a model that fails the test when anything touches it: the way to prove "no model call"), `clerk(bars_root)`, `tools(bars_root)`.
- The spec's `tests/trader_helpers.py` (`decided`, `MarketClerk`, `manager`, `ScriptedModel`, `call`, `say`) is **not written**: it belongs to phase 2. `MarketClerk` must answer `request_quotes` (`data.quotes` = `{symbol: {"price": ..., ...}}`) and `request_bars` (`data.bars`), as `AccountManager.request_prices` and `request_bars` read them.

### Deviations from the spec, with the reason

1. **`bars.py` is changed** (the spec's list of changed files does not have it): `BarStore.load(..., start=None)` leaves out every day before `start`, in the day files and in the REST files. Without it `daily_bars` would read up to 300 day files and cut them afterwards, and the spec says "load only as many day files as the range needs". The default `None` leaves every other caller as it was (the Advisor's and the Critic's tests pass unchanged). The gap message when nothing is found is the old text without `start`, and "no day_aggs_v1 files from START to END under ROOT" with it.
2. **An amount is rounded half up, in `trade_amount`**, not with `round(shares * price, 2)` as section 13 writes it. The account book (`user_account.cents`) rounds half up on the Decimal; Python's `round` rounds half to even on the float. For a price like 500.125 the two differ by a cent, and then the portfolio in memory, the fill's `amount` and the booked cash would disagree. `trade_amount` is the one function; for every amount that is not exactly a half cent the result is the same as `round`. Spec section 6 also says "amounts to cents": phase 2 should use `trade_amount` there.
3. **`daily_bars` raises `FileNotFoundError` when the bar folder does not exist**, and `request_bars` then makes the order `data_failed`. The spec says `data_failed` only "when the store could not be read", and a folder that is not there is that, not "no bars". (An empty folder, or a range with no files, is still a result.)
4. **A `bars` request makes the order `schema_version` 2**, as `metrics`, `case`, `sector` and `verify` do (a request field). The spec says only that an order with trades is 4. Say so if 4 was meant.
5. **`verify_trades` has four checks the spec does not list**: a trade that is not a record, a missing or repeated `trade_id` (`trade_of` would be ambiguous), and for a filled recommendation trade a symbol or a side that is not the decision's. Each is cheap and each is a way the trades could disagree with the decisions. The listed reasons are all there. Phase 2's good orders must satisfy them: ticket `symbol` is the decision's, `side` is `sell` exactly when the decision's action is `reduce` (as spec section 5 already says).
6. **`.env.example` also gets `PAPER_TRADING=true`** (with a comment), not only `TRADER_MODEL`. It documents the switch T4 reads. A test pins both lines.
7. **`paper_trading_enabled()` and `TRUE_WORDS` live in `config.py`**; the spec describes the read in step 1 of `execute.py` only. One place for the rule, tested here, used by phase 2.
8. **`connect` needs both arguments** (`connect(None, None)` is an error), while `__init__` accepts "neither". The spec says "the same checks as `__init__`"; attaching nothing makes no sense. "Forget a profile read from a file" is done with a flag (`_profile_from_file`) set when the `profile` property reads a file nobody gave: a profile given as an object, a path not read yet, or set through the `profile` setter stays. The desk should call `connect` first and set a given profile after.
9. **The docstrings that said "The Trader is not built"** (the work order's module text and `verify_trade`, `AccountManager.record_trade`) now describe the Trader as the design has it (the precondition, the output guard, what `notify_trades` is for). The class does not exist until phase 2, so those sentences are ahead of the code until then. Phase 3 should read them once more.

### Things in the spec that are ambiguous or may be wrong

- **A funding sale can fill and its purchase still be rejected.** Section 7 says the buy is checked first (permission, price inside the limits, would fill now), then the sells fill, then the buy "with the cash now there (the cash rule applies)": if the cash is still short after the sells, the buy can be `rejected` `insufficient_cash` with the sells already filled. Section 11's guard only asks that `funds_for` names a buy on the order, so it lets that through (a rejected buy is still a buy). I kept to the spec. Phase 2 should either make step 1 of the execution check that the proceeds plus the cash cover at least one share, or the guard should require the funded purchase to be `filled`. Decide before the Trader is written.
- **`record_id` of a funding sale.** Section 14 has the field on every `Trade` and section 11 does not say what a funding sale carries. This phase assumes the record id of the purchase it pays for (`trades_of` returns both). The guard does not look for a decision of a funding sale's own.
- **`HAS_DATA` now has `traded`.** That is what section 11 asks; it means `request_data`, `request_quotes` and the others refuse a traded order ("already has its data"), which is right.
- **Statuses elsewhere that do not know `traded` yet (phase 3):** `prometheus/langgraph/desk.py` `after_decision` and the report, and `prometheus/application/recommendations.py` (lines checking `("recommended", "reviewed", "decided")`). `prometheus/agents/critic/review.py` returns silently for `reviewed` and `decided` but gives a `traded` order the message "the Critic needs an order with recommendations (status traded)"; `decisions.decide_trades` gives it "decides on an order the Critic reviewed (status traded)". Neither was changed (not in this phase).
- Section 16 says the replay command's default mode is `backtest`, and section 1 says the setting `mode` default is `paper`. Both are in the spec on purpose (the file default and the replay command's default differ); noted so nobody "fixes" one.

### What could not be done, and what was not verified

- **The Postgres test passed against the real database.** `test_fills_the_trader_reports_are_booked_through_notify_trades_and_the_book_and_the_portfolio_agree` in `tests/test_account_book.py` uses a throwaway account (a deposit, a buy and a sell through `notify_trades(..., book=True)`, the cash, the position and the realized gain, a booking the book refuses, a call without `book`) and reads the three sample accounts before and after. At the start of the phase the database was down (the baseline's 5 errors): the container `prometheus-postgres` had been stopped at 2026-11-14 03:12 UTC and its last start attempt had failed with "error mounting /run/media/jpyrett/980pro/.../sql/pg_hba.conf ... not a directory", because the container was created when the disk was mounted at `.../980pro/` and it is `.../980pro1/` now. I did not start or recreate it. It was up again ("healthy") when I made the last full run, and all 9 tests of the file passed. If it goes down again, `bash scripts/setup_postgres.sh` recreates the container from the current path and keeps the volume `prometheus_pgdata`. I wrote the test before it was reachable, from reading the code, and ran it only once it was.
- **Each guard was broken on purpose once** (cash check in `apply_fill`, the second-filled-trade check in `verify_trades`, the `filled`-only rule of `notify_trades`): the tests failed, and the code was put back (`cmp` shows the files identical). The T1 to T10 exercise of section 18 belongs to phases 2 and 3.
- **`__pycache__` appears under `prometheus/` after a full run even with `PYTHONDONTWRITEBYTECODE=1`.** `tests/fastmcp_tools.py` starts the MCP server in a subprocess whose environment the MCP client restricts (it does not pass that variable on), and that process writes the bytecode in the folders it imports (`prometheus/`, `common`, `mcp`, `rag`, `tools`, `agents/data_clerk`). They are gitignored; I removed them after the last run, and the next phase has to do the same: `find . -name __pycache__ -not -path './.git/*' -exec rm -r {} +` after a full run.
- Nothing was run that calls a paid API or a model. No `AccountManager()`, `DataClerk()`, `Critic()` or `Advisor()` was built with a real model; every one in the tests has `model=object()` or a model that fails the test when touched. `.env` and `secrets/` were not read (`paper_trading_enabled()` and `get_chat_model` read `.env` through `setting()` when the environment says nothing; the tests set the environment first).

## Phase 2: what was built

Done 2026-11-13 to 14 by Sonnet 5.5 (phase 2 of 3). Scope: the Trader itself, spec sections 4 to 10 and 16 (the report and the replay command), the one check the spec added to `verify_trades`, and the tests of section 18 that belong to them. No desk, no runner, no docs: those are phase 3's. Nothing was run that calls a paid API or a model.

**The suite.** Command: `PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/cmu/bin/python -m pytest -q -p no:cacheprovider --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo`.

| run | passed | skipped | errors |
|---|---|---|---|
| baseline, before any change (Postgres was up) | 3274 | 30 | 0 |
| after phase 2 | 3725 | 30 | 0 |

New tests, by file (pytest items): `test_trader_simulator.py` 94, `test_trader_tickets.py` 61, `test_trader_broker.py` 60, `test_trader_tools.py` 45, `test_trader.py` 145, `test_trader_report.py` 20, `test_trader_replay.py` 21, and 5 more in `test_work_order_trader.py`. `tests/trader_helpers.py` is the shared helper module. A full run leaves `__pycache__` under `prometheus/` (the MCP subprocess, as phase 1 said); it was removed after every full run. No `.pytest_cache`.

### What was built, and where

| part | where | what it holds |
|---|---|---|
| fill rules and exits (spec 6, 10) | `prometheus/agents/trader/simulator.py` | PURE functions: `fill_paper`, `fill_backtest` (each returns an `Execution`: filled, active or rejected with a code), `affordable_shares` (the cash cut, to the cent), `calendar_entry_day(now)` and `entry_date(now, bars)` (the two steps of the backtest's one entry date), `bar_on`, `follow` (the exits: returns an `Outcome`) |
| tickets and funding legs (spec 5, 7) | `prometheus/agents/trader/tickets.py` | `build_tickets`, `permission` (the one call to `verify_trade`), `limit_range`, `shown_range` (what the model reads, rounded inward to the cent), `default_order`, `funding_legs` (all or nothing), `funding_trades` (`T1-F1`, ...), `next_number`, `utc_text` |
| the paper broker (spec 6, 7) | `prometheus/agents/trader/broker.py` | `PaperBroker` (the running portfolio, the trades of the run, `place`, `rejudge`, `close_out`, `add`, `reject`), `Prices` (ONE snapshot), `Refused` (a placing the broker will not do, with the sentence to fix it), `cut_note`, `CASH_HINT`. EVERY trade goes through `place` or `rejudge`, whoever places it: that is where T1, T3 and the range live |
| the tools (spec 8) | `prometheus/tools/trader.py` | `TRADER_TOOLS` (the four), `TraderTools`, `Budget`, `budgeted`, `call_tool` (the one function the loop's tools node calls), `loggable`, `describe_tool_error`, `ToolInputError`, `ToolLimitError`, `market_price` |
| the prompt (spec 8) | `prometheus/prompts/trader.py` | `system_prompt(mode, today, max_summary_words)`, `user_message(request_id, mode, n_new)`, `LAST_TURN` |
| the class and the loop (spec 4, 8) | `prometheus/agents/trader/trader.py` | `Trader`, `TEMPERATURE`, `NullLedger`, the LangGraph loop (`run_loop`, `_graph`) with its own sequential tools node |
| the ten steps (spec 4, 9) | `prometheus/agents/trader/execute.py` | `execute` (steps 1 to 10), `Context`, `Notifier` (step 8 as a one-shot callable), `check_summary`, `code_summary`, `funding_card`, `arranged`, `fill_order` |
| report (spec 16) | `prometheus/agents/trader/report.py` | `trade_section(order)`, `header`, `trade_line`, `account_line`, `outcome_line` |
| replay command (spec 16) | `prometheus/agents/trader/replay.py` | `main(argv, clerk, trader, out)`, `load_order`, `default_now`, `moment_of`, `start_fresh`. The ONE Trader module that imports the Data Clerk |
| work order (spec 11) | `prometheus/common/work_order.py` | the new check in `verify_trades`: a `filled` funding sale whose purchase is not `filled` |
| test safety | `tests/conftest.py` | the autouse fixture `no_real_trader_model`: `get_chat_model` in the Trader's module fails like a missing key, so a test can never build the real (paid) model by accident |

### The loop in one paragraph

Step 4 asks the Account Manager ONCE for the quotes (paper) or the bars (backtest) and makes a `Prices` snapshot; a ticket with no price is rejected `price_unavailable` at once. The broker is built over that snapshot and the trades of the run. Active trades of an earlier run are judged again by code (`rejudge`). The funding question (step 5) is asked, once per funded purchase, and only a yes attaches the sells as trades of their own. The model's loop (step 6) runs only when a ticket is `new`; its tools node runs the calls of a turn one after the other. The sweep (step 7) places every `new` ticket (sells first), `close_out` rejects what is left `not_attempted`, and the `Notifier` (step 8) runs the output guard, tells the Account Manager what was filled in the order it was filled, and closes the broker. Backtest: `follow` each filled purchase (step 9). Step 10 writes the records. The model is never needed: `use_model=False`, a model that raises and a model that cannot be built all end with every permitted trade executed.

### How a run reads (which field answers which question)

| question | where |
|---|---|
| Did the Trader run at all? | `order.trader_runs[-1]`. A refused run (PAPER_TRADING off, `live`, an undecided order, no portfolio) still writes one, with `portfolio_before` None and the reason in `errors`; the same text is the one new message on `order.errors` |
| What happened to a trade, and why? | `order.trades[i].status` (`filled`, `active`, `rejected`; never `new` on a finished order), `reject_code` (the closed list), `reason` (code's own sentence) |
| Who placed it? | `trades[i].placed_by`: `model`, `code` (the sweep), or None (rejected before anybody placed it: at ticket time, `not_attempted`, a funding sale) |
| At what price? | `trades[i].fill` (`price`, `shares`, `amount`, `commission`, `date`, `price_source` quote / bar_open / bar_limit, `slippage_bps`). `shares` of the trade is what was filled; `shares_approved` is what the decision allowed |
| What order was it? | `order_type`, `limit_price`. On a rejected-at-ticket trade these are empty; on a `new` ticket they hold the DEFAULT order; after placing, the order as placed |
| What range could it fill in? | `ref_price`, `limit_low`, `limit_high` |
| Which sells paid for a purchase? | `kind == "funding"` and `funds_for` (the purchase's `trade_id`); `trades_of(order, record_id)` returns both; the funding answer is `trader_runs[-1].funding` {status, approved, note} |
| Was it booked in the account book? | `trades[i].booked` and `trader_runs[-1].booked` (always false without `book`, and in backtest) |
| How did a backtest purchase go? | `trades[i].outcome` (status open or closed, the exit, `realized_gain`, `return_pct` as a FRACTION, `sessions_held`, `benchmark_return_pct`, `excess_return_pct`) |
| What did the model say and do? | `trader_runs[-1].tool_calls` ({name, args, outcome ok / invalid / limit / error}; the note is only a word count there), `turns`, `tokens_in`, `tokens_out`, `placed_by_model`, `placed_by_code` |
| Whose summary is it? | `trader_runs[-1].summary` and `summary_by` (`model` when code kept it, else `code`); `dropped` says what was wrong with the model's |
| What went wrong? | `trader_runs[-1].errors` (also copied onto `order.errors`): a model that failed, a quote that was missing, a ledger that could not be written, the output guard's findings, the Account Manager's problems |
| What did the account look like? | `portfolio_before` (the start of THIS run, with the fills of an earlier run that were never booked applied) and `portfolio_after` (what the Account Manager returned); `price_order` is the request id of the quote or bars order |
| Is the order done? | `order.status == "traded"`: it was `decided`, every permitted decision has a trade that is `filled` or `rejected`, at least one permitted decision exists, and the output guard had nothing to say. Otherwise the status stays what it was. `schema_version` is 4 once the order carries a trade |

### Adding a rejection code, or a tool

- **A rejection code.** Add it to `REJECT_CODES` in `record.py` (`tests/test_trader_record.py` pins the list), then use it: `broker.reject(trade, code, reason)`, or in `tickets.build_tickets` for a ticket that cannot be traded, or as the `reject_code` of an `Execution` in `simulator.py`. Nothing else lists the codes: the tools show `reject_code` and `reason` as they are, the report prints them, the ledger writes the trade. Add the test where the code is raised; `test_every_rejection_a_ticket_can_be_built_with_is_in_the_closed_list` and `test_every_code_the_rules_use_is_in_the_closed_list` are the pattern.
- **A tool.** A method on `TraderTools` with a docstring in the Data Clerk's style (an `Args:` section for every argument), its name in `TRADER_TOOLS`, a budget in `Budget` (a count in `limits`, or a rule in `check` as `place_order` has), a sentence in `prompts/trader.py` `SYSTEM`, and the tests: `test_the_model_has_exactly_four_tools_and_each_one_is_documented` pins the list, `test_no_tool_has_an_argument_that_names_a_symbol_a_side_or_a_number_of_shares_and_none_creates_a_trade` is rule T2 and must keep passing. A tool must not take a symbol, a side or a number of shares and must not create a trade.

### Signatures phase 3 needs (exact)

```python
# prometheus/agents/trader/trader.py
Trader(model=None, model_id=None, config_path=None, mode=None, effort="auto", providers="auto", use_model=True, sweep=True, ledger=None)
Trader.execute_trades(order, account, portfolio=None, now=None, funding=None, approver=None, book=False) -> WorkOrder     # the SAME order; ValueError only for a naive `now`
NullLedger()                                     # `.append(record)` keeps nothing: for --no-ledger
TEMPERATURE = 0.0
# prometheus/agents/trader/execute.py
execute(trader, order, account, portfolio, now, funding, approver, book) -> WorkOrder     # what execute_trades calls; the ten steps
# prometheus/agents/trader/report.py
trade_section(order) -> list[str]                # [] when the Trader never ran
# prometheus/agents/trader/replay.py
main(argv=None, clerk=None, trader=None, out=None) -> int      # 0: the Trader ran and the order is traded, decided or reviewed; 1: it refused to run; 2: a file or an argument is not usable
# the Account Manager (phase 1) calls it:  manager.perform_trades(order, portfolio, trader, now, funding=..., approver=..., book=...)
```

What the Trader reads of its arguments: `account.trading_portfolio(portfolio)`, `account.profile.user`, `account.request_prices(symbols)`, `account.request_bars(symbols, start, end)`, `account.notify_trades(portfolio, trades, traded_on, book=)`, and `account.store` / `account.account_id` (to know whether an account is connected). `funding` is read as `funding["status"]`, `funding["target"]` and `funding["plan"]["sells"]` (each `{symbol, shares, price}`); `approver(card)` is called with `{"text", "symbol", "shares", "est_cost", "question", "kind": "funding"}` and must answer True for a yes (anything else is a no).

Things phase 3 must know:

- **`Trader()` needs no key to be built.** Its model is built on first use; with no key the run carries "the Trader's model could not start" in `errors` and the sweep trades. So `trader_factory(order)` may build `Trader(...)` freely. `--no-trader-model` is `Trader(use_model=False)`; `--no-ledger` is `Trader(ledger=NullLedger())`; the mode is `Trader(mode="paper" | "backtest")`.
- **The ledger.** In paper mode the default is `data/ledger/trades.jsonl` through `Ledger`, one line per trade that BECAME filled or rejected in the run (`{"request_id", "recorded_utc", **trade}`). Tests must pass a ledger of their own (the conftest guard fails any test that appends under `data/ledger/`; the Trader lets that `AssertionError` through on purpose).
- **A refused run** leaves the status as it was and adds one message to `order.errors`; to tell it from a run that went through, look at `trader_runs[-1]["portfolio_before"]` (None when refused). `decisions.decide_trades` and the Critic's `review_order` still have to return a `traded` order unchanged (spec section 11, phase 3).
- **A `reviewed` order is accepted** (decisions still pending): the Trader trades the `yes_*` ones, the status stays `reviewed`, and a later call after `record_user_decision` carries on (numbering continues, a trade already on the order is final). The node should still skip the Trader when no decision says yes, as the spec says; called anyway, the Trader makes no trade, leaves the status, and writes a run whose summary says "no trades to place".
- **`book` and backtest.** `book` is refused in backtest mode (a refused run). `book` without a connected account is one message ("book: the fills were not booked: ...") and the fills stay in memory. `notify_trades` is only called when something was filled.
- **The desk's saved JSON works as the replay's input** (it has an `order` key beside `funds`), and `--save` writes the same shape.
- **Docs.** `design/how_to_code_trader.md` above ("How to run it once it is built") names `python -m prometheus.agents.trader.trader --order ...` for the free replay; that is wrong, the command is `python -m prometheus.agents.trader.replay --order FILE --mode backtest --no-model`. `HowToRun.txt` should say so. The replay also has two flags the spec does not list, `--fresh` and `--no-ledger` (deviation 17).

### Test helpers (reuse them)

- `tests/trader_helpers.py`: `decided(name="PG_open", answer=True, mode=None, now=NOW)` (the captured PG_open order, reviewed and decided), `synthetic(*rows, status="decided")` with `row(symbol, action, shares, price, zone=None, limit=None, will_trade="yes_auto", approved=True, valid_until=..., stop=, target=, time_stop=)` (a made-up decided order, one decision per row, for the cases the capture cannot give), `plan(target, *sells, status="plan")` (a RaiseFunds answer: each sell is `(symbol, shares, price)`), `account(cash, **positions)` (Jacob's portfolio: `MSFT=(5, 400.0)` or a bare share count), `MarketClerk(quotes, bars, fail=, refuse=)` (answers `request_quotes` and `request_bars`, records the orders; `.asked("quotes" | "bars")`), `manager(quotes, bars, clerk=, **options)` (a real `AccountManager(model=object(), ...)`), `config(**sections)` (the shipped settings with some changed), `ScriptedModel(script)` (`.seen`, `.bound`, `.texts()`, `.ran_dry`), `BrokenModel`, `call`, `calls` (several tools in one turn), `say`, `answer(summary, (trade_id, status), ...)`, `bar`, `weekday_bars`, `paper_prices`, `backtest_prices`, `make_broker(order, portfolio, quotes=, bars=, funding=, mode=, loaded=)` (a `PaperBroker` over the tickets the Trader builds), `FakeNotifier`, `say_yes`, `other_agents_work(order)` (the bytes rule T10 compares), `MemoryLedger`, `NOW`, `at`.
- `tests/test_trader.py`: `go(order, quotes, bars, model, portfolio, now, funding, approver, book, mode, use_model, sweep, acct, ledger, **trader_options)` runs a Trader over fakes and returns `(order, account, trader, ledger)`; it fails a test whose scripted model ran dry (the Trader hides a model's failure, as it must). `BARS` (a stored-bars set for PG and SPY around the captured decision), `PLACE_ALL` (the model's four calls), `funded_run`, `SCENARIOS`, `last`, `by_id`. `test_trader_report.py` and `test_trader_replay.py` import from it.

### Deviations from the spec, with the reason

1. **`Trade.order_type` and `limit_price` hold the DEFAULT order on a `new` ticket** (the spec has no other place to keep it, and `verify_request` and the sweep both need it). Placing replaces them with the order as placed.
2. **A purchase whose reference price sits so far under the entry zone that no price is left to pay** (the zone's bottom is over `ref x (1 + max_chase_pct)`) is rejected at ticket time with `price_outside_limits`. The spec's list of ticket rejections does not have this case; the nearest code of the closed list was used.
3. **`place_order`'s details.** "limit" with a `limit_price` of 0 takes the default limit price (the docstring says so); an empty `order_type` with a price, and "market" with a price, are refused. The range shown to the model is rounded INWARD to the cent (the bottom up, the top down), the check is against the real 4-place range, so any price the model reads as allowed is allowed.
4. **The budget.** The counts are module constants in `tools/trader.py` (2, tickets + 2, 2; as the Data Clerk's `DEFAULT_CALLS` are), not config keys: the spec gives the numbers in the text. `place_order` is "once per trade_id" for a placing that happened; a call the broker REFUSES (a bad price, a purchase before a sale) does not use the trade's one placing, so the model can correct it, at most `REFUSED_ATTEMPTS` = 3 times per trade.
5. **An argument a tool does not have is REFUSED, not ignored** (`call_tool`): LangChain's schema alone drops it silently, and rule T2 is stronger when `symbol=`, `side=` or `shares=` is an error the model reads. The run's log of calls records such an argument as `(refused)` and a `note` as a word count: the model's words reach the records in two places only.
6. **Funding details the spec left open.** The funding question comes AFTER the prices (step 5) but the sells' symbols must be in the price request, so the legs are worked out before step 4 and attached only on a yes. A funding sale's reference price is the price its plan was made at; its range is a sale's (-1% and +3%); its order is a market order, or with market orders off a limit at the bottom of its range. The purchase's cost for "the cash does not cover it" is the snapshot price plus slippage plus the commission. The sells count toward `limits.max_trades` (the config's comment says "funding sales included"; spec section 4 says tickets only): they are attached only when they fit. Sales of a purchase that was not attempted are `not_attempted`, of one that failed `funding_failed`; sales of a purchase that is `active` are `active` too ("waiting for its purchase") and are filled by the later run that fills it. The estimate of what the sells bring (the dry run before anything is sold) is the SAME fill rule on the same snapshot, so it is exactly what happens: the guard against "holdings sold, nothing bought" does not depend on an approximation. A test sweeps prices and cash and checks it.
7. **The commission.** `costs.commission_per_order` is counted in the cash a purchase needs and written on the fill, but `apply_fill` and `notify_trades` (phase 1, spec section 13) have no commission, so it is NOT taken out of the portfolio's cash. The shipped setting is 0. With a commission above zero the cash check is conservative and the account is not charged: say so if it matters.
8. **`simulator.py` imports `trade_amount` (portfolio) and `market_calendar`** (the entry date needs both). It is still pure: no I/O, no clock, no model.
9. **Returns are fractions** (`return_pct` 0.0077 is 0.77%, as every `_pct` of the desk and as phase 1's record test wrote), rounded to 6 places. `sessions_held` counts the sessions AFTER the entry session through the exit session (0 when the stop fires on the entry bar); an open trade holds the sessions up to the last stored bar. A series that does not start with the entry bar gives `Outcome(status="open")` with nothing in it.
10. **Fill labels.** `price_source` is `bar_limit` whenever the limit set the price (a limit buy whose open plus slippage was over the limit, or one filled at its limit because the day touched it), `bar_open` when the price came from the open. `slippage_bps` on a fill is what the fill really had (a price capped by the limit has less than the setting, a fill at the limit has 0).
11. **`Trader.run_loop`, `NullLedger`, `Prices`, `Execution`, `Refused`, `Context` and `Notifier` are additions** (the spec names none of them): the loop is a method so that `execute.py` stays the steps; `NullLedger` is the runner's `--no-ledger`.
12. **A resting order is not made final by an outage.** An `active` trade of an earlier run whose price cannot be had in this run (the quote or bars request failed, or no quote for the symbol) stays `active` with the reason, and is judged again in the next run; a NEW ticket with no price is `price_unavailable` as the spec says. The permission is still checked first: an approval that ran out rejects the resting trade.
13. **Nothing is told to the account when nothing was filled** (`notify_trades` is not called with an empty list: no spurious "book:" message). Everything else of step 8 runs.
14. **`AssertionError` is not swallowed by rule T7** (`execute` re-raises it; the ledger step catches only `OSError`, `ValueError` and `TypeError`). An assertion is a programming mistake, and the conftest guard against writing the real ledger is an assertion: swallowing it would make a test that writes the real ledger pass.
15. **Active trades of the other mode** (a paper trade, replayed in backtest) are left exactly as they were, with a message; they are not judged. The unbooked FILLS of any mode are applied to the next run's starting portfolio.
16. **The status becomes `traded` only when at least one permitted decision exists** and the output guard passes. An order `decided` with no permitted decision (all denied) keeps `decided`; the Trader writes a run saying so.
17. **The replay has two flags the spec does not list.** `--fresh` ignores the trades the order already carries (a saved order a desk run already traded would otherwise replay as "nothing new", rule T3), `--no-ledger` gives a paper replay a throwaway ledger (otherwise every paper replay appends to `data/ledger/trades.jsonl`). Its exit code is 0 when the Trader ran, 1 when it refused, 2 for a bad file or argument.
18. **`verify_trades` check added** (the spec's one change): a `filled` funding sale whose purchase is not `filled`. It names the sale and the purchase's status. Phase 1's test `test_a_filled_sale_of_a_reduce_decision_is_a_trade_the_decision_allowed` built a filled funding sale paying for a REJECTED purchase; it now drops the funding sale (a sale of a `reduce` has none), and two tests were added.
19. **The conftest fixture `no_real_trader_model`** (not in the spec): see the table above.

### Things in the spec that are ambiguous or may be wrong

- **Section 6, commission against section 13's `apply_fill`**: the spec's cash rule subtracts the commission but the portfolio functions do not charge it (deviation 7).
- **Section 5, "`today` is `eastern_date(now)`" in a backtest.** The tickets are built with the T+1 rule of the decision's day, while the fill happens on the entry date (usually the next session). A position bought on the decision's own day is locked at ticket time and is rejected `not_sellable_today`, though by the entry date it would be sellable. Conservative and as written; the broker re-checks against the entry date for shares bought in the run.
- **Section 4, step 3: "Tickets beyond `limits.max_trades`"** counts decisions in order, a rejected ticket included.
- **Section 7 asks the question once per funded purchase "in Ask mode and in Auto mode"**: the Trader does not know the mode, it always asks (when the cash does not cover the purchase and a plan is valid).
- **`verify_trade`'s reasons embed fields of the order** (the decision's `will_trade`, the review's `valid_until_utc`). They are code's sentences and the spec says they are fine for the model to read, and they are on the order as it is. On an order file somebody edited by hand, those two fields could carry text. If rule T9 should hold against that, `verify_request` can show the `reject_code` alone (the spec's example does) and not the `reason`; `place_order`'s result carries the reason as the spec says.
- **`PAPER_TRADING=` set to the empty string counts as unset** (`rag.config.setting` treats an empty value as missing, and the default is true). Anything else but true, yes, on, 1 refuses the run. Phase 1's `paper_trading_enabled()` is unchanged.
- The tickets read `verify_trade`'s wording ("the approval ran out ...") to tell `approval_expired` from `not_permitted`: a test (`test_an_approval_that_ran_out_is_approval_expired_but_only_when_that_is_the_one_reason`) fails if `verify_trade`'s sentence changes.

### Rules T1 to T10: each guard broken once

Each guard below was changed on purpose, the named test failed, and the file was put back (a script restored the original bytes and checked them; the whole suite was run again afterwards: 3725 passed).

| rule | what was broken | file | first test to fail |
|---|---|---|---|
| T1 | the permission check inside placing (`self.permit(trade)` ignored) | `broker.py` | `test_a_trade_that_stopped_being_permitted_after_its_ticket_was_built_is_rejected_when_it_is_placed` (and `test_the_approval_that_runs_out_between_the_ticket_and_the_placing_is_caught_again_by_the_broker`) |
| T1 | the permission at ticket time (`verify_trade` not called) | `tickets.py` | `test_a_decision_whose_review_did_not_approve_is_not_permitted_and_the_reasons_of_verify_trade_are_the_reason` |
| T2 | an argument the tool does not have is dropped, not refused | `tools/trader.py` | `test_an_argument_a_tool_does_not_have_is_refused_not_ignored_and_nothing_is_traded` |
| T2 | the limit price may be anywhere | `broker.py` | `test_an_order_the_model_may_not_give_is_refused_with_the_way_to_fix_it_and_nothing_is_placed` |
| T3 | a trade that is not `new` can be placed again | `broker.py` | `test_a_trade_is_placed_once_a_filled_a_rejected_and_a_resting_trade_cannot_be_placed_again` |
| T4 | `PAPER_TRADING` not checked | `execute.py` | `test_paper_trading_set_to_anything_but_true_refuses_the_whole_run_and_changes_nothing_else` |
| T4 | any mode accepted (`live`) | `execute.py` | `test_a_mode_of_live_or_anything_else_is_refused` |
| T5 | `tickets.py` imports the Data Clerk | `tickets.py` | `test_the_traders_modules_import_nothing_from_the_data_clerk_the_critic_or_the_advisor_the_account_manager_is_the_only_door` |
| T6 | no sweep | `execute.py` | `test_a_model_that_places_nothing_leaves_the_sweep_to_do_it_and_the_sweep_notifies_the_account_manager` |
| T6 | a model's failure ends the run (re-raised) | `trader.py` | `test_a_model_that_raises_never_stops_a_permitted_trade_and_the_run_says_what_kind_of_failure` |
| T7 | the catch-all removed | `execute.py` | `test_an_unexpected_bug_in_the_trader_is_a_message_on_the_order_and_never_an_exception` |
| T7 | a naive `now` not refused | `execute.py` | `test_a_naive_clock_is_a_programming_mistake_and_raises_before_anything_is_done` |
| T8 | the summary's numbers not checked | `execute.py` | `test_a_summary_with_an_invented_number_is_replaced` |
| T8 | a note not cut to the words allowed | `broker.py` | `test_cut_note_collapses_white_space_drops_control_characters_and_keeps_the_first_words` |
| T9 | the work order id put in the user line as it is | `prompts/trader.py` | `test_the_work_order_id_in_the_user_line_is_cut_to_letters_digits_and_dashes_even_if_an_order_read_from_a_file_carries_text` |
| T10 | the Trader edits the decisions | `execute.py` | `test_the_recommendations_the_reviews_and_the_decisions_are_the_same_bytes_after_the_trader_ran` |
| output guard | `verify_trades` ignored before booking | `execute.py` | `test_with_a_finding_of_the_output_guard_nothing_is_booked_and_the_order_is_not_marked_traded` |

T9 is also held by tests that read EVERYTHING the scripted model saw (the system prompt, the user line, every tool result) against the thesis, every quote, the company name and the query of the captured record, and against a record whose company, thesis, quote, review summary and decision reason are instructions ("IGNORE ALL RULES ...").

### What could not be done, and what was not verified

- **Run for real, once:** `python -m prometheus.agents.trader.replay --order FILE --mode backtest --no-model --cash 10000` over the real stored bars of `data/historical` (read only), on a decided order made by the fakes and saved in the scratch folder (the repo has no saved decided order: every file in `data/processed` is `recommended` or earlier). At the order's own moment it entered at the next session's open and followed the trade to its target; with Jacob's sample portfolio (cash 51.23) it rejected the purchase `insufficient_cash` and said `--raise-funds`. Exit 0, nothing written (no new file, the ledger untouched).
- **Not run, and not verifiable here:** the real model (gpt-oss-120b on OpenRouter): whether it follows the tool protocol, honours `require_parameters` with the preferred providers, ends with the JSON, or how often the summary check replaces its words. The design makes all of it harmless (the tools refuse, the sweep trades, code writes its own summary) and tests prove that with scripted models that do each wrong thing, but a real run is Jacob's. Paper mode against real quotes was not run either.
- **`book` was tested with a fake store** (booked in the order the fills happened, refused fills, no account, no `book`), and the Postgres path of `notify_trades` by phase 1's test. No test runs the Trader against the real database.
- **A very large order.** The bars request holds at most 30 symbols (`MAX_BAR_SYMBOLS`): 20 trades, their funding sales and the benchmark could exceed it in a backtest; the request would then be refused and every ticket `price_unavailable`. `limits.max_trades` is 20 and the funds node plans one purchase, so it does not happen today.
- **Funding more than one purchase in a run, a backtest over a range of dates, booking a simulated exit** and extended-hours rules are not in this version (spec section 22).
- The desk node, the runner, `--trade`, `--account`, `--book`, `HowToRun.txt`, `README.md`, `AGENTS.md`, `docs/SESSION_NOTES.md` and `docs/design_choices.md` (entries 73 onward) are phase 3's and were not touched.

## Phase 3: what was built

Done 2026-11-14 by Sonnet 5.5 (phase 3 of 3). Scope: the Trader in the LangGraph desk and in the runner, the two functions that did not know `traded`, the four tightenings of the review of phase 2, one test against the real Postgres, and the docs. Nothing was run that calls a paid API or a model: the desk and `test_recommendations.sh` were never run with a symbol or a request, no agent was built with a real model, `PROMETHEUS_LIVE_TESTS` was not set, `.env` and `secrets/` were not read, and nothing was written under `data/historical/` or `data/processed/`. What was run: the test suite, `desk --show-graph`, `desk --help`, the runner's `--help`, the script with `PROMETHEUS_DRY_RUN=1` and `--help`, and the parser's refusals (which stop before anything is built).

**The suite.** Command: `PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/cmu/bin/python -m pytest -q -p no:cacheprovider --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo`.

| run | passed | skipped | errors |
|---|---|---|---|
| baseline, before any change (Postgres was up) | 3725 | 30 | 0 |
| after phase 3 | 3823 | 30 | 0 |

98 new tests, by file (pytest items): `test_desk_trader.py` 51, `test_recommendations_trader.py` 19, `test_recommendations_script.py` 9, `test_trader.py` 9, `test_trader_config.py` 3, `test_trader_tools.py` 2, and one each in `test_trader_record.py`, `test_trader_tickets.py`, `test_critic.py`, `test_account_manager_decisions.py` and `test_account_book.py` (the real-Postgres test, which ran and passed: the container was up). Changed, not new: five assertions of `test_trader_tools.py` and one of `test_trader.py` that pinned a tool result's `reason` (see D1). A full run leaves `__pycache__` under `prometheus/` (the MCP subprocess); it was removed after every run, with the command of the brief. No `.pytest_cache`.

### What was built, and where

| part | where | what it holds |
|---|---|---|
| A. the desk | `prometheus/langgraph/desk.py` | `build_desk(..., with_trader, trader_factory, store_factory)` (`with_trader` without `with_critic` is a `ValueError`); the `trader` node and `permitted`, `after_decision`, `after_funds` (the edges, with and without the funds node); `open_book` (the lazy import of the Postgres store) and `saved_account` (connect first, profile after, the saved portfolio, a database failure becomes a `RuntimeError` that names its kind); `account(state)` takes a saved account when the request says `account: {"account": name or number}`; `report()` (the TRADER section, `trader` in the timing, the ACCOUNT line, `traded` is a finished path); the flags `--trade [paper\|backtest]`, `--no-trader-model`, `--account`, `--book` and their checks; the exit code of a `--trade` run; `--show-graph` draws the trader node |
| B. `traded` | `prometheus/agents/account_manager/decisions.py` (`decide_trades`), `prometheus/agents/critic/review.py` (`review_order`) | both return a `traded` order unchanged and without a message, as for a `decided` one (the Critic: `reviewed` too) |
| C. the runner | `prometheus/application/recommendations.py`, `test_recommendations.sh` | `run(..., trader=None, funding=None, book=False)` calls `manager.perform_trades` after the decisions; the TRADER section; `traded` is a finished order; `--trade [paper\|backtest]`, `--no-trader-model` (and `--no-ledger` gives the Trader a `NullLedger`); the script passes them through and validates them |
| D1. no `reason` in a tool result | `prometheus/tools/trader.py`, `prometheus/prompts/trader.py` | `_view`, `_status` and `place_order` give the `reject_code` and numbers; the docstrings and the system prompt say so |
| D2. a ticker | `prometheus/agents/trader/record.py` (`is_ticker`, the validator on `Trade.symbol`), `tickets.py` (`build_tickets(..., problems=None)`), `execute.py` (step 3 names it in the errors) | a decision whose symbol is not a ticker gets no ticket and no number |
| D3. commission 0 | `prometheus/agents/trader/config.py` | `costs.commission_per_order` must be 0, with the reason in the message |
| D4. groups of 30 | `prometheus/agents/trader/execute.py` (`groups`, `_prices`, `_snapshot`), `broker.py` (`Prices.unavailable`) | the price request is made in groups of at most 30 symbols (`MAX_QUOTE_SYMBOLS`, `MAX_BAR_SYMBOLS`) and the answers are joined |
| E. real Postgres | `tests/test_account_book.py` | `test_the_trader_in_paper_mode_books_its_fill_in_a_throwaway_account_and_the_three_samples_do_not_move` |
| F. docs | `docs/design_choices.md` (entries 73, 74, 75), `HowToRun.txt` (PART 12; two stale lines of PARTS 10 and 11), `README.md`, `AGENTS.md`, `docs/SESSION_NOTES.md`, the docstrings (work order module text and `verify_trade`, `AccountManager.record_trade` and `__init__`, `decisions.py`, `tickets.py`, `execute.py`) | |

### How it behaves

- **The graph.** `... -> critic -> account_manager_decision -> trader -> END`; with the funds node `... -> account_manager_decision -> account_manager_funds -> trader -> END`. The trader node runs only when the order is `decided` or `reviewed` and at least one decision says `yes_auto` or `yes_user`; otherwise the edge goes to END (with the funds node, the funds node still runs first, as it did, and answers "refused" when nothing was approved). Without `with_trader` the nodes and every edge of the four older shapes are pinned by a test.
- **The node** is the spec's: `account(state)`, the factory, `manager.perform_trades(order, portfolio, agent, clock(), funding=state.get("funds"), approver=approver, book=...)`. `trade: {"mode", "book", "model"}` in the request: the default Trader is `Trader(mode=..., use_model=...)`; a factory a test gives is used as it is. `model` is a bool.
- **A saved account.** In `account(state)`: the store (`store_factory`, else `open_book`), the id (`account_id_for_seed`, or the number when the text is digits), `manager.connect(...)`, then a profile only if one was given, then `manager.list_portfolio()`. Any error that is not a `ValueError` or a `RuntimeError` becomes `the account book could not be read (<ExceptionName>)`; a missing password file becomes "the account book is not set up ..." with no path; a test feeds a connection error that carries a host, a port, a user and a password and checks none of them appears on the order. The funds node, with a saved account, plans for that account's holdings (not for a JSON file), because the Trader will trade those.
- **The runner.** After `decide_trades`, `run` calls `manager.perform_trades` when a `trader` is given and at least one decision permits a trade; otherwise it adds a note ("the Trader did not run: ...") so that `--trade` is not silently ignored. The default manager now holds the runner's Data Clerk (`AccountManager(model=object(), clerk=clerk, ...)`), so the Trader's quotes and bars come through the same clerk. Nothing is booked from the runner: it holds no saved account.
- **The report.** The TRADER section follows the Critic's (and the funds section when there is one); the runner's last line says the trades are simulated and the Trader can only place what was decided.

### Deviations from the spec, with the reason

1. **`trade.model` is a bool** (use the model: false for `--no-trader-model`). The spec's request carries `{"mode", "book"}`; the brief added `"model"` and did not say what it holds.
2. **`--account` needs `--critic`** (a parser check the spec does not list). Without the Critic nothing reads the account, so the flag would be silently ignored.
3. **`--profile`'s default is now `None`** (the help says the sample profile is used). With `--account` the account's own row is the profile unless `--profile` is given, and the desk has to tell "not given" from "the default file"; without `--account` the request carries `str(DEFAULT_PROFILE)` exactly as before (a test pins the request and the options of a `--critic` run without `--trade`).
4. **With `--account` the funds node is given no `portfolio`** and plans for the saved holdings. The spec says nothing about the funds node and a saved account; a plan for the JSON file's holdings would have sells the Trader then refuses (`funding_legs` checks they are holdings).
5. **The PATH line.** `traded` is a finished path as the spec says; so is any run in which the `trader` node ran (an order that stays `decided` with a resting trade, or `reviewed` with a decision still pending, is not "stopped here" either). Without `--trade` the line is as it was.
6. **The ACCOUNT line of the report** (not in the spec): a saved account is written to (the decisions, always; the fills, with `--book`), and the report says so (see "Found" 2).
7. **`Prices.unavailable`** (D4, `broker.py`): when only some groups of a price request fail, each symbol of a failed group is `price_unavailable` with the group's reason and a message on the run; when all fail the snapshot has `failed`, as before. `run.price_order` and `Prices.order_id` are the groups' request ids joined with ", " (still one string). One more difference from phase 2: a backtest whose bars request raised now also puts the reason on `run.errors` (before, only on each trade).
8. **`is_ticker` is stricter than `check_symbol`** (D2): the symbol must already be in capitals with no space round it (`check_symbol` would clean " pg " to "PG"). A trade holds exactly the symbol its decision holds, because the output guard compares them. `Trade.model_validate` (a saved order whose trade carries a sentence) now fails too, which is the point.
9. **`check_trade_status` of an `active` trade shows `limit_price`** (D1), where it showed `reject_code: null` and the reason: a number code made, which tells the model what the resting order waits for. A rejected trade shows `reject_code`; nothing shows `reason`. Refusal messages are as they were.
10. **The commission code stays in `broker.py` and `simulator.py`** (D3): with the loader refusing any value but 0 it is unreachable from a file, and the tests that exercise it build their settings with `model_copy` (which skips validation). Left in, so that the version that charges a commission has its arithmetic; say so if it should come out.
11. **The script's `--trade`** takes the next word as its mode only when it is `paper`, `backtest` or `live` (`live` is read so it can be refused with "there is no live trading"); any other word after it is the request, so `./test_recommendations.sh --trade PG` works. A bare `--trade` is always passed on as `--trade paper`.
12. **Two stale lines of `HowToRun.txt`** (PART 10: "the Trader is not built"; PART 11: "the desk still reads the JSON ...") were corrected; the older long text of `README.md` ("the Critic, the Trader and a full Account Manager are next") and the line "Not built: ... the Trader" in section 9 of `docs/SESSION_NOTES.md` were left as history, and the new line below it says the Trader is built.

### Found in phases 1 and 2 (and in the desk), for the spec's author

1. **FIXED 2026-11-14 (Jacob asked): every `--critic` run of the desk without `--trade` exited 1 and printed "(stopped here)".** The rule was `order.status == "recommended"`, and a Critic run ends `decided` or `reviewed`. Now one function, `desk.reached_its_end(state)`, says whether the run went as far as the desk it was built as goes, and both the PATH line and the exit code use it: a Critic run ends at the decision (`decided`, or `reviewed` while a decision waits for the user); with nothing recommended the Critic is never asked and `recommended` is finished; an order that stays `recommended` after the Critic ran (it could not start) is NOT finished; a Trader desk ends at the trader node, and a decided order the Trader had nothing to do with still "stopped". The desk's state gains `with_trader` (set by `run_desk`) so the report can tell the two desks apart. The exit code of a `--trade` run is unchanged (spec section 16). Tests: `test_desk_critic.py` (four) and `test_desk_trader.py` (the exit-code table); six of them fail with the old rule.
2. **`desk --critic --account X` writes to the account book even without `--book`.** `AccountManager.decide_trades` saves the verdicts whenever the manager has a store (entry 72), and a connected manager is what `--account` builds. The decisions go into `recommendation_decisions` with the recommendation's own `record_id` (so every run adds rows), and `tests/test_account_book.py::test_the_sample_accounts_match_the_seed` pins the three samples' approved and denied counts for 2026: **a live run on `jacob_sample` would break that test.** Not changed (it is entry 72's design); the report, the help, `HowToRun.txt` and `AGENTS.md` say it, and live runs should use a throwaway account (below) or `bash scripts/setup_postgres.sh --reseed` after.
3. **The desk has no `--no-ledger`.** A paper `desk --trade` appends its trades to `data/ledger/trades.jsonl` (the runner's `--no-ledger` gives a `NullLedger`). Spec section 16 asks for the flag on the runner only.
4. **The Trader does not know the trade mode** (Ask or Auto), so the funding question is always asked (known, phase 2); it is also why a funded purchase in Auto mode still stops for a yes.
5. **The phase 2 notes name the wrong replay command** in "How to run it once it is built" (it says `...trader.trader --order ...`; the command is `python -m prometheus.agents.trader.replay`). Phase 2 said so itself; `HowToRun.txt` PART 12 has the right one.
6. **`desk --trade` overwrites `data/processed/desk_<symbol>.json`**, as `--critic` already did. That folder is "saved evidence the docs cite": save under another name with `--json > file` if an earlier run matters.
7. **The Trader has no way to change its provider list or reasoning effort from a command** (`Trader(effort=, providers=)` exist; nothing passes them). If a fast host does not support tool calls with `require_parameters`, the run says "the Trader's model could not ..." or the model call failed, the sweep trades, and the only fix is code.

### Each new guard broken once

Each change below was made on purpose, the first test named failed, and the file was put back (copies kept in the scratch folder; `cmp` shows the originals identical). 21 changes to the desk, the runner and the script, and 7 to the four tightenings and B.

| what was broken | file | first test to fail |
|---|---|---|
| `with_trader` without the Critic allowed | `desk.py` | `test_a_trader_without_the_critic_is_refused_whatever_else_is_asked_for` |
| the trader node runs with nothing permitted | `desk.py` | `test_the_trader_does_not_run_when_nothing_is_permitted_and_the_graph_ends_at_the_decision` |
| the default profile set before `connect` (the account's row never read) | `desk.py` | `test_a_saved_account_gives_the_profile_and_the_portfolio_and_the_run_is_connected_before_a_profile_is_set` |
| a database error's own text in the message | `desk.py` | `test_a_database_failure_is_a_message_that_names_the_kind_and_never_the_connection_details` |
| the exit rule of `--trade` removed | `desk.py` | `test_trade_with_no_value_is_paper_and_the_trade_part_of_the_request_says_...` |
| `--book` without `--account`; `--book` in a backtest | `desk.py` | `test_the_flags_that_do_not_go_together_are_refused_by_the_parser_before_anything_is_paid_for` (two cases) |
| `trade.model` ignored | `desk.py` | `test_without_a_factory_the_node_builds_the_trader_from_the_trade_part_of_the_request` |
| no TRADER section in the report | `desk.py` | `test_the_report_has_the_trader_section_after_the_critics_and_the_trader_in_the_timing_and_the_path_is_finished` |
| the funding plan or the approver not passed to the Trader | `desk.py` | `test_the_funding_plan_of_the_funds_node_reaches_the_trader_...` and `test_the_trader_is_given_the_clock_the_approver_and_the_order_...` |
| the funds node ignores a saved account; funds -> END (no trader after it) | `desk.py` | `test_the_funds_node_plans_for_the_saved_accounts_holdings_...` and `test_with_the_funds_node_the_funds_node_runs_first_...` |
| the runner trades without a yes; passes no funding; the report omits the section or does not know `traded`; `--no-ledger` does not reach the Trader; `--trade` with `--no-critic` | `recommendations.py` | `test_a_decision_that_is_still_waiting_for_the_user_...`, `test_the_funding_answer_the_approver_and_book_reach_the_trader`, `test_the_report_has_the_trader_section_after_the_critics_...` (two), `test_trade_alone_is_a_paper_trader_...`, the refusal test |
| the script drops the mode; does not refuse `--trade` with `--no-critic` | `test_recommendations.sh` | `test_trade_alone_is_paper_...`, `test_bad_trader_options_are_refused_...` |
| B: `decide_trades` and `review_order` do not return a `traded` order | `decisions.py`, `review.py` | one test each |
| D1: `reason` back in a tool result | `tools/trader.py` | `test_verify_request_shows_a_rejected_ticket_as_final_with_its_code_only_...` |
| D2: the validator off; the ticket made for a bad symbol | `record.py`, `tickets.py` | `test_a_trades_symbol_must_be_a_ticker_...`, `test_a_decision_whose_symbol_is_not_a_ticker_gets_no_ticket_...` |
| D3: any commission accepted | `config.py` | `test_each_value_this_version_refuses_is_an_error_naming_the_setting` |
| D4: one request for all symbols | `execute.py` | `test_a_run_with_more_than_30_symbols_asks_for_the_quotes_in_groups_of_30_and_joins_the_answers` |

(The fake Data Clerk of the D4 tests refuses what the real one refuses, an order of more than 30 symbols; without that the tests would pass without the fix.)

### What could not be done, and what was not verified

- **Run for real, once, not by me:** nothing. In particular not verified: that gpt-oss-120b on Cerebras or Groq with `require_parameters` follows the tool protocol and ends with the JSON; paper mode against real quotes; a desk run with a real Postgres account (the real-database test uses `notify_trades`/`execute_trades` directly, not the desk); `desk --trade` in a terminal (the approver's `input` prompt, twice). Everything the model could do wrong is covered with scripted models; the real thing is Jacob's first run.
- **The runner has no funds node and no saved account**, so `funding` and `book` are parameters of `run` that the command never sets (tested through `run`, as the spec's "`run(..., trader=None, funding=None, book=False)`" asks).
- **Funding more than one purchase, a backtest over a range of dates, booking a simulated exit** and extended-hours rules stay out (spec section 22).
- **A backtest from the desk at the time of a decision** usually finds no stored bar from the next session on, so its orders wait (`ACTIVE`); the useful backtest is the replay of a saved order once the later bars are stored (`scripts/sync_day_aggs.py`, which writes `data/historical/` and is Jacob's to run).

### First live run: a checklist for Jacob

In this order, cheapest first. A "paid" step needs your yes (AGENTS.md); the free ones do not.

1. **Free, now.** The suite (the command at the top of this file: 3823 pass). `python -m prometheus.langgraph.desk --show-graph --critic --raise-funds --trade` (the funds node before the trader node). `./test_recommendations.sh --help` (the THE TRADER paragraph). `bash scripts/setup_postgres.sh` if the container is down, then `python -m prometheus.agents.account_manager.account_manager --account jacob_sample` (free: no quote, no model).
2. **A throwaway account for every run that uses `--account`** (free, local): `python -c "from prometheus.persistence.accounts import open_account_store as o; s=o(); i=s.create_account('Trade Test','moderate'); print(i)"`, then `python -c "from decimal import Decimal; from prometheus.agents.account_manager.account_manager import AccountManager as A; from prometheus.persistence.accounts import open_account_store as o; A(model=object(), store=o(), account_id=ID).add_funds(Decimal('10000'))"` (ID is the number printed). Delete it afterwards with `o().delete_account(ID)`. Do not use `jacob_sample` for a live run (Found 2).
3. **The cheapest paid run: no Trader model, backtest, no quote of the Trader's own.** `./test_recommendations.sh PG --no-fetch --no-ledger --mode auto --no-ask --trade backtest --no-trader-model` costs the Advisor's two calls, the Critic's Finnhub calls and GLM's call, and nothing for the Trader. Look at: "the Trader executes ..." in the progress lines; the TRADER header says `no model: the program placed the orders; BACKTEST mode, prices: the stored daily bars`; a trade that is `ACTIVE` with "waiting for the next session" (the next session's bar is not stored yet) or `FILLED ... bar_open` / `bar_limit` with an OUTCOME line when it is; "the Trader did not run: no decision of this order permits a trade" when the Critic denied the recommendation or Ask mode had nobody to answer (use `--mode auto` and a profile whose `auto_min_score` the score reaches, or answer yes in Ask mode, without `--no-ask`).
4. **Paper, no model** (Finnhub quote only): `python -m prometheus.langgraph.desk --symbol PG --critic --trade --no-trader-model` (it also fetches news: use `--no-fetch` to keep to the stored news). Look at: one quote asked per symbol, the fill price a little above the quote (5 basis points) or at the limit, `placed by code`, `account  cash A -> B ... kept in memory only`, and a new line in `data/ledger/trades.jsonl` (the desk has no `--no-ledger`: that file is real).
5. **Paper, with the model: the check that matters** (`--trade` alone). OpenRouter, a cent or two. In the report: the header names `model openai/gpt-oss-120b at temperature 0` (or says "no model could run" and why: then look at `trader note:` lines). In `--json`, `order.trader_runs[-1]`: `tool_calls` should read `verify_request, place_order, check_trade_status, notify_account_manager`, every `outcome` `ok`; `placed_by_model` equal to the number of trades and `placed_by_code` 0; `summary_by: "model"` and `dropped: []`; `turns` 4 or 5; `errors: []`. **Whether the real model's tool calls work on the fast hosts is exactly this**: if the run says the model call failed (an error kind such as `NotFoundError` or `BadRequestError`, "No endpoints found that support tool use"), the sweep traded anyway and the report says so; the Trader's `providers` and `effort` arguments are not reachable from a command (Found 7), so the fix is in code. If `placed_by_code` is not 0 the model skipped a trade and code placed it, which is also fine and also worth knowing. If `summary_by` is `code`, `dropped` says what was wrong with the model's summary (a number it made up, a wrong status, too many words): this is how often T8 fires.
6. **Funding:** `--raise-funds --trade` on Jacob's sample portfolio (cash $51.23), `--no-fetch`. Answer **n** to the sells first: no holding is sold and the purchase is `rejected insufficient_cash`, the funding line says "(you said no)". Then run again and answer **y**: the sells are listed under the purchase, `FILLED`, the purchase `FILLED`, the cash after is what is left. A purchase the cash covers asks nothing.
7. **A saved account, booked** (the throwaway of step 2): `python -m prometheus.langgraph.desk --symbol PG --critic --trade --account ID --book --no-fetch`. Look at the ACCOUNT line (decisions are saved, fills booked), `booked` on the trade, then `account_manager --account ID` shows the cash and the position; delete the account afterwards.
8. **A backtest of that decision, free,** once the next session's bars are stored: `python -m prometheus.agents.trader.replay --order data/processed/desk_PG.json --mode backtest --no-model --cash 10000` (or `--portfolio FILE`); it enters at that session's open and follows the trade to its stop, target or time stop, with SPY alongside. `--yes-funding` answers the funding question when there is a plan.

## First live run: what was run (2026-11-14, Sonnet 5.5 at Jacob's go-ahead)

*(The first part. Steps 3, 4, 6, 7 and 8 were blocked then; the data was refreshed afterwards and they ran: see "First live run, second part" below.)*

The checklist above, in its order. Fetching stayed off (`--no-fetch`) everywhere, so nothing paid was downloaded and nothing was written to `data/`; a throwaway Postgres account (id 54) was made and deleted; the three sample accounts were checked before and after (cash 51.23, approved 1, denied 1: unchanged); `data/ledger/trades.jsonl` was not created.

| step | result |
|---|---|
| 1 free checks | done: the suite (3,834 pass after the exit-code fix below), `--show-graph --critic --raise-funds --trade`, `./test_recommendations.sh --help`, Postgres up, `--account jacob_sample` |
| 2 throwaway account | done: account 54, $10,000, deleted afterwards |
| 3 cheapest paid run | **ran, but no trade was permitted, so the Trader did not run.** See "What blocks a full live run" |
| 4 paper, no model, desk | not run: needs an approved decision (blocked, same reason) |
| 5 paper, with the model | **done in a harness, twice** (below) |
| 6 funding, y and n | not run: needs an approved decision and the funds node (blocked, same reason) |
| 7 saved account, booked | **done in a harness** (below); the desk's `--account ... --book` itself is blocked, same reason |
| 8 free replay | not run: there is no saved order with a permission (the one step 3 saved was denied) |

### Step 5 in a harness: the real model on the fast hosts

`Trader(mode="paper")` with its real model (gpt-oss-120b through OpenRouter, temperature 0, `require_parameters`, Cerebras then Groq), over the captured PG order made by the test fakes, a fake quote and a memory ledger: no Finnhub, nothing written. About 17,000 tokens in all.

- **One purchase** (PG, 44 shares): `verify_request`, `place_order T1`, `check_trade_status`, `notify_account_manager`, every call `ok`; `placed_by_model` 1, `placed_by_code` 0; `summary_by: "model"`; `dropped: []`; `errors: []`; 5 turns. Filled 44 at 167.10, cash 10,000.00 to 2,647.60.
- **A sale and two purchases** (MSFT sale, KO and PG purchases, KO's quote 16% above its judged price): the model placed the SALE first, then the purchases; KO was `rejected price_outside_limits` (the price had left what the Critic judged), the other two filled; `placed_by_model` 3, `placed_by_code` 0; its summary named only real numbers and was kept; 7 turns; `errors: []`.
- So on this evidence the tool protocol works on the fast hosts with `require_parameters`, the model follows "sales before purchases", and the sweep was not needed. Two runs are not a measurement: how often T8 drops a summary, or the model skips a trade, needs more runs (cents each).

### Step 7 in a harness: the real model and the real local Postgres

The same PG order, `book=True`, an `AccountManager` connected to throwaway account 54. The model placed the purchase; `notify_trades` booked it. Before: cash 10,000.00, no positions. After: cash 2,647.60, PG 44 at cost 167.10, 1 trade booked, `booked: true` on the trade. The three sample accounts unchanged. The account was deleted.

### What blocks a full live run (the desk, with the real Critic)

The Trader only trades what the Critic approved, and today nothing can be approved from stored data:

1. **The stored bars end 2026-11-11.** Reviewed as of 2026-11-13 the Advisor's `freshness` gate fails ("the newest bar is 2026-11-11, 2 sessions before"), so nothing is recommended and the Trader is never asked.
2. **A review replayed in the past is denied.** Reviewed as of 2026-11-10 10:30 (`--now`), the Advisor recommended PG (score 76, open, 78 shares) and the Critic ran live (GLM 5.3, 44 s; 27 of 27 values matched what the Data Clerk fetched again, 3 of 3 sources read again, accuracy 30/30, chart 10/10), and then DENIED it: `quote_stale`, because the quote is Finnhub's live one (Friday's close) and the review's clock says it is the 10th. Without that blocker the score would have been 87. `--now` can replay the Advisor and the Critic's arithmetic, never an approval, while the Critic's quote is live.

To get an approved decision, the data must be refreshed (`./test_recommendations.sh ... --fetch`, or `scripts/fetch_bars.py` / `scripts/sync_day_aggs.py`: **paid, and they write under `data/`, `data/historical/` included**, so Jacob's yes) and the review must be made in a regular session (next: Monday 2026-11-16, 09:30 to 16:00 Eastern; `IGNORE_MARKET_HOURS` is set in `.env`, so a closed market would not deny, but the stored bars and the day the record is for would still have to match). Then steps 3, 4, 6, 7 and 8 run as written.

### One bug found and fixed on the way (the exit code of a Critic run)

Every `desk --critic` run without `--trade` exited 1 and printed "(stopped here)", because the check wanted `status == "recommended"` and a Critic run ends `decided` or `reviewed`. Fixed at Jacob's request: `desk.reached_its_end(state)` is now the one rule for the PATH line and the exit code (details under "Found in phases 1 and 2" above). 11 tests added or rewritten; six fail with the old rule.

### Cost of this session's live calls

Two `./test_recommendations.sh PG` runs (the Advisor's plan call each, the thesis call and GLM 5.3's one call in the second, Finnhub for the Critic's metrics and quotes), and three Trader model runs. All in cents.

## First live run, second part: after the data refresh (2026-11-14)

Jacob said yes to the paid refresh. `./test_recommendations.sh PG --yes ...` (the script's default is to fetch) refreshed PG's news and the daily bars of PG, SPY, the sector ETFs and the holdings (Massive REST); press releases and filings had nothing new. Bars now run to 2026-11-13. The review is made on a Saturday, so it uses `--today 2026-11-13` (the last trading day: a closed market accepts a record and a quote of that day, and `IGNORE_MARKET_HOURS` is set).

| step | result |
|---|---|
| 3 backtest, no Trader model | **works end to end.** PG recommended (score 76 to 81), the Critic approved it (80/100 in the run that traded), decision `yes (AUTO)`, the Trader ran; the order is `ACTIVE`, "waiting for the next session": no bar from Monday is stored yet, as it should be on a Saturday |
| 4 paper, no model | **works.** The Trader asked Finnhub for the quote (169.80, equal to the Friday close). On Jacob's sample portfolio ($51.23) the purchase was `rejected insufficient_cash` and the report named `--raise-funds`; with `--cash 10000` the Critic sized it at 5 of 10 shares and the Trader filled 5 PG at 169.80 = $849.00, cash 10,000.00 to 9,151.00, `placed by code` |
| 5 paper, with the model | **works.** The same, `placed by model`, `summary (by model)`, no error. Together with the harness runs of the first part: four real-model runs, every tool call `ok` |
| 6 funding | **works, both answers.** The real Account Manager (Tree of Thought, 4 calls, 2 s) planned: sell 15 MSFT at 523.88 to buy 46 PG at 169.80 (score 92.3). Replayed in paper mode with the real model: answer NO, the purchase `rejected insufficient_cash`, nothing sold ("you said no"); answer YES, the purchase `FILLED 46 at 169.80 = $7,810.80` and the sale `T1-F1 FILLED 15 MSFT at 523.6181 = $7,854.27` (5 bps under the quote), cash 51.23 to 94.70, MSFT 50 to 35, PG 0 to 46 |
| 7 saved account, booked | **works on live pieces, not through the desk.** The approved order, the real model, a real quote, `book=True` into a throwaway Postgres account: `FILLED 46 at 169.80 ... booked`; the account book held cash 2,189.20 and PG 46 at cost 169.80, 1 trade; the account was deleted and the samples are unchanged |
| 8 replay | **works on a real saved decision**: `ACTIVE`, waiting for Monday's bar. The outcome (stop, target, time stop, against SPY) can only be seen once a bar after the entry day is stored |

### What has not been run

- **The desk** (`python -m prometheus.langgraph.desk ... --critic --trade`, with `--raise-funds`, with `--account --book`). It has no `--today` or `--now` (the runner has both), so on a weekend its recommendation is dated Saturday and the Critic denies it as `record_expired`. Run it on a trading day (Monday 2026-11-16, 09:30 to 16:00 Eastern is the natural time), or give the desk `--today` and `--now` like the runner (not built; Jacob's call). Steps 6 and 7 above used the replay command and a harness for the same Trader calls.
- **A backtest that fills.** Needs a stored bar from the next session. After Monday's close: refresh the bars (`./test_recommendations.sh PG --yes ...`, or the day files below), then `replay --order FILE --mode backtest --no-model --fresh --cash 10000`.
- **The Ask-mode prompt** at a terminal (every run here was Auto, no prompt).
- **The AWS CLI sync of the day files: done (2026-11-14, after Jacob replaced the Massive S3 keys).** `scripts/sync_day_aggs.py --from 2025-09 --to 2026-11 --yes` first failed twice with `403 Forbidden` on the first listing (the keys in `.aws/credentials` dated 2025-04-04 were refused); with the new keys it synced all 15 months in 2 minutes and added 287 files. Checked through `BarStore` (dates and closes only): 304 day files, 2025-09-02 to 2026-11-13; the day files ALONE give 300 bars for PG, SPY, MSFT, TSLA, GM and XLP, newest 2026-11-13; no unreadable file; the only gaps are market holidays; and every close agrees with the one the REST fetch stored (0 differences). So the Advisor no longer depends on the per-ticker REST files for these symbols, and a candidate found by a phrase search has bars in the day files too.

### Things the live run showed (none of them a fault of the Trader)

- **The Critic's score moves between runs** (77, 80, 79, 81 for the same recommendation, because GLM's labels of the quotes differ), and the Auto mark is 80, so a run at 79 asks the user instead of trading. The runs above used a scratch profile (`auto_min_score` 70, in the scratchpad, never in `configs/`) with `--profile`. The checklist expected this.
- **One review's GLM call timed out** (240 s; the run took 320 s). The Critic then marked the evidence `unreviewed` and `requires_human`, so in Auto mode the decision stayed `pending_user` and the Trader did not run: the guards worked as designed. The rerun took 38 s.
- **A misleading note in the runner.** On a Saturday with `--today` the last trading day, the runner says "the run date 2026-11-13 is not the Eastern date of the review (2026-11-14): every recommendation is expired. This is a replay, not a trade", although the Critic approved and the Trader traded. The note is right for a review in a regular session and wrong for a closed market, where the Critic accepts the last trading day. Not changed.
- **The first attempt used `--fetch`, which the script rejected** ("unknown option"), so it exited before any call. Added: `./test_recommendations.sh ... --fetch` is accepted (typing it is the yes to the paid fetch, like `--yes`; with `--no-fetch` it is refused). Tests and `HowToRun.txt` updated.

### Cost

About 12 runs of the Advisor (two or three calls each) and the Critic (one GLM 5.3 call each, Finnhub for the numbers and quotes), 6 runs of the Trader's model, one Tree of Thought (4 calls), the refresh fetch (daily bars for about 16 symbols, 21 news articles), and a few Finnhub quotes. All in cents.

---

## 2026-11-14: the per-trade handshake (design choices entry 81)

Built from `design/trader_handshake_spec.md` (with its amendments of the same day) after the sizing move (entries 79 and 80). The spec text of the Trader is amended at the end of `design/trader_v1_final_spec.md`, which says what it supersedes.

**What changed.** The Trader holds no cash and no portfolio: `PaperBroker` has no `portfolio` (nor `apply_fill`); it is given the Account Manager and the work order. Every trade goes through the one funnel `PaperBroker._fill`: the price rules (`_simulate`) give the fill price, `account.approve_trade` (the Account Manager says for how many shares, or no, with a code), the fill for at most the shares approved, `work_order.verify_fill` (the single-trade output check), and `account.record_fill` (applies and, with `book`, books ONE fill in ONE Postgres transaction, then reads the cash back); `place` returns only when the Account Manager answered. The Account Manager's side is `agents/account_manager/handshake.py` (`Approval`, `Booking`, `decide_approval`, `unbooked_fills`) and, in `account_manager.py`, `trading_portfolio(portfolio, book)` (which now OPENS the run: it stores the Account Manager's working portfolio), `apply_earlier_fills`, `current_portfolio`, `approve_trade`, `preview_trade` and `record_fill`. `execute.py`: step 2 asks the Account Manager to apply the earlier unbooked fills and reports the portfolio it gives back; the `Notifier` closes the run and reports (it tells nothing); `_consent` and `_fund` ask `preview_trade` instead of working out the cash.

**Order of the work, and how to find your way.** (1) `affordable_shares` moved from `trader/simulator.py` to `common/portfolio.py` (its test moved to `tests/test_portfolio.py`). (2) `work_order.verify_fill` and `EXPIRY_PREFIX`, `verify_trades` rebuilt on the same per-trade function (the old tests of `verify_trades` pass unchanged). (3) `handshake.py` and the Account Manager's methods. (4) `broker.py` (the funnel, `halted`, `answers`, `notes`, `cash_now`, `preview`), `execute.py`, `tools/trader.py` and `prompts/trader.py`. (5) The tests: `tests/trader_helpers.make_broker` builds a real Account Manager (`acct_now(broker)` reads its funds); the older tests assert the new protocol; the new ones are `handshake_helpers.py`, `test_account_manager_handshake.py` and `test_trader_handshake.py` (real data, and the real Postgres on throwaway accounts).

**Deviations from the spec, with the reason.**
- **`approve_trade` takes `order`, `now` and `on` (keyword only), and `commission`.** The spec writes `approve_trade(trade, price)`; the Account Manager cannot say the decision still permits the trade (`work_order.verify_trade`) without the order and the moment, nor apply the T+1 lock without the trade date (a backtest's is its entry date, not `now`). `commission` is the Trader's setting (0 in this version).
- **Three more small doors than the spec names:** `apply_earlier_fills` (amendment 1: somebody must hand the Account Manager the fills that live only on `order.trades`), `current_portfolio` (the spec's "reads the final portfolio from the Account Manager"), and `preview_trade` (the spec keeps the funding pre-check and forbids the Trader to compute the funds: a question, not an approval, is how both hold). `trading_portfolio` gained `book` (default false, so a caller that forgets it books nothing).
- **"Approve, place, record" is "place (the rules), approve, fill, record".** The approval needs the price the fill will have, so the price rules run first and the fill takes effect only after the approval. Nothing is filled before the approval and nothing is placed after a booking that is not yet answered.
- **`Trade.booked` is set at the fill, not at the end of the run,** because each fill is booked as it happens; the end-of-run `Notifier` only counts them.
- **The per-trade output check rejects with `internal_error`** (an existing code: the Trader's own result disagreeing with the decisions is what it is for), not a new one. `account_refused` was already in the closed list.
- **A run that was stopped (an output check found something, a booking could not be confirmed) is not marked `traded`,** although its trades are final: somebody should look at the order and the account book first. The spec only says "stop placing further trades".
- **The funding pre-check counts each sale at the size `funding_legs` validated** (on the Account Manager's snapshot) rather than re-cutting it to the sellable shares; the Account Manager still approves each sale for what is really sellable at its placing.
- **`notify_trades` is no longer called by the Trader** and stays as it was (its tests are unchanged) for a caller that reports a list of fills to a portfolio it holds; the spec said it "may call `record_fill`", and it does not.

**Tests deleted, and why.** `a_portfolio_from_the_account_manager_that_is_not_the_one_the_fills_lead_to_is_an_error_on_the_order` (the Trader has no portfolio of its own to compare with the Account Manager's: what it proved no longer exists) and `an_account_manager_that_cannot_be_told_leaves_the_fills_on_the_order_and_a_message` (the Trader no longer makes a call at the end that can fail; replaced by `an_account_manager_that_raises_instead_of_answering_a_fill...` and `...instead_of_approving...`). Changed to the new protocol, keeping what each proved: the output-guard test (now a whole-run REPORT), the close that breaks, `make_broker`-based broker and tool tests (the funds are read with `acct_now(broker)`, and the cases that set `broker.portfolio` change the Account Manager's account instead), and the doubles `Crashing` and `NoProfile` (their `trading_portfolio` takes `book`). `FakeStore` and `BookStore` return the account's `status` from `get_user`, as the real store does.

**Run for real, free.** `python -m prometheus.agents.trader.replay --order tests/fixtures/session_real/bob_jones_oil.json --mode backtest --no-model --fresh --cash 15000 --now "2026-11-12 10:30"` fills XOM and EOG at the next open of the stored bars (8 shares each, 171.5157 and 157.3286) and prints the account before and after ($15,000.00 to $12,369.24 in memory, as the Account Manager reported it). Nothing paid was run (no model, no quote, no `--book` against a sample account).

**Added the same day: the handshake does not wait for a next trade** (Jacob: "The handshaking should not wait if there are no further trades to execute"). Nothing in the funnel waited, so the code change is one thing: step 2 of `execute.py` no longer asks the Account Manager for the portfolio a second time (`current_portfolio`) unless the order carries a filled, unbooked trade of an earlier run to apply, so a run with no ticket reads the book exactly twice (the start and the end). The rule is written in the docstrings of `handshake.py` and of the broker's funnel and tested in `tests/test_trader_handshake.py` ("the handshake never waits" section): the real Account Manager's calls and the real store's reads are recorded over real runs (N fills are N approvals and N bookings and nothing after the last answer; zero tickets, nothing permitted, everything final, every ticket expired, no price and a resting order are zero handshake calls), `time.sleep` is patched to fail, and the source of the funnel, the steps, the class and the handshake module is read for a wait (an import of `time` or `threading`, a name like `sleep` or `wait`, a `while`).

**Added 2026-11-15 (design choices entry 82): the booking is three reads around one locked commit, and nothing halts the run.** `AccountManager.record_fill` for a run that books SELECTs the account, commits the trade in ONE transaction (`AccountStore.book_trade`: `FOR UPDATE` on the account row and on every position row, a lock timeout, a re-read inside the transaction that rolls everything back on a difference), and SELECTs again on a new connection (`read_booked` and `list_portfolio`): the trade row is there and the cash and the shares are different from before. `PaperBroker.halted` is gone: a trade that fails (`account_refused`, `internal_error`) is shown to the user by the Account Manager (`screen`, `tell_user`, `show_portfolio`), never retried, and the next trade is placed. Tests: `tests/test_account_booking_real.py` (real Postgres, throwaway accounts) and the lost-connection test of `tests/test_trader_handshake.py`.
