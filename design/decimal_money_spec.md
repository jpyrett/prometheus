# Money and share counts are `decimal.Decimal`

Status: spec for the coding worker. Written 2026-11-16 by the design advisor for Jacob. To be built AFTER `design/dollar_orders_spec.md` (design choices entry 111) is finished and green: it converts that code too.

## 1. What Jacob said

2026-11-15: *"All the share counts should be decimal and not string when arithmetic is used with counts and prices."*

2026-11-16, after I had read that as "a decimal number, held as an int or a float": *"Need to always use decimal.Decimal type when dealing with money and share counts. See this example code how to work with Decimals in python:*

```python
from decimal import Decimal, ROUND_HALF_UP

price = Decimal("19.99")
tax_rate = Decimal("0.075")

# Exact multiplication
tax = (price * tax_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
total = price + tax  # Decimal('21.49')
```

*Change all the processing code from float to properly used Decimal types."*

This replaces decision F2 of `design/fractional_shares_spec.md` (a whole share count is an `int`, one with a fraction a `float`) and item 3 of section 10 of `design/dollar_orders_spec.md`.

## 2. The rules

| # | Rule |
|---|---|
| D1 | **In the account's code every dollar amount, every price a trade is sized or filled at, and every share count is a `decimal.Decimal`.** They are added, subtracted, multiplied and compared as `Decimal`. No `float` takes part in that arithmetic, anywhere. |
| D2 | **A `Decimal` is made from digits, never from a binary float**: `Decimal("19.99")`, `Decimal(3)`, or the digits a number was written with. `Decimal(19.99)` is never written. |
| D3 | **A result is rounded once, on purpose, by name**: money to the cent, half up (`quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)`), a price to four places, half up, a share count to six places, DOWN when it is bought with money (never more than the money pays for). The functions that do this exist and stay the only ones: `common/user_account.cents` and `price_scale`, `common/portfolio.trade_amount`, `common/shares.buys`. |
| D4 | **A number comes in through a door, once.** A bool or a string is an error at every door (a numeric string such as "2" too). A `float` that arrives from OUTSIDE (a JSON file or reply read by a library, a quote from an API client, a stored bar, the Advisor's plan) is turned into a `Decimal` at the door by the digits it is written with (`Decimal(str(x))`) and checked against its scale; after the door there is no float. |
| D5 | **In JSON a money amount, a price and a share count is a NUMBER, as today, never a string.** Every real capture still loads, and the page and the models still get numbers. |
| D6 | **The measurements stay `float`**: stored bars as market data, technical indicators, returns, volatilities, ratios, percentages, scores, and everything the Advisor and the Critic calculate to JUDGE a stock. They are statistics, not the account's money. A price crosses from there into the account's code once, through the door (D4). |

A consequence that is wanted: `Decimal * float` raises `TypeError`. A float that was missed stops a test instead of putting a wrong cent in the account.

## 3. The two sides

**The account's code (all `Decimal`, D1):**
- `prometheus/common/`: `portfolio.py`, `shares.py`, `user_account.py`, and in `work_order.py` everything about direct orders, trade decisions, trades and fills.
- `prometheus/persistence/accounts.py` (already `Decimal` inside; its edges stop converting to `float`).
- `prometheus/agents/account_manager/` (every module: sizing, decisions, direct, neutral, clarify, intent's budget and shares, handshake, missing, account_manager, config).
- `prometheus/agents/trader/` (tickets, broker, simulator, record, execute, report, replay, config) and `prometheus/tools/trader.py`.
- `prometheus/reasoning/raise_funds.py` and `prometheus/tools/trading_math.py`.
- The money and the shares in `prometheus/langgraph/desk.py` (the plan's budget, the funds node), `prometheus/application/session.py`, `recommendations.py` (its trading part) and `prometheus/application/web/` (views, accounts, desk_run, replay, runs, emitter).
- The facts about money that a prompt prints (`prometheus/prompts/account_manager*.py`, `trader.py`).

**The measurements (stay `float`, D6):** `prometheus/agents/advisor/`, `prometheus/agents/critic/`, `prometheus/agents/data_clerk/` (bars, technicals, fundamentals, the metrics catalogue), `prometheus/rag/`, `prometheus/memory/ledger.py`, the tools and prompts of those agents, and `labs/`. Not touched by this change.

**Where a price crosses** (each is a door, D4):
- the Data Clerk's quotes and bars asked for by the Account Manager and, through it, the Trader (`request_prices`, `request_quotes`, `request_bars`);
- the Critic's deal price and the Advisor's plan prices (entry zone, stop, target, limit) when `decisions.py` makes a purchase to size and `trader/tickets.py` makes a ticket;
- a candidate's or a review's price read off the work order by the Account Manager.

**Ratios that multiply a price or an amount** are `Decimal` too, read from their digits: the Trader's `orders.max_chase_pct` and `limit_band_pct`, `costs.slippage_bps` and `commission_per_order`, and a percentage the user typed ("20% of my account"). The config loaders give a `Decimal` for them.

**What stays a float inside the account's code, on purpose:** a result that is a measurement and is never added to money: a return (`return_pct`, the benchmark's, the excess), a share of the account shown as a percentage, a model's temperature, a timing. It is worked out at the very end from `Decimal` values (`float(gain / cost)`), and the code says so at that line.

## 4. The types and the doors

Build on what is there; do not write a second rounding of anything.

| Name | Where | What it is |
|---|---|---|
| `Money` | `common/user_account.py` (beside `cents`, `parse_money`) or a small `common/money.py` that re-exports them: your choice, one home | The pydantic type of a dollar amount: a `Decimal` to the cent. Its door takes a `Decimal` or an `int`, and a `float` by its written digits; refuses a bool, a string, a value that is not finite, and one with more than two decimals (it is not rounded at the door: a caller that has a longer number rounds it with `cents` first, on purpose). `PositiveMoney`, `NonNegativeMoney` where a field needs them. |
| `Price` | same home | A `Decimal` to four places, above zero. Its door takes the same kinds; a price from OUTSIDE with more than four decimals (a quote) is rounded to four places half up there, because the account book holds four (`numeric(14,4)`), and that is the one rounding of a price. |
| `Shares`, `PositiveShares` | `common/shares.py` | A `Decimal` with at most six places. `quantity(x)` returns a `Decimal`, always (81 is `Decimal("81")`, no longer an `int`; 81.004455 is `Decimal("81.004455")`). `plus`, `minus`, `difference`, `same`, `floor`, `ceil`, `buys`, `text`, `signed`, `noun` keep their meaning; `as_decimal` is the quantity at scale six for the book. `whole(x)` is "has no fraction". |
| `trade_amount(shares, price)` | `common/portfolio.py` | Returns a `Decimal` to the cent, half up. Still the one rounding of a price times shares. |
| `cents`, `price_scale`, `parse_money`, `parse_price` | `common/user_account.py` | Unchanged. |

**JSON (D5).** One encoder, used wherever the desk writes JSON that can hold a `Decimal` (a work order, a capture, an event to the page, the facts given to a model, a record read back by the page): a `Decimal` is written as a JSON number: a whole value as an integer (`81`, `10000`), any other by its own digits (`63.163213`, `4907.92`). With the standard library that is `float(d)` in a `default=` hook; it must first check that `Decimal(repr(float(d))) == d` and raise when it is not (more digits than a float holds), so a digit is never dropped in silence. The pydantic types serialise the same way in JSON mode. No new dependency.
Reading: the typed doors (the pydantic fields and the door functions) turn the float that `json` gives into a `Decimal`. Do NOT read a whole work order with `parse_float=Decimal`: the Advisor's and the Critic's numbers in it are measurements and stay floats (D6).

**Printing.** `f"{amount:,.2f}"` and `f"{price:,.4f}"` work on a `Decimal` as they are. Share counts print through `shares.text`.

**Config.** `configs/account_manager_v1.json` `sizing.float_guard` exists only to floor a float share count; with `Decimal` there is nothing to guard. Remove the key, its field and its uses, and say so in the file's note. The Trader's ratios are read as `Decimal` (section 3).

## 5. What changes, module by module

The list is the contract for WHERE; the worker reads each module for the lines.

- **`common/portfolio.py`**: `Portfolio.cash_avail` is `NonNegativeMoney`, `Position.cost_basis` is `Price`, `Position.shares` and `recent_shares` are quantities. The sample file's `"cash_avail": "51.23"` (a string, in `configs/portfolios/jacob_pyrett.json`) is the one string that is read on purpose: a portfolio FILE may write its cash as text; `Portfolio`'s own loader parses it with `parse_money`, and that is said at the field. `market_value`, `affordable_shares`, `apply_fill`, `after_purchase`, `_average_cost` are `Decimal` arithmetic; the cost basis is `price_scale` of the weighted average.
- **`persistence/accounts.py`**: what it reads from Postgres is already `Decimal`; `list_portfolio`, `account_balance`, `Booked` and the rows it returns stop turning it into `int` or `float`. What it is given is a `Decimal` already (no `Decimal(str(...))` of a float inside the book).
- **`agents/account_manager/sizing.py`**: `Purchase.price` is `Price | None`; `Sized.est_cost`, `target_dollars`, `dollars`, `pot` are `Money`; `shares`, `asked`, the caps are quantities; `size_purchases(purchases, cash, ..., budget, wanted)` takes `Decimal` cash, budget and wanted. The round robin and the two even splits are already cents in `Decimal`: the float conversions around them go. `_amount`, `_whole`, `_dollars`, `_pools` lose their float handling. `equity_of` and `funds_check` return `Decimal` money.
- **`decisions.py`, `direct.py`, `neutral.py`, `clarify.py`, `handshake.py`, `account_manager.py`**: every amount, price and count on a decision, a card, an approval, a booking and a plan. `clarify.plan_choice`: a percentage is `Decimal` from the user's digits; dollars / price is `shares.buys` or a `Decimal` floor. `intent.py`: `TradeIntent.budget` is `Money`, read from the digits the user typed (the checker compares it with their words as today); `DirectOrder.shares` is a quantity.
- **`common/work_order.py`**: the validators of direct orders, decisions, trades and fills accept what the doors accept and compare as `Decimal` (the pot of entry 111, the dollars of an order, the shares against the decision). The work order still holds plain dicts; a value in them is a `Decimal` after the code that wrote it, or a float after `json` read it, and every reader goes through a door.
- **`agents/trader/`**: `record.py` (`Fill.price`, `amount`, `commission`; `Trade.dollars`, `limit_price`, `ref_price`, `limit_low`, `limit_high`; `Snapshot.cash`; the outcome's prices and gains are money, its returns stay floats); `tickets.py` (the limit band is `Decimal` arithmetic); `simulator.py` (the fill rules: `price * (1 + slip)` with `Decimal`; `r4` is `price_scale`; `follow`'s gains are `trade_amount` differences, its returns floats at the end); `broker.py`, `execute.py`, `report.py`; `tools/trader.py` (what the model is shown goes through the encoder).
- **`reasoning/raise_funds.py`, `tools/trading_math.py`**: cash, prices, proceeds, costs, gains and the shortfall are `Decimal`; the shares a plan sells are whole quantities; `math.ceil(shortfall / price)` is a `Decimal` ceiling. The model's reply is checked as today, with a number that is not a string.
- **`langgraph/desk.py`, `application/session.py`, `recommendations.py`, `application/web/`**: they pass and print `Decimal`; the web bridge's JSON goes through the encoder; `views.py` computes a holding's value with `trade_amount`. The page (`ui/`) is NOT changed: it receives the same numbers and does no arithmetic with money that the account depends on.
- **Prompts**: the facts are numbers written by the encoder.

## 6. A test that reads the source

One new test reads the source of every module on the account's side (section 3) and fails when it finds, outside an allow-list that names each exception with its reason:
- a call of `float(`,
- an annotation `: float` or `-> float`,
- `round(` (money and prices are rounded by `cents`, `price_scale`, `trade_amount`; counts by `shares`),
- a float literal used in arithmetic.

The allow-list is short and explicit: the JSON encoder; the returns of section 3's last paragraph; a temperature; a timing. It is how "no float in the account's code" stays true after this change.

## 7. Tests

Real data only; no stand-in agent or model in a new test; do not edit a real capture; no test for each boundary value.

- **Expected values are `Decimal`.** An existing assertion that compares money, a price or a share count with a float literal (`== 4907.92`) becomes a `Decimal` made from the same digits (`== Decimal("4907.92")`); a whole number may stay an `int` (`Decimal("5000") == 5000` is true). Never `pytest.approx` for money or shares: the point of this change is that the cent is exact. This is most of the work by volume. It is a mechanical change of the expected TYPE, not of a value: a test whose expected VALUE has to change is a finding; say in the report which and why (a float rounding that was a cent off is the likely cause).
- **A capture's own numbers are floats after `json` reads them.** A test that compares a recorded value with a calculated one sends the recorded one through the door first.
- **The doors**: a bool and a string are refused by `Money`, `Price` and `Shares`; a float is taken by its written digits (0.1 + 0.2 is refused by `Money` as more than two decimals, not rounded); a quote with five decimals is rounded to four, half up.
- **Exactness, on real data**: decide and trade the real captures again and check the identities that floats only held to a tolerance: the cash after a run is the cash before less the sum of the fills, exactly; an even split's amounts add up to the pot exactly; `est_cost` is `trade_amount(shares, price)` exactly; the book's cash equals the working portfolio's cash exactly after every fill (`tests/test_trader_handshake.py`, throwaway accounts).
- **JSON**: a work order with `Decimal` values written and read back gives the same numbers; a capture file made before this change loads; a `Decimal` with more digits than a float holds is refused by the encoder, not shortened; what the page receives is numbers (the existing `tests/test_web_*.py` and the front-end tests still pass without a change to `ui/`).
- **`Decimal * float` is a test failure, not something to quiet**: do not wrap a value in `float(...)` or `Decimal(str(...))` at the place it fails; find where the float came in and convert it at its door.
- **One live session** in Auto mode on a throwaway account, no fetch (the standing yes of entry 84), at the end, to see the real agents' numbers pass every door: a request with a budget and "split evenly". Save it as a real capture, pin it, delete the account and confirm it is gone.
- The whole Python suite and the front-end tests are green apart from the known failure in `docs/SESSION_NOTES.md`.

## 8. Order of work

The change cannot be made one module at a time with the suite green in between unless each step ends at a clean edge, so work from the inside out and keep each step's edge explicit:

1. The types and the doors (`Money`, `Price`, `Shares` as `Decimal`, `trade_amount`, the JSON encoder), with their own tests. Nothing uses them yet.
2. `common/portfolio.py` and `persistence/accounts.py` and everything that reads a portfolio's cash, cost basis or shares for ARITHMETIC (the sizing, the handshake, RaiseFunds, the Trader's broker): this is the big step, because the portfolio is read everywhere. Do it in one piece.
3. The Account Manager's decisions, the work order's validators and the Trader's records, tickets and simulator.
4. RaiseFunds and `trading_math`.
5. The desk, the session, the web bridge and the prompts' facts.
6. The source-reading test of section 6, then remove every temporary conversion it finds.
7. Documents: a new entry at the end of `docs/design_choices.md` (problem: floats for money, a rounding per call site; decision; alternatives: keep floats and round carefully (entry 110's F2); `Decimal` everywhere including the measurements; strings in JSON; consequences), `AGENTS.md`, `README.md`, `HowToRun.txt`, `docs/SESSION_NOTES.md`, dated amendments at the end of `design/trader_v1_final_spec.md`, `design/trader_handshake_spec.md`, `design/sizing_in_account_manager_spec.md`, and the module text of every file changed. Do not edit this spec, `design/fractional_shares_spec.md` or `design/dollar_orders_spec.md`.
8. The whole suite, the front-end tests and build, the live session.

Before a script rewrites test expectations in a file, copy that file to the scratch directory: the working tree is uncommitted, and git cannot give an edited file back.

## 9. Open, for Jacob

1. **The measurements stay floats in THIS change** (section 3): moving averages, volatility, returns, ratios, scores, and the Advisor's and the Critic's own calculations. A price becomes a `Decimal` the moment the Account Manager or the Trader takes it. **Jacob, 2026-11-16: converting the rest of the sensitive calculations that use `float` is a TODO for later** (item 10 of "Open and next" in `design/week6_updated_design.md`, and the open list of `docs/SESSION_NOTES.md`). Do not start it in this change.
2. **A price is held to four decimal places**, the account book's scale; a quote with more is rounded half up when it comes in.
3. **JSON keeps numbers**, so captures, the page and the models are unchanged; a `Decimal` is written by its own digits.
