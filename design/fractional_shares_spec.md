# Fractional shares: an even split is equal to the cent

Status: BUILT on 2026-11-15 (design choices entry 110), uncommitted, for Jacob's review. Written the same day by the design advisor for Jacob as the spec for the coding worker; the text below is that spec as it was handed over.
It follows Part G of `design/approved_or_denied_spec.md` (the even split in whole shares), which stays as the fallback when the switch of F9 is off or an account book has not had its schema change.
What the build found that this text did not foresee is in entry 110 and in section 11's last item.

## 1. What Jacob asked for

2026-11-15: *"try to split the money evenly as much as possible. ... Split evenly will try to split the holdings so the balance of each holding is as close as possible. It is possible to buy fractional shares. Example: $30000 split evenly is $10000 for stock x, $10000 for stock b, $10000 for stock c and the remainder is again divided equally as close as possible."*

Asked whether the desk should really hold fractional shares, he chose: whole shares now (Part G), **fractional shares next, as their own change**. This is that change.

**The result wanted:** "split $30,000 evenly over three stocks" buys $10,000.00 of each, at the prices the Account Manager sized at, by buying a fractional number of shares of each. A leftover cent is shared out again, one cent at a time.

## 2. The decisions

| # | Decision |
|---|---|
| F1 | **A share quantity is a number with at most 6 decimal places** (0.000001 share is the smallest step). |
| F2 | **Whole stays whole.** A quantity that is a whole number is a Python `int` exactly as today, in memory, in JSON, in every capture and in every printed line. Only a quantity with a fraction is a `float`. Every real capture in `tests/fixtures/` still loads and every whole-share run gives the output it gives today. |
| F3 | **Two things produce a fraction, and nothing else does:** (a) the Account Manager's EVEN split; (b) a DIRECT ORDER in which the user typed a decimal number of shares ("sell 0.222222 shares of DPZ"). |
| F4 | **Everything on the way from a decision to the account book carries a fraction**: the trade decision, the Trader's ticket, the handshake's approval, the fill, the working portfolio, the Postgres account book, the page. |
| F5 | **Round-robin stays whole shares** (one share of each in turn). The how-much answer in shares stays a whole number. RaiseFunds still sells whole shares only. The Advisor and the Critic are not touched (they know no size). |
| F6 | **Money is unchanged**: cents, rounded half up by `trade_amount`, which is still the one rounding everybody uses. |
| F7 | **No limit is added.** No minimum order, no size limit, no liquidity rule. The book's own rule that a trade moves at least one cent is the only floor. |
| F8 | **The Postgres book needs a schema change** (four `integer` columns become `numeric(18,6)`). Until a database has it, that account book cannot hold a fraction, the Account Manager says so and sizes an even split in whole shares (Part G). Nothing is ever rounded into an integer column. |
| F9 | **A switch**: `configs/account_manager_v1.json`, section `sizing`, `fractional_even_split` (true). False gives Part G's whole-share split. |

Why F2 and not `Decimal` everywhere: prices and cash are floats in memory, `Decimal * float` raises, and thousands of lines and every capture hold `int` shares. Keeping whole numbers as `int` limits the change to the code that can meet a fraction.

## 3. One module for share quantities: `prometheus/common/shares.py`

New, plain Python, no model, no network. Everything that reads, adds, compares or prints a share quantity uses it; nothing else rounds a share quantity.

| Name | What it does |
|---|---|
| `PLACES = 6`, `STEP = Decimal("0.000001")` | The scale. A test reads `sql/prometheus_accounts.sql` and fails if the columns' scale is not `PLACES`. |
| `quantity(x) -> int \| float` | The one door. Accepts an `int`, a `float` or a `Decimal` (never a `bool`, never a string) that is finite, zero or more, and has at most `PLACES` decimals (worked out from `Decimal(str(x))`, never by rounding: 0.1234567 is refused, not rounded). Returns an `int` when the value is whole (81.0 is 81), else the `float` of that decimal. Raises `ValueError` with the reason otherwise. |
| `as_decimal(x) -> Decimal` | `x` at scale 6, for the book and for exact arithmetic. |
| `plus(a, b)`, `minus(a, b)` | Exact (in `Decimal`), then `quantity`. 81.004455 minus 81 is exactly 0.004455; a holding sold in full is exactly 0. `minus` raises below zero. |
| `whole(x) -> bool`, `floor(x) -> int` | Is it a whole number; the whole shares in it. |
| `buys(dollars, price) -> int \| float` | The MOST shares (at `STEP`) whose cost, `trade_amount(shares, price)`, is not above `dollars`. Closed form (divide, round down to the step, then try one step more): never a loop of steps. 0 when nothing fits. |
| `text(x) -> str` | "81", "1,234", "81.004455", "0.5": thousands separators, the decimals as they are, no trailing zeros. `signed(x)` is the same with a sign ("+0.5", "-81"). `noun(x)` is "share" for exactly 1, else "shares". |
| `Shares` | The pydantic type for a field: `int \| float` checked by `quantity`, with `PositiveShares` (above zero) where a field needs it. Checked on pydantic 2.14.1: with an after-validator 23 stays `23`, 23.0 becomes `23`, `Decimal("15.000000")` becomes `15`, 81.004455 stays a float, and a float that is arithmetic noise (0.1 + 0.2) is refused, which is wanted: quantities are added with `plus` and `minus`, never with `+` and `-`. But pydantic turns `True` into 1 and `"2"` into 2 BEFORE an after-validator runs, so run `quantity` before pydantic's own conversion (a before-validator), and for each field keep what it accepts and refuses today for a whole number. |

`prometheus/common/portfolio.trade_amount(shares, price)` multiplies `Decimal(str(price))` by `Decimal(str(shares))` (today it multiplies by the raw value, which raises for a float). Its result for a whole number is what it is today.

## 4. The even split with fractions (`sizing._split_evenly`)

`size_purchases` gets one new argument, `fractional: bool = False`. With False the even split is exactly Part G. With True:

1. **The pot** is as today (the budget when one was given, else the cash, never more than the cash), in cents.
2. **Who is in the split:** every purchase with a usable price. Nobody is dropped for "cannot buy one share" (that rule belongs to whole shares). A purchase is `no_room` only when its amount buys nothing at all (`shares.buys` returns 0).
3. **The equal amount:** the pot's cents divided by the number of purchases; the cents left over by that division go one cent each to the first purchases in rank order. That is Jacob's "the remainder is again divided equally as close as possible". Example: $30,000.02 over three is $10,000.01, $10,000.01, $10,000.00.
4. **A smaller ask holds it down, as today.** When a purchase was asked for fewer shares than its amount buys (`wanted`, whole shares), the equal amount of EVERY purchase is that purchase's amount (`binding: "even_split"`), and what is left is not spent. Unchanged rule.
5. **Each purchase takes `shares.buys(its amount, its price)`.** For a share price under $10,000 one step costs under a cent, so the cost is its amount to the cent.
6. **What is still left** (a few cents at most, when a step costs more than a cent) goes to the purchase with the smallest amount so far, as the most it buys (`shares.buys`), then to the next smallest, once round. Not when point 4 applied.
7. `Sized.shares` is `Shares`. `target_dollars` is the equal amount (the larger one when a cent was shared out) and `of` how many shared it. The notes say the split is in fractional shares and what is left.

The total never passes the pot. Property tests on real prices: every amount is within one cent of the others when every price is under $10,000; the total is not above the pot; what is left is less than one step of every purchase.

**Who says `fractional`:** `decisions.decide_trades` and `AccountManager._size_direct` pass `fractional = config.sizing.fractional_even_split and the account book can hold a fraction` (section 7: no book, or a migrated book). When the config says yes and the book cannot, the Account Manager tells the user in one line why the split is in whole shares.

Round-robin (`_round_robin`) is not changed at all.

In `decisions.py`: a purchase is `no_funds` when it sized to nothing (`shares <= 0`), not when it sized to less than one share.

## 5. A direct order with a decimal number of shares

Without this a holding smaller than one share could never be sold, and the fraction left after "sell 81 shares" of 81.004455 could never be sold either.

- `intent.py`: the number of shares right before the stock may have a decimal part of 1 to 6 digits ("0.5 shares of MSFT", "81.004455 shares"). More than 6 decimals is refused with the reason. The same change in `HOW_MUCH` (so "0.5 shares" counts as having said how much). `DirectOrder.shares` is `PositiveShares`, at most `MAX_DIRECT_SHARES`. `parse_intent` checks the model's number against the user's own words exactly as today (same value, same place). `describe` prints with `shares.text`.
- `prompts/account_manager_request.py`: say a number of shares may be a decimal, with one example. The model still only fills the form; code checks every field.
- `work_order.verify_direct_orders`: a quantity above zero, at most `MAX_DIRECT_SHARES`.
- `direct.py`: prices and decides it as today. "Placed as typed or not at all" is unchanged: never cut. A sale needs the account to hold at least the shares named (`shares.minus` does not go below zero); a sale of exactly the holding closes the position.
- The real test that says "Buy 50.5 shares of MSFT" is refused is amended on purpose: it is now an order for 50.5 shares.
- Not changed: a dollar amount is still not a number of shares; a word that asks for a recommendation still makes it a request to recommend; a sale with no number is still not placed.

Capture the model's real replies for three or four such sentences with ONE live run of `understand_request` (the standing yes of entry 84 covers it: no fetch, no account), saved beside `tests/fixtures/session_real/direct_order_forms.json`, and test against them.

## 6. The path to the book

Every place below must accept a fractional quantity, and every upper-bound guard must still fire for one. The survey of 2026-11-15 found three guards that go SILENT for a value that is not an `int`; they are marked. File and symbol names are the contract; line numbers may have moved.

### 6.1 `prometheus/common/portfolio.py`
- `Position.shares` is `PositiveShares`; `recent_shares` is `PositiveShares | None`.
- `sellable_shares`: `shares.minus`, never below 0.
- `affordable_shares(cash, price, shares, commission)`: for a WHOLE `shares` exactly today's answer (the most whole shares). For a fractional `shares`: `min(shares, shares.buys(cash - commission, price))`. So a whole order is cut to whole shares and a fractional order to a fractional quantity. No step loop.
- `_sell_from`, `_buy_into`, `apply_fill`, `after_purchase`: `quantity` instead of `isinstance(x, int)`; arithmetic through `plus` and `minus`; a position is removed when exactly 0 is left. The cost basis is the same weighted average to 4 places.

### 6.2 `prometheus/common/work_order.py`
- `verify_trade`: the decision's shares are a quantity above zero. **Silent guard 1:** the check that a direct decision does not exceed what the user typed runs only for an `int` today; it must run for every quantity.
- `verify_fill` (`_trade_reasons`): the fill's shares are a quantity above zero. **Silent guard 2:** "filled more than the decision gave" runs only when both are `int`; it must run for every quantity.
- The reasons say "a number of shares above zero" where they say "a whole number of shares".

### 6.3 The Trader (`prometheus/agents/trader/`)
- `tickets.py`: **Silent guard 3:** `_whole` turns anything that is not an `int` into 0 approved shares; it becomes "a quantity above zero, else 0". A sale is still whole-or-not-at-all against the shares sellable today ("whole" there means "the full quantity approved"; reword it). `funding_legs` stays whole shares (RaiseFunds).
- `record.py`: `Fill.shares`, `Trade.shares`, `Trade.shares_approved` are `Shares`.
- `broker.py`: `_fill` takes `min(wanted, approval.shares)`; `Fill.amount` is `trade_amount`.
- `simulator.py`, `execute.py`, `report.py`: the types, `shares.text` wherever a quantity is printed, and a holding's change is compared as quantities (a float `!=` must not report a change that is float noise).
- The Trader's model still cannot name a number of shares. Its tools return the quantity as a JSON number.

### 6.4 The handshake (`agents/account_manager/handshake.py`, `account_manager.py`)
- `Approval.shares` is `Shares`. `decide_approval`: "at least one share" becomes "above zero"; a purchase is approved for `affordable_shares(...)` (6.1); a sale for `min(wanted, sellable)`; a direct order is still never cut.
- `check_booked`: compares the booked row's shares with the fill's as quantities (the book returns a `Decimal`, the fill holds a `float`: compared raw, every fractional booking would look unconfirmed).
- `account_manager.py`: `_check_purchase`, `record_fill`, `_book_fill` (the shares go to the book through `as_decimal`), `notify_trades`, `show_portfolio` and `holding_lines` (print with `shares.text`, the column wide enough).

### 6.5 The words and the reports
Every printed quantity goes through `shares.text` (and `shares.noun` where "share(s)" is chosen): `direct.py`, `decisions.py`, `neutral.py`, `clarify.py` (its `:g` would cut 81.004455 to 81.0045), `session.py`, `langgraph/desk.py`, `critic/report.py`, the Trader's report, `application/web/views.py`.
Two places CRASH on a fraction today and must be fixed first: the `f"{delta:+d} {symbol}"` in `session.py` and in `web/views.py` (use `shares.signed`).
`session.py`'s sentence "in whole shares" is true only for a whole-share split: say which it was.

### 6.6 RaiseFunds (`reasoning/raise_funds.py`, `sizing.funds_check`)
It plans WHOLE shares, as today. What changes is only that a holding it reads may have a fraction:
- `Holding.shares` and `held` are `Shares`; `sellable` stays a whole number: the whole shares of (the T+1 sellable quantity, and with `never_liquidate` the holding less one share), never below 0. A holding of one share or less is not sold from.
- `Problem.shares` (the purchase it raises funds for) stays a whole number: the desk passes the purchase's quantity rounded UP to the next whole share, so the plan raises at least enough.
- `_clean`, the hard checks and the prompts keep "a whole number of at least 1" for what the model may sell.
- `closes_position` compares quantities.

### 6.7 The how-much questions (`clarify.py`)
Unchanged: an answer in shares is a whole number, `wanted` is whole shares. An answer that says "split it evenly" gives `balance: "even"`, which is now a fractional split through section 4.

## 7. The account book (Postgres)

### 7.1 The schema (`sql/prometheus_accounts.sql`)
Four columns become `numeric(18,6)`: `positions.shares`, `positions.recent_shares`, `trades.shares`, `recommendation_decisions.shares`. Their CHECKs stay as they are.
- In the `CREATE TABLE` statements (a new database).
- AND as idempotent `ALTER TABLE ... ALTER COLUMN ... TYPE numeric(18,6)` statements beside the file's other migrations (an existing database: `CREATE TABLE IF NOT EXISTS` changes nothing there). Running them twice does nothing.
- No trigger, function or view holds a share count (the survey read them), and `sql/seed_test_users.sql`'s whole literals still load.

`bash scripts/setup_postgres.sh` applies the file to the running container without dropping a table or an account. **Jacob agreed (2026-11-15) that the coding worker runs it, ONCE, at step 3 of section 10**, after `accounts.py` reads decimals and its tests pass against the file. Run it without `--reseed`. Before it, note the three sample accounts' positions and cash; after it, check they are the same, that the four columns are `numeric`, and that `tests/test_account_book.py` still passes. If the script fails, stop and report: do not mend the database by hand.

### 7.2 `prometheus/persistence/accounts.py`
- `AccountStore.holds_fractions() -> bool`: reads the type of `trading.positions.shares` from `information_schema.columns` once and remembers it. True only when it is `numeric`.
- `book_trade`, `record_trade`: the shares come in as a quantity and are held as `as_decimal` from the door on (the code multiplies them by a `Decimal` price and adds them to a `Decimal` holding). Above zero. **A quantity with a fraction on a book that does not hold fractions is refused before anything is written**, with a message that names the fix (`bash scripts/setup_postgres.sh`). The rule "the trade must move at least one cent" stays.
- `_buy_position`, `_sell_position`: `Decimal` arithmetic; the position row is deleted when exactly 0 is left; the T+1 lot (`recent_shares`) may be fractional.
- The read-back after the commit compares at scale 6.
- `Booked.shares`, `held_before`, `held_after`, and the positions `list_portfolio` returns, are quantities (`quantity(Decimal)`): an `int` when whole, as today.
- `_decision_row`: a quantity, zero or more.
- The ledger memo prints with `shares.text`.

### 7.3 Without a book
A run on a JSON portfolio file (no Postgres) can hold fractions with no migration.

## 8. The page (`ui/`) and its bridge

Targeted edits only. Do not rewrite a stylesheet or a component.
- `ui/src/format.ts`: a `shares(x)` formatter with up to 6 decimals and no trailing zeros (today `count` uses the default 3 decimals and would silently show 81.004). Use it wherever a share quantity is shown: the portfolio table, the question card, the recommendation panel, the activity lines (`state/selectors.ts` prints the raw number today).
- `state/selectors.ts` `changeBadge`: "NEW" and "SOLD" compare quantities rounded to 6 places, not raw floats.
- The API types are already `number`.
- One dev fixture from the real fractional session of section 9 (`/?fixture=...`), so the fractional rows can be looked at for free, and one front-end test of the formatter.
- `application/web/views.py`: section 6.5. No rule is added to the page.

## 9. Tests

Real data only (the stored bars, the real captures, live runs under the standing yes). No stand-in agent or model in a new test. No test file is deleted. Do not add a test for each boundary value.

- **`shares.py`**: one compact file: the door (what it takes and refuses), exact `minus` to zero, `buys` against `trade_amount` on real prices, `text`.
- **The even split:** the real oil session decided again with fractions is three purchases of $5,000.00 each at the session's own prices (EOG, CVX, CNQ), nothing rejected; the three property tests of section 4; the switch off gives exactly Part G; a smaller ask still holds the split down.
- **Whole stays whole:** every existing capture decides, trades and prints as it does today. The existing suite is that test: an existing test may be amended only where it pins "a fraction is refused" (below) or the wording of a message.
- **The path:** one real fractional purchase from decision to fill to the working portfolio; a purchase cut by the cash to a fractional quantity; the three silent guards each fire for a fraction; a fractional holding sold in full closes the position; T+1 holds a fractional lot.
- **The book** (throwaway accounts, never a sample; skipped with the reason when Postgres is not reachable OR the book does not hold fractions): a fractional buy is booked and read back exactly; part is sold; the rest is sold and the position row is gone; on a book that does not hold fractions the fractional trade is refused and nothing is written (run this one against whatever the database is: it is one of the two).
- **Direct orders:** the captured real replies of section 5; "sell 0.222222 shares" of a real holding; more than 6 decimals refused.
- **The schema:** the four columns are `numeric(18,6)` in the file, in both the `CREATE TABLE` and the `ALTER`, and `shares.PLACES` is 6.
- **Amended on purpose** (they encode "a fraction is refused"; each gets a line saying why, and keeps its other cases): `test_portfolio.py` (1.5), `test_work_order_critic.py` and `test_work_order_trader.py` (2.5), `test_trader_tickets.py` (2.5), `test_trader_record.py` (1.5), `test_trader_handshake.py` and `test_account_manager_handshake.py` (2.5), `test_direct_order.py` ("50.5 shares"), `test_account_manager_sizing.py` (`isinstance(s.shares, int)` holds for round-robin and the whole split only). **Not amended**, because the rule stays: RaiseFunds' whole-number tests (`test_raise_funds.py`, `test_green_first.py`, `test_two_purchases.py`, `test_account_manager.py`), `clarify`'s whole shares.
- **One live session** in Auto mode on a throwaway account, no fetch: a request that says "split evenly" with a budget, through to the fills. Save it as a real capture in `tests/fixtures/session_real/`, pin it in a test, delete the account. If the book does not hold fractions it will come out in whole shares: say so in the report and do not fake the other.
- The whole Python suite and the front-end tests are green apart from the known failure in `docs/SESSION_NOTES.md`.

## 10. Order of work

Each step leaves the suite green before the next begins.

1. `common/shares.py`, `trade_amount`, `portfolio.py`. Nothing produces a fraction yet.
2. `work_order.py`, the Trader, the handshake, the Account Manager's glue. Still nothing produces one; the path is tested by handing it a fractional decision made from a real one.
3. The book: the SQL file and `accounts.py`, with `holds_fractions`.
4. The even split (`sizing.py`, `decisions.py`, the config switch).
5. Direct orders with a decimal (`intent.py`, `direct.py`, the prompt, the live capture of the forms).
6. The words and the reports, then the page.
7. RaiseFunds' reading of a fractional holding.
8. Documents: a new entry at the end of `docs/design_choices.md` (problem, decision, alternatives, consequences; cite the entries it changes: 47, 78, 80, 81, 85, 87, 91, and Part G's own), `AGENTS.md`, `README.md`, `HowToRun.txt`, `docs/SESSION_NOTES.md`, a dated amendment at the end of `design/trader_v1_final_spec.md`, `design/trader_handshake_spec.md` and `design/sizing_in_account_manager_spec.md`, and the module text of every file changed. Old entries are not rewritten.
9. The whole suite, the front-end tests and build, the live session.

## 11. Open, for Jacob

1. ~~Who applies the database change~~ Settled by Jacob the same day: the worker runs `bash scripts/setup_postgres.sh` during the build (section 7.1). It keeps every table and account and also applies the pending column drop of entry 104. Any OTHER Postgres (another machine) sizes an even split in whole shares, and says why, until the script is run there.
2. **The amounts are equal at the price the Account Manager sized at.** The Trader fills at the quote of that moment (paper) or the next open (backtest), so the filled amounts can differ by the price's move. An order in DOLLARS that the Trader turns into shares at the fill would remove that; it changes what a decision permits (dollars, not shares) and is not in this change.
3. **A dollar amount for ONE stock** ("Buy $5,000 of MSFT") still buys whole shares. It could buy exactly $5,000.
4. **Round-robin keeps whole shares**, so it can still leave less than one share's price unspent.
5. **The how-much answer in shares stays a whole number.** A fraction of a share is typed in the request itself.
6. **RaiseFunds sells whole shares only** and still keeps one share of every holding.
7. **Six decimal places.**
**Items 2, 3 and 8 were settled by Jacob the same day** (`design/dollar_orders_spec.md`): no slippage, a run never spends more than its money, an order in dollars is built, and a direct order with money buys whole shares and a fraction with what is left.

8. **Found by the build (2026-11-15): the simulated broker's slippage is on top of the equal amounts.** The real Auto session "Split $30,000 evenly among the 3" was sized at $10,000.00 each and filled at $10,005.00 each (5 basis points), so $30,015.00 was spent of a $30,000 budget; the three fills are equal to each other. When the pot is ALL the cash, the last purchase is cut by the handshake to what the cash still pays for (the real oil session decided again: $5,002.50, $5,002.50, $4,995.00). Both follow from item 2 and were true of whole shares too. An order in dollars would end both.
