# A run never spends more than its money: orders in dollars, no slippage, and a share count is a number

Status: spec for the coding worker. Written 2026-11-15 by the design advisor for Jacob. It follows `design/fractional_shares_spec.md` (built the same day, design choices entry 110) and changes what a trade decision permits.

## 1. What Jacob said

On the real session "Split $30,000 evenly among the 3" that filled $10,005.00 each, $30,015.00 in all (the simulated broker added 5 basis points to every price):

> *"Slippage is not allowed. This is real money. It is impossible to spend more than the amount in the portfolio and budget. So, $30000 available and spending $30015 is impossible. Fix this. It is OK to have unused money left over but never is allowed to spend more than is available."*

On an order in dollars that the Trader turns into shares at the fill: *"Yes, I want #2 built."*

On a dollar amount for a stock the user named: *"When a direct order is given, buy as many whole shares as possible and a fractional share for the leftover funds available."*

On what a share count is: *"True as a share count is now refused and is an error. Getting any numeric string like "2" is an error also. All the share counts should be decimal and not string when arithmetic is used with counts and prices."*

## 2. The rules

| # | Rule |
|---|---|
| M1 | **A run never spends more than its money.** The money of a run (its POT) is the budget the user gave, else the cash, and never more than the cash. The purchases of a run never cost more than the pot in all, whatever price they fill at. Money left over is fine. This is checked where the money is: at the handshake, by the Account Manager, and again by a program on the finished work order. |
| M2 | **An order in dollars.** A purchase the Account Manager sized from MONEY is permitted in dollars: "at most $X of S". The Account Manager turns the dollars into shares at the handshake, at the price the trade fills at. The fill never costs more than the dollars. |
| M3 | **A direct order with money** (a stock the user named, with dollars or with no number) buys as many whole shares as the money pays for and a fractional share with what is left. |
| M4 | **No slippage.** The simulated broker fills at the price itself: `costs.slippage_bps` is 0 in `configs/trader_v1.json`. |
| M5 | **A share count is a number.** A bool or a string is an error at every door, also a numeric string such as "2". Money worked out from a share count and a price is decimal arithmetic through `trade_amount`, never a float product rounded. |

No limit is added that is not the user's own money: no size limit, no risk rule, no minimum order.

## 3. M1: the pot of a run

### 3.1 Where the pot is written
`sizing.size_purchases` already works the pot out (the budget when one was given, else the cash, never more than the cash). It now also reports it: every `Sized` carries `pot` (dollars, to the cent), and `decisions._size` and `AccountManager._size_direct` copy it to `decision["size"]["pot"]`. Every purchase sized in one call carries the same pot.

A purchase that was NOT sized from money has no pot: a direct order in which the user typed the shares ("Buy 50 shares of MSFT"), and the purchase of a RaiseFunds plan. Its limit is the cash, as today.

### 3.2 The handshake (`AccountManager.approve_trade`, `preview_trade`, `handshake.decide_approval`)
For a purchase whose decision carries a pot, the Account Manager works out what the run may still spend:

    pot left = pot - the amounts of the purchases of this work order that carry the same pot and are already filled

worked out from the fills on the work order and the fills recorded in this run (so a run that is resumed, `apply_earlier_fills`, counts what was filled before), never from memory alone.

`decide_approval` gets it as a new argument (`spendable`, None when there is no pot) and approves a purchase for what `min(cash now + expected proceeds, spendable)` pays for at the price it would fill at. The cut is today's (`affordable_shares`): a whole order is cut to whole shares, a fractional one to a fraction; nothing left to spend is `insufficient_cash` with a reason that says it is the money of the request that is used up, not the cash, when that is the case. A direct order with typed shares is still never cut.

### 3.3 The program check (`work_order.verify_trades`)
For the purchases of a work order whose decisions carry a pot: all carry the same pot (two different ones is a defect), and the sum of their fills' amounts is not above it. A work order that breaks this fails `verify_trades` with the two numbers in the reason.

## 4. M2: an order in dollars

### 4.1 The decision
A trade decision may carry `dollars` (a number to the cent, above zero): the most this purchase may cost. `shares`, `est_cost` and `price` stay: they are the ESTIMATE at the price the Account Manager sized at. A decision without `dollars` is an order in shares, exactly as today.

Which purchases are orders in dollars (only when fractions are on for the run: `AccountManager.fractional_split`, entry 110; when they are off or the account book cannot hold a fraction, nothing is an order in dollars and M1 alone holds the pot):

| Purchase | `dollars` |
|---|---|
| Every purchase of an even split (approved or neutral) | its equal amount |
| Every purchase of a direct order sized from money (section 5) | its amount (section 5) |
| A purchase sized round-robin from recommendations (the default) | none: an order in whole shares, held by the pot (M1) |
| A purchase held to the shares the user asked for (`binding: "asked"`, an answer in shares) | none: an order in shares, held by the pot |
| A direct order with typed shares; a sale; a funding sale | none |

The sum of the `dollars` of a run's purchases is never above the pot.

### 4.2 The handshake turns dollars into shares
For a purchase whose decision carries `dollars`, `decide_approval` approves

    shares.buys(min(dollars, spendable, cash now + expected proceeds), the fill price)

so the approved shares can be FEWER than the estimate (the price is higher) or MORE (it is lower). The Trader's record of the trade carries the `dollars` (`Trade.dollars`, copied from the decision by `tickets.py`), and `PaperBroker._fill` fills the shares approved, not the smaller of the estimate and the approval, for such a trade.

The Trader still decides no number: it asks, the Account Manager answers with the shares, the Trader fills at most those. Its model's tools are unchanged and still cannot name a symbol, a side, a number of shares or a number of dollars.

### 4.3 The checks
- `work_order.verify_trade`: a decision with `dollars` needs a number above zero to the cent, and a direct one not above what the user typed or the pot.
- `work_order.verify_fill` (`_trade_reasons`): for an order in dollars the fill's AMOUNT is not above the decision's `dollars` (to the cent), and the check "more shares than the decision gave" does not apply to it (its shares are an estimate). For an order in shares that check stays exactly as it is. One of the two always runs.
- `handshake.check_booked` and the book are unchanged: they see the shares that were filled.

### 4.4 What the user is shown
Wherever a decision or a proposal is printed or shown (the session's report, the Ask-mode card, the page's order column and recommendation panel, `web/views.py`): an order in dollars says so, with the dollars first and the shares as an estimate ("buy $10,000.00 of EOG, about 63.163213 shares at $158.32"). The fill line shows the shares and the amount that were filled, as today. In `ui/`: targeted edits only.

## 5. M3: a direct order with money

This is `AccountManager._size_direct` (a stock the user named with dollars, or with no number: Auto spends the user's words, Ask asks how much first), when fractions are on.

1. **The money is spread as today**: round-robin in whole shares, one share of each stock in turn, in rank order (or the even split when the user's words ask for one, which is already equal dollars).
2. **New: what the round-robin leaves over buys a fraction of a share.** The leftover (less than one more share of any stock still open) goes to the stock whose turn is next: among the purchases not closed by their own ask, the one with the fewest shares, a tie in rank order. It takes `shares.buys(its cost so far + the leftover, its price)`.
3. **Every such purchase is an order in dollars**: its `dollars` is what the sizing gave it (its cost at the sizing price; for the one of point 2, plus the leftover). The dollars add up to the pot. At the fill each buys its dollars' worth: whole shares and a fraction.

For ONE stock that is Jacob's sentence as it stands: "Buy $5,000 of MSFT" is as many whole shares as $5,000 pays for and a fraction of a share with the rest: $5,000.00.

The how-much answers of Ask mode, for a direct order with fractions on:
- **Dollars or a percentage in all** is the pot (as today), spread as above.
- **Dollars or a percentage for EACH stock** gives each stock that amount in full: `wanted` is `shares.buys(the amount, its price)` (it may have a fraction), and the purchase is an order in dollars for that amount. `sizing.size_purchases` therefore accepts a quantity in `wanted` (a whole number or one with a fraction), and the round-robin's last step for such a purchase is what is left of its ask when that is less than one share. Never a loop of 0.000001 steps.
- **A number of shares** stays a whole number, and such a purchase is an order in shares.

Not changed: a direct order with typed shares is placed as typed or not at all; a sale; the recommendation path (its round-robin stays whole shares and leaves its leftover as cash: Jacob's sentence is about a direct order).

## 6. M4: no slippage

`configs/trader_v1.json`: `costs.slippage_bps` is 0, and the file's note says why (Jacob's words). The simulator keeps its argument (its own tests pass a number of their own), and `Fill.slippage_bps` stays on the record, so the real captures recorded with 5 basis points still load: they are history and are not edited.

With no slippage a paper fill is at the quote. The price can still differ from the price the Account Manager sized at (a later quote; the next open in a backtest), which is what M1 and M2 are for.

`direct.py`'s note that "the paper broker adds a few basis points of slippage" goes.

## 7. M5: a share count is a number

- **Every door that takes a share count refuses a bool and a string**, with the reason. `shares.quantity` already does. Find the doors that do not go through it and make them: the survey of 2026-11-15 found `reasoning/raise_funds._clean` (it accepts a digit string from the model's reply: "2" becomes 2) and `AccountManager._check_purchase`; look for `.isdigit()`, `int(` and `float(` applied to a share value. A model's reply with `"shares": "2"` is an error the model is told about, as its other mistakes are.
- **Reading the user's own typed words is not a share count as a string**: `intent.py`'s phrases, `parse_purchase` (the command line's `MSFT:10`) and `argparse` turn TEXT the user typed into a number once, at the edge, and refuse what is not a number. They stay. After that edge nothing holds a count as a string.
- **Money from a count and a price is `trade_amount`** (decimal, half up to the cent). Replace the float products the survey found: `direct.py` (`round(shares * price, 2)`, two places), `reasoning/raise_funds.py` (`Problem.cost`, the proceeds and the gains), `trader/execute.py` (`funding_card`), `web/views.py` (a holding's value), and `sizing.equity_of` and `portfolio.market_value` where they feed a number the user is shown or a decision is made from. Where a result is compared with a pinned real value and differs by a cent, the test is amended with a line saying why.
- Counts are added and subtracted with `shares.plus` and `shares.minus` (entry 110); look for a `+` or `-` between two share counts that entry 110 missed.

## 8. Tests

Real data only (stored bars, real captures, the live runs under the standing yes of entry 84). No stand-in agent or model in a new test. Do not edit a real capture. No test for each boundary value.

- **The session that started this**, `tests/fixtures/session_real/auto_even_split_fractions.json`: decided and traded again under the new rules it fills $10,000.00 of each and $30,000.00 in all, not $30,015.00. Its pinned test is amended and says the recording was made with 5 basis points, before this entry. The file is not edited.
- **A price that moved, with real prices**: the same decisions traded in `backtest` mode fill at the next session's real open. Every fill of an even split is its dollars to the cent (each price is under $10,000) and the sum is not above the pot, with opens above and below the sizing price. A round-robin run in whole shares in the same mode never passes its pot; where the opens are higher the last purchases are cut, and the report says why.
- **The pot is enforced in three places**: the handshake refuses or cuts a purchase that would pass the pot while the cash would cover it; `verify_trades` fails a work order whose fills pass the pot; `verify_fill` fails a fill above its decision's dollars. Each with a real decision made to break the rule.
- **Direct orders**: "$5,000 of MSFT" (a real captured form) is whole shares and a fraction, $5,000.00; several stocks with a budget add up to the budget; "dollars each" gives each its amount; typed shares are as they were.
- **A share count is a number**: `True` and `"2"` are refused at each door named in section 7, with a real RaiseFunds reply made to say `"2"`.
- **No slippage**: the config says 0; a test that pinned a fill 5 basis points above the quote through the real config is amended with a line saying why.
- **Whole stays whole**: with fractions off nothing is an order in dollars and every existing whole-share run gives what it gives today, apart from the slippage.
- **One live session** in Auto mode on a throwaway account, no fetch, with the neutral setting on: "split evenly" with a budget smaller than the cash, and asking for MORE companies than are likely to be approved (for example the 5 most recommended oil companies), so that a neutral company fills the number. Through to the fills booked in Postgres. Save it as a real capture, pin it (each fill its dollars, the total not above the budget), delete the account and confirm it is gone.
- **The page's neutral row.** `ui/src/fixtures/pam.json` was deleted on 2026-11-15 at Jacob's word (the recording was too old), and with it two front-end tests that read it: "runReducer over a neutral pass: has neutral rows whose decision says where they came from" (`state/runReducer.test.ts`) and "a neutral pass says the Advisor found no strategy and that it counts as approved" (`components/panels.test.tsx`); a comment marks where each was. If the live session above has a neutral purchase, record it for the page from your own replay server (`ui/scripts/record_fixture.mjs`, a port that is not 8000) as a new fixture and put the two tests back on it. If it has none, say so in the report and leave the two comments; do not make a neutral row up.
- The whole Python suite and the front-end tests are green apart from the known failure in `docs/SESSION_NOTES.md`.

## 9. Order of work

Each step leaves the suite green before the next begins.

1. M4 (the config and the tests that pinned 5 basis points).
2. M5 (the doors and `trade_amount`).
3. M1 (the pot on the decision, the handshake, `verify_trades`).
4. M2 (the decision's `dollars`, the handshake, the Trader's record and fill, `verify_fill`, what the user is shown).
5. M3 (direct orders with money, the how-much answers).
6. Documents: a new entry at the end of `docs/design_choices.md` (problem: $30,015.00 spent of $30,000; the decision; the alternatives: keep the slippage and only hold the total; hold every purchase to its estimate in whole shares; consequences), dated amendments at the end of `design/trader_v1_final_spec.md`, `design/trader_handshake_spec.md` and `design/sizing_in_account_manager_spec.md`, `AGENTS.md`, `README.md`, `HowToRun.txt`, `docs/SESSION_NOTES.md`, and the module text of every file changed. Do not edit this spec or `design/fractional_shares_spec.md`.
7. The whole suite, the front-end tests and build, the live session.

## 10. Open, for Jacob

1. **The recommendation path's round-robin still buys whole shares** and leaves less than one share's price unspent. Jacob's sentence about the leftover was about a direct order.
2. **When a price is higher at the fill, a whole-share run is cut at its end**: the last purchases get fewer shares, in rank order, because the pot is used up. An order in dollars is cut evenly by its own dollars instead.
3. **SUPERSEDED on 2026-11-16.** Jacob: *"Need to always use decimal.Decimal type when dealing with money and share counts."* The reading below was wrong; `design/decimal_money_spec.md` replaces it and is built after this spec. As first written: **"Decimal" is read as a decimal NUMBER**: a count is a number (a whole number or one with at most 6 decimals), never a string or a bool, and every sum, difference and product with a price is done in exact decimal arithmetic. The counts are not held as Python `Decimal` objects in memory (entry 110, decision F2: prices and cash are floats there).
