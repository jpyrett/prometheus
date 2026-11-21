# Raising funds for a direct order (spec)

Jacob, 2026-11-21: *"I did a direct order to buy 2 shares of ANT. There was not enough funds available and the RaiseFunds did not automatically work, even when I asked it in the prompt to raise funds ("Buy 2 shares of ANT and raise funds if needed"). ... The desk should have prompted me (asked) the User: Do you want to raise funds? and if yes then ran my existing Tree-of-Thought RaiseFunds code to make this happen as was in Ask mode. It would have sold (taken some profits) and then bought the two shares. Fix this scenario."*

What happened (account 6553, page, 2026-11-21T04:10Z): `direct._decide_one` found cash $60.14 against 2 x $88.91 = $177.82 and decided `no_system_denied` ("the order is placed as you typed it or not at all"). The desk never reaches RaiseFunds on this path: the funds node (`account_manager_funds`, entry 42) is built only with the CLI flag `--raise-funds` (`build_desk(with_funds=...)`), only after the Advisor/Critic, and never by the page or `session.py`. The words "raise funds if needed" are read by nothing.

Design choices entry 142 (written by the main session).

## Rules

| # | Rule |
|---|---|
| R1 | **When.** A DIRECT BUY with a fixed number of shares (typed shares, entry 87; also a dollars order whose dollars exceed the available cash) that the AVAILABLE cash (the book's door, entry 140: `cash_avail`) does not cover. The shortfall is `cost - cash_avail` at the decision's price (Decimal, `trade_amount`). A direct order is still placed as typed or not at all: RaiseFunds raises the money for THE ORDER AS TYPED, it never cuts the order. Nothing on the recommendation path changes (its purchases are sized to the cash). A waiting limit order (entry 140) is not funded by sales (out of scope: it is refused at placement as now, and the user is told). |
| R2 | **Permission in the words.** A closed pattern decided by code, `intent.RAISE_FUNDS_CUE`: "raise (the )?funds", "raise (the )?(cash|money)", "free up (cash|money|funds)", "sell (something|some|what(ever)? (you\|I) need|holdings|shares|stock)s? to (pay\|cover\|fund\|buy)", "take (some )?profits? to (pay\|buy\|fund)" (negation as entry 138's `NO_RECOMMEND` style: "don't raise funds", "do not sell anything" means NO). `TradeIntent.raise_funds: bool` (in the model's form too; code decides from the words, the model only has to agree). |
| R3 | **Ask mode, no permission in the words:** the Account Manager ASKS one yes/no question through the existing approver channel (a new card kind `raise_funds`): "You have $60.14 available; 2 ANT at $88.91 cost $177.82, $117.68 short. Shall I raise the money by selling some of your holdings? I will show you what I would sell first." No answer or no is NO: the order is not placed and the reason says so (no is never assumed yes). |
| R4 | **Ask mode, permission in the words (or a yes to R3):** the Account Manager runs the EXISTING RaiseFunds Tree of Thought (`AccountManager.raise_funds(portfolio, target, shares)`, `reasoning/raise_funds.py`) for exactly the shares typed, never selling the stock being bought and never shares a waiting sale has set aside (the door). It SHOWS the plan (which holdings, how many shares, at what price, the gain or loss on each, the cash after) and the user approves the trades as in every Ask run (entry 79: each trade, sells first). A sale the user refuses means the buy is not placed if the cash then still does not cover it (never half). |
| R5 | **Auto mode** asks nothing (entry 104). WITH permission in the words: RaiseFunds runs and its sales and the buy are placed with no question (the user's own words are the approval, like a direct order's). WITHOUT it: nothing is sold, the order is not placed, and the message says exactly what to add: "say 'and raise funds if needed' to let me sell holdings to pay for it". A sale is never made without the user's words or answer (entry 105 holds: a sale comes only from the user's own order or RaiseFunds). |
| R6 | **The desk.** The direct path becomes `account_manager -> account_manager_direct -> account_manager_funds (only when a direct buy needs funding and is permitted by R2-R5) -> trader`, on EVERY desk that can trade a direct order (the page's and `session.py`'s), not only with `--raise-funds`. The `--raise-funds` CLI flag keeps its old meaning on the recommendation path. The funds node's answer is the Trader's funding plan (`perform_trades(..., funding=state["funds"])`, as entry 42 built it). |
| R7 | **The decision.** A funded direct buy is decided `yes_user` (origin `direct`) with `funding: {"shortfall", "plan_id"}` instead of `no_system_denied`. The Trader places the funding sales FIRST, one trade at a time through the handshake (entry 81/82): each sale is approved, filled, booked; only then the buy, whose `approve_trade` reads the cash as it is now. If the sales did not raise enough (a sale refused, rejected, or filled lower), the buy's approval refuses it (the cash is the only check, never overspent: entry 111) and the user is told; the sales that filled stay (they were approved). |
| R8 | **RaiseFunds cannot pay for it** (all holdings together are worth less than the shortfall, or the tree finds no plan): not placed, and the reason names the shortfall and what the holdings could raise. |
| R9 | No size limit, no risk rule, no minimum (AGENTS.md). Money, prices and shares are Decimal through the doors. RaiseFunds sells whole shares (entry 110). |

## Tests (real data)

- The model's real reply to "Buy 2 shares of ANT and raise funds if needed", "Buy 2 shares of ANT", "Buy 2 shares of ANT but don't sell anything", "buy 5 MSFT, sell some stock to pay for it if you have to" (capture, cents; extend `scripts/capture_request_numbers_forms.py` or a sibling).
- A THROWAWAY Postgres account with account 6553's real holdings and cash copied (read 6553 only; never write it), real recorded quotes (capture ANT and the holdings' quotes once from Finnhub into `tests/fixtures/`, allowed): Ask with words (no raise question; plan shown; sells then buy; the book's cash never negative), Ask without words (the question; yes and no), Auto with words (no question), Auto without (not placed, the message). The RaiseFunds model's reply: use the existing real captures of `tests/` for the tree if they fit; otherwise capture its real reply for this case (cents).
- One REAL live end-to-end session on a throwaway account copied from 6553 (`session.py` or the page's `desk_run` with a scripted approver that answers as a user would): allowed by Jacob's standing yes for paid runs that test the desk; NO fetch; save it as `tests/fixtures/session_real/ant_raise_funds_ask.json`.
- Every existing test stays green; list any whose expected value changed.

## Open for Jacob

- R5 (Auto mode sells on the words alone) is the reading of entry 104 + 105 chosen here.
- Raising funds for a purchase of the recommendation path with typed shares (entry 140 Part C) is not built. (Built: see the addendum.)

## Addendum (2026-11-21, Jacob's later words, as built)

- R5 confirmed by Jacob: Auto mode sells when the words say raise funds.
- R10-R11: a recommendation purchase with TYPED shares (entry 140 Part C) is funded the same way, only when EVERY approved or neutral purchase has typed shares, within the budget; one plan for the whole shortfall that never sells a stock being bought; Auto without permission keeps the round robin with the cash.
- R12: the setting `ALWAYS_RAISE_FUNDS` (Jacob put `ALWAYS_RAISE_FUNDS=false` in `.env`) and the page's box "Always raise funds" (default off, one run) are the same as the words in every request; "don't sell anything" in the request still wins.

## Addendum (2026-11-21, entry 143): the four open ends, as built

- S1-S3: the user's own sale in the same request is cash for the request's purchases (placed first; estimate at the quote; the approval still reads the real cash), the shortfall is net of it, and RaiseFunds works from the account once it is filled, so it never sells the share the user sold.
- P1: when only some purchases of a recommendation run have typed shares, the typed ones are funded in full and the others get nothing (never money raised by selling); without permission nothing changes.
- W-F1 to W-F5 (Jacob: "Sell when the price is reached"): a waiting PURCHASE the free cash does not cover waits when the user's WORDS (or ALWAYS_RAISE_FUNDS) permit it and the holdings could pay at its limit; it sets aside the free cash, carries `raise_funds_by`, and sells nothing while it waits or if it expires; when the watcher meets the limit it raises the shortfall at that price (RaiseFunds, the Account Manager's model) and sells first, buys after. No question (nobody is there). A waiting sale is never funded.
- B1: `Venue.report_fill`, `PaperBroker._sales_fell_short`, reject code `sales_fell_short`: sales that were booked and brought less than planned stay, the purchase is refused with the cash as the only check, and the run says so.
