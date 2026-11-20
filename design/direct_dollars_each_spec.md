# A dollar amount for EACH stock of a direct order is that stock's limit (entry 122)

Found 2026-11-20 on the first LIVE-QUOTE session (`USE_LIVE_TRADING=true`, market open, Auto mode, a throwaway account with $10,000):

```
"Buy $3,000 of XOM and $3,000 of MSFT"
understood: buy XOM (you did not say how many shares), buy MSFT (you did not say how many shares)
T1 buy $2,571.55 of XOM  FILLED 15.946607 at 161.26     T2 buy $7,428.45 of MSFT  FILLED 15 at 495.23     cash $10,000.00 -> $0.00
```

The user said $6,000 and the desk spent $10,000. The reading (`intent`) was `budget: null`, each order `shares: null`: the request form has ONE `budget` (a total) and no amount per stock, so the model had nowhere to put "$3,000 of XOM"; `AccountManager._size_direct` then took `how: "cash"` (all the cash, round robin, entry 106). An amount for each stock exists today only as an ANSWER to the Ask-mode questions (`clarify.py`, entry 111 M3). Jacob's rule (entry 111): *"It is impossible to spend more than the amount in the portfolio and budget"* — the user's money words are the limit.

## Rules

| # | Rule |
|---|---|
| D1 | The request's ORDER gets an amount in money: `DirectOrder.dollars` (a `Money`/`PositiveMoney` of `prometheus/common/money.py`, or None), next to `shares`. The model READS it (the Account Manager's request prompt, `prometheus/prompts/account_manager_request.py` or wherever `understand_request`'s prompt is: one line and the JSON shape; examples with made-up names only, as the prompt already does). An order has `shares` or `dollars` or neither, never both (the form check refuses both). A total for several stocks stays `budget` ("Buy $5,000 of MU and AMD"); "$3,000 of X and $3,000 of Y" is two `dollars`. |
| D2 | CODE checks the number (R2 of `design/request_reading_spec.md`, entry 111 M5): `dollars` must be a money amount the user WROTE (the same `numbers_in` / written-as-money check that already refuses dollars read as shares: `$3,000`, `3,000 dollars`, `3k`…), a positive amount to the cent; a bool or a string is refused at the door. A `dollars` the user did not write is a reading error (the existing path for a reading that fails its check). No word-pattern filter on the stock, the side or the phrase is added. |
| D3 | `_size_direct`: a purchase with `dollars` is sized as "an amount for EACH stock" exactly as the clarify answer of entry 111 M3 is: `wanted` = `shares.buys(dollars, price)` when the account book holds fractions (`fractions_on`), else the whole shares the dollars pay for; it is an ORDER IN DOLLARS (`dollars` on the decision, `in_dollars`), so the handshake re-sizes at the fill price and never past it (M2). The POT of the run is the sum of the per-stock dollars (never above the cash: `sizing.pot_of`). `how: "dollars"`. The cash is still the only other limit: with less cash than the sum, the sizing's own rule cuts (no new rule). |
| D4 | `said_how_much()` is true when every purchase has `shares` or `dollars` (or the request has a `budget` / "as much as I can"), so Ask mode does NOT ask a question the user already answered. |
| D5 | FAIL CLOSED on a misread amount: when the user's text contains a money amount (code: `numbers_in` finds a number written as money) and the reading has NO `budget`, NO `dollars` on any order and NO `shares`, the Account Manager does not fall back to all the cash: it is a request to clarify (`direct.CLARIFY`, the desk's `clarify: True`, nothing traded, the user is told what was not understood) in both modes. A request with no money words at all keeps entry 98/106 as today (Ask asks; Auto spends the cash round robin). |
| D6 | Mixed requests: a purchase with no number in a request where another purchase has `dollars` or `shares` ("Buy $2,000 of AAPL and MSFT"): ASK mode asks (as today for a purchase with no number); AUTO mode does not place THAT purchase and says so ("say how much of MSFT"), the others go ahead. Never spend the cash on it by default. |
| D7 | Unchanged: no profile/size/risk limit (entries 91, 88); a sale with no number is not placed; RaiseFunds, round robin of the recommendation path, the recommendation path itself, the Critic, the Trader. Money is Decimal end to end (entry 112); the float lint test must stay green. |

## Tests (real data only)

- Capture the REAL model's replies with the amended prompt (standing yes, entry 84; cents) for: `Buy $3,000 of XOM and $3,000 of MSFT`, `Buy $5,000 of MSFT`, `Buy $2,000 of AAPL and 10 shares of MSFT`, `Put $1,500 into NVDA and $500 into AMD`, `Buy $5,000 of MU and AMD`, `Buy $2,000 of AAPL and MSFT`, plus every request of the existing direct-order forms capture (`tests/fixtures/session_real/direct_order_forms.json` and `direct_order_decimal_forms.json`) — into a NEW fixture (do not edit an existing capture). Look at how those captures were made and reuse that script/path.
- Tests on the replies: the readings (dollars per order, budget for the total, shares, the mix); the code checks of D2 (a reply edited by one field: dollars not written, both shares and dollars, a string); sizing with the REAL stored closes or the real quotes in the capture: XOM+MSFT spends at most $3,000.00 each and $6,000.00 in all, exactly to the cent with fractions on (Decimal compares, never approx); D5 and D6 decisions. Every existing direct-order and account-manager test still passes (update one only if it pinned the old all-cash behaviour for a request WITH money words, and say which).
- After the tests pass, ONE real live session (standing yes for paid runs that test the desk; a throwaway account; `.env` now has `USE_LIVE_TRADING=true`, the market is open): `~/miniconda3/envs/cmu/bin/python -m prometheus.application.session --name "Dollars Each" --age 45 --risk moderate --cash 10000 --mode auto --prompt 'Buy $3,000 of XOM and $3,000 of MSFT' --yes --cleanup --save tests/fixtures/session_real/auto_direct_dollars_each.json`. It must spend at most $6,000.00 and leave the account deleted. Add a test that reads that capture (the money identities, the pot, verify_trades). No `--fetch`.
- Full suite: `~/miniconda3/envs/cmu/bin/python -m pytest -q -p no:cacheprovider --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo`. Known failures: `test_approved_or_denied`, `test_rag_loaders`, `test_trader_handshake` (resting order), and after 12:00 UTC `test_data_clerk_verification::test_recent_sources_respects_the_window_the_symbol_and_k`.

## Do not

Two other workers are active: one edits `prometheus/agents/data_clerk/` and adds a prompt in `prometheus/prompts/` (theme membership), one edits `prometheus/security/`, `configs/security*`, `prometheus/prompts/guard.py` and the security/eval tests. Do not touch those. Do not edit `docs/design_choices.md`, `AGENTS.md`, `HowToRun.txt`, `README.md` (the main session writes entry 122 from your report). Never read or edit `.env`. No fetch. Do not commit. Leave no `__pycache__`.

## Report

Files and functions changed, the captured readings, the live session's result (fills, spent, cash left), test results, cost, and the text of entry 122 (problem, decision, alternatives, consequences).
