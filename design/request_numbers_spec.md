# Reading the request: dollars or shares, a price limit, "skip recommendations" (spec)

Jacob, 2026-11-20 (night), in his words:

> Adding the phrase "skip recommendations" did not skip the Advisor and still generated recommendations.
> Add: price limits like "under $50 a share".
> In the main request prompt from the user, if they use a number with a $ (dollar sign) or word "bucks" or "dollars" then they are referring to using price. Without these they are referring to quantity.
> Examples: "I want to buy $1000 of MU." means to purchase one thousand dollars worth of stock symbol MU (Micron). "I want to buy 10 of MU." means to purchase 10 shares of stock symbol MU (Micron). Change the intent.py and other code as needed.

Design choices entry 138 (written by the main session). The numbers stay a PROGRAM's decision (they move money: [[feedback-never-spend-more-than-available]] and `design/request_reading_spec.md` keeps numbers checked by code); the model's prompt is updated to read them the same way, and code has the last word.

## 1. The bug of "skip recommendations"

Real request (saved session `data/proving/sessions/58dd133075ef.json`, audit log 2026-11-20T23:27:45Z): **"Buy $2500 worth of symbol KO and skip recommendations."** `intent.RECOMMEND_CUE` matches `recommend\w*` in "recommendations", so `as_direct_order` and `parse_intent` (line ~508) send it down the recommendation path. The words ASK FOR NO recommendation.

| # | Rule |
|---|---|
| S1 | A recommend cue that is NEGATED is not a recommend cue. `intent.NO_RECOMMEND` (one closed pattern): a negating word (`skip`, `skipping`, `no`, `without`, `don't need`, `do not need`, `don't want`, `do not want`, `never mind`, `forget`, `bypass`, `not` + `(a )?recommend`) followed within 3 words by a recommend-cue word (`recommend\w*`, `advice`, `advisor`, `research`, `analysis`, `analyze`, the Critic), e.g. "skip recommendations", "no research", "without the advisor", "don't need a recommendation". One helper `asks_for_recommendation(typed) -> bool` = RECOMMEND_CUE found OUTSIDE every negated span. Every current reader of `RECOMMEND_CUE.search(typed)` goes through it (intent.py lines ~444, ~508, ~580, and any other in the tree: grep). |
| S2 | "Buy $2500 worth of symbol KO and skip recommendations." is then a DIRECT ORDER of $2,500 of KO (entry 98's path: no Data Clerk node, no Advisor, no Critic). Test on the real text and on the model's REAL reply (capture it: see section 4). |
| S3 | A request that says to skip recommendations and names NO stock (a theme only: "buy AI stocks, skip recommendations") cannot be served (only the Advisor chooses stocks for a theme): the run stops with the clarify message (`direct.CLARIFY` path, `clarify: True`), worded "name the stocks to buy, or let me recommend some". |

## 2. Dollars or shares

| # | Rule |
|---|---|
| D1 | A number is MONEY only when written with `$` before it, or `dollars`/`dollar`/`bucks`/`usd` after it (with or without `k`/`thousand`/`m`/`million`: "$5k", "5 thousand dollars", "1000 bucks"). A percentage stays a percentage ("20% of my cash"). |
| D2 | A bare number right before a stock (ticker or company, with or without `shares`, `of`, `in`) is a NUMBER OF SHARES: "buy 10 of MU" = 10 shares, "buy 1000 MU" = 1000 shares, "buy 10 shares of MU" = 10 shares. The existing `_order_phrases` pattern already allows `of`; check why "10 of MU" is not read as shares end to end (the model's reading, `check_order`, `check_amount_was_read`, `money_amounts`/`MONEY`, which accepts a number with no `$`) and fix the cause, not the symptom. |
| D3 | A bare number before `stocks`/`companies`/`names`/`picks` ("the top 3 AI stocks") is the COUNT, as today. |
| D4 | The code decides the kind from the written form; when the model's reading puts a number in the wrong kind (a bare 1000 as a budget, a $1000 as shares) the code's kind wins and the reading is corrected in code (logged on the intent as `corrected: [...]`). Never a guess the other way. |
| D5 | Safety (real money): a bare number after a money verb (`spend`, `invest`, `put ... into`) with no `$`/dollars/bucks ("spend 1000 on MU") is AMBIGUOUS: Ask mode asks the user "1000 shares or $1,000?" (the existing how-much question path, `clarify.py`); Auto mode stops with the clarify message. It is never bought as 1000 shares without a question. |
| D6 | The model's prompt for `understand_request` (in `prometheus/prompts/`) states D1 to D3 with Jacob's two examples word for word. |

## 3. A price limit ("under $50 a share")

| # | Rule |
|---|---|
| L1 | A PRICE LIMIT is a `$` number (D1) tied to one share: followed by `a share`, `per share`, `each share`, `/share`, `a pop`; or preceded by `under`, `below`, `less than`, `at most`, `no more than`, `cheaper than`, `priced under` when it is not followed by `of`/`worth`/`in` a stock ("under $50 a share", "stocks below $20", "priced under $100"). It is `TradeIntent.max_price: Price | None` (`money.price`, Decimal) and `DirectOrder.max_price` for a direct order. It is NEVER a money amount: `money_amounts`, `check_amount_was_read`, `check_dollars`, `check_each` skip its span. "Buy $1000 of MU under $50 a share" = a budget of $1,000 and a limit of $50. |
| L2 | Only a ceiling is built. "over $20 a share" (a floor) is not: such a request is read as no limit and the user is told in one line that a minimum price is not supported (open item). |
| L3 | **Recommendation path:** the limit goes on the work order's request (where the topic and the count already go) and the Advisor drops, IN CODE, every candidate whose last price (the price its trade plan uses) is above the limit BEFORE ranking, so `top_k` is filled from those under it. Each drop is recorded with its reason on the order (the same place other dropped candidates are listed). Put this in ONE small function in `advisor/recommend.py` (or `advisor/ranking.py`) and nowhere else in the Advisor: another worker is changing `advisor/families.py`, `trade_plan.py` and `configs/advisor_v1.json` at the same time, do not touch those three. The limit is NOT cash and NOT a size: the Advisor's no-portfolio and no-size tests stay green. |
| L4 | **The Account Manager checks it again** (the price moves): `decisions.decide_trades` gives NO decision to a purchase whose sizing price is above the limit (the user is told, like a neutral company beyond the count), and the handshake (`handshake.decide_approval` / `approve_trade`) refuses a purchase whose fill price is above the limit (rejected, shown, never retried: entry 82). A sale is never limited by it. |
| L5 | **Direct order:** "buy 10 of MU under $50 a share" is placed only when the quote is at or under the limit; otherwise it is not placed and the user is told the price and the limit. NO resting order is left (limit orders are still Jacob's TODO: say so in the message). |
| L6 | The Proving Ground replays requests as typed; the limit is read by the same code, so a Time Machine run honours it with no extra code (check it with one cached run if one exists, else a unit test on the Advisor's filter with a real recorded candidate list). |

## 4. Tests (real data; AGENTS.md)

- The model's REAL replies: extend `scripts/` capture used for `tests/fixtures/session_real/direct_order_forms.json` (or add a sibling fixture `request_numbers_forms.json`) with these texts, capturing `understand_request`'s model reply only (OpenRouter, a few cents; allowed by Jacob's standing yes for paid runs that test the desk; NO fetch, no Postgres): "Buy $2500 worth of symbol KO and skip recommendations.", "I want to buy $1000 of MU.", "I want to buy 10 of MU.", "Buy 1000 MU", "Buy 1000 dollars of MU", "buy 500 bucks of AMD", "spend 1000 on MU", "Buy the top 3 AI stocks under $50 a share", "Buy $1000 of MU under $50 a share", "Recommend 3 oil stocks below $40", "buy AI stocks, skip recommendations", "I want to buy MSFT, no research needed", "Buy 10 of MU if the recommendation passes" (a recommendation: the negation does not apply).
- `tests/test_request_numbers.py` (new) on those replies and texts: S1 to S3, D1 to D5, L1, L2.
- L3 on a real recorded Advisor candidate list from `tests/fixtures/` (find one with prices both sides of a limit). L4 and L5 with the existing real captures and throwaway accounts where Postgres is needed (never the samples, never 8277/8349).
- Every existing test stays green; LIST any existing test whose expected VALUE you changed, with why.

## 5. Order of work

1. S1/S2/S3 (smallest, and Jacob's live bug). 2. D1 to D6. 3. L1 to L6. 4. Full suite.

## 6. Open for Jacob

- A floor price ("over $20 a share"), and a resting limit order for a direct order (TODO since entry 87).
- D5 (a bare number after "spend" is asked about) goes beyond Jacob's words on purpose: buying 1000 shares when $1,000 was meant spends real money he did not intend.
