# Approved or Denied: the Critic's verdict is final, Auto mode asks nothing, and only the Account Manager touches the holdings

Decided by Jacob on 2026-11-15, in his words:

- *"if the Critic wants a person to review. Then Critic needs to Deny the recommendation and return the reason in the WorkOrder and update the recommendation status to Denied. THe AccountManager should look for these denials when it reviews the results from the Critic. THe AccountManager will make the Trade order of the trades which have Approved recommendations. The denied reocmmendations need to be returned to the user from the AccountManager. Update the code and specs if something different is coded for this process"*
- *"Auto mode will not ask any prompts."* and, asked what Auto does with an approved recommendation under the automatic-trading mark: *"It depends if the Strict recommendations is enabled. Trade all approved recommendations in auto mode."*
- Asked what Auto does with an approved sale: *"The Advisor never recommends reducing a holding or any other portfolio related matters. Only the AccountManager can raise funds and recommend this. The Critic never will approve or deny (judge) such a decision that affects the User's holdings. The Advisor and the Critic only approve and deny recommendations and never any holdings or other portfolio actions. This must be fixed in the tests and code"*

- Asked whether the Advisor's `add` (buy more of something held) should stay: *"no, remove the add method and any other Advisor method which adds and manages holdings. The Advisor and Critic only add recommendations and updates the recommendations section with the scoring, the actual recommendation, the status, and other metadata. But never any portfolio related information. The Critic is the same behavior."*

- Told that in Ask mode his request ("I have 10000 to spend ... Buy as much stock in: F, GM, Procter and gamble and Google") put the whole $10,000 into F (839 shares) and left GM, PG and GOOGL with "no funds": *"No. its suppose to be round-robin. Not all in one. ... fix this"*

- On what the first live run of these parts showed: *"neutral passes are driven by the UI checkbox setting for: "ALLOW_NEUTRAL_RECOMMENDATIONS=true" which will treat neutral as a Approval (pass). This is for both modes. Neutral applies to all recommendations regardless of who typied the company. For #2) remove this rule about a company less than 126 bars. This does not matter because it eliminates IPOs from being traded and is a bad rule to be removed from code and tests."* And on the Critic's model: *"They purposely are reviewed by a differnt model so they are less biased and more objective. The critic decides indepently if the recommendation should be approved or denied on purpose using a different model."*

Written by Fable 5.2 as the design advisor. Read `AGENTS.md` first. This amends Advisor v1 (`design/advisor_v1_final_spec.md` 9.7 and the record), Critic v1 (`design/critic_v1_final_spec.md`), and entries 53, 56 to 59, 75, 79, 80 and 91 of `docs/design_choices.md`. It gets new numbered entries at the end; the old entries and specs are amended at their end, never rewritten.

## 1. The process, as Jacob states it

```
Advisor   recommends companies to BUY. Nothing else. It is never given the portfolio.
Critic    judges each recommendation: APPROVED or DENIED, with the reason, on the work order. Nothing in between.
Account   reads the verdicts. Approved -> the trade order. Denied -> returned to the user with the Critic's reason.
Manager   It alone knows the holdings, raises funds, and sells.
```

Three things are coded differently today. Each is a part below.

| Part | Today | After |
|---|---|---|
| A | A review can be approved AND flagged "a person should look" (`requires_human`); the Account Manager then asks the user, in Auto mode too. | A review that wants a person is **Denied**, with that as its reason. |
| B | In Auto mode an approved recommendation under a second mark (`auto_min_score`, 80) is not traded by itself: the user is asked. A sale is never automatic: the user is asked. | In Auto mode **every approved recommendation trades and nothing is asked**. The second mark is gone. Denied recommendations are returned to the user by the Account Manager. |
| C | The Advisor is given the portfolio, reviews the holdings first (the "holdings pass") and may write `reduce` (sell the holding) or `add`; the Critic reviews those, and reads the cost basis once. | The Advisor is given **no portfolio** and writes only recommendations to buy. The Critic judges only those. A sale comes only from the user's own direct order or the Account Manager's RaiseFunds. |

## 2. Part A: a review that wants a person is a denial

**Where:** `prometheus/agents/critic/scoring.py` (`decide`, `summarize`), `report.py`, `feedback.py`, `record.py`.

- `decide`: a finding with `requires_human` is a reason to deny. `denial_reasons` gets `needs_human_review` (after any blocker codes and `score_below_minimum`), and `approved` is false. `requires_human` and `requires_human_reasons` (the finding codes) stay on the review: they say which finding it was.
- **The score is not capped.** The cap (`blocked_score_cap`) is for blockers, which say the recommendation is bad. This denial says the review could not be trusted without a person; the score stays what the dimensions gave.
- The summary says it in words: `DENY YUM: a person must look before this can trade: evidence_not_reviewed (The quotes could not be reviewed by the model: ...)`. The report's line "needs your eyes before any trade" becomes the denial's reason.
- **The recommendation's status** is what the Critic already writes on the record after its review (`recommendation["critic"]`, `critic/feedback.py`): `verdict.approved: false`, `verdict.decision: "deny"`, the summary and the reasons. Check that a `needs_human_review` denial is written there like any other, and that the reason is on the work order in `order.reviews[]` (`denial_reasons`, `summary`, the findings).
- Nothing else about a finding changes: the same findings are raised by the same checks.

## 3. Part B: Approved trades, Denied is returned, and Auto mode asks nothing

**Where:** `prometheus/agents/account_manager/decisions.py`, `account_manager.py` (`decide_trades`), `prometheus/application/session.py`, the profile, the configs, the account book, the web bridge.

### 3.1 The decision table (`decisions.decide_trades`)

For each review, in order, the first row that fits:

| Review | Decision |
|---|---|
| not approved | `no_system_denied`. Final. The reason is the Critic's own summary. The user is never asked to override it. |
| approved, but `now` is after `valid_until_utc` | `no_expired` |
| approved, but not one share fits | `no_funds` |
| approved, **Auto mode** | `yes_auto`. **Nothing is asked.** |
| approved, **Ask mode** | the user's answer for each trade: `yes_user`, `no_user_denied`, or `pending_user` when nobody is there (entry 79 is unchanged) |

- `_why_user` and every "escalated to the user in Auto mode" path go. `escalated` on a decision is always false; keep the key only if the account book's table needs it (say which you did).
- **The second mark is removed, everywhere:** the `auto_min_score` argument of `decide_trades`, `UserProfile.auto_min_score`, `verdict.auto_min_score` in `configs/critic_v1.json` and its config model, the session's `--auto-min-score` and `with_overrides`, the web bridge's `RunSettings.auto_min_score`, and anything worded from it (a decision's reason now says "approved by the Critic (score 86): Auto mode trades what the Critic approves"). If a band or a grade wording was derived from it, derive it from the grades in the config.
- **The account book** has an `auto_min_score` column. Stop reading and writing it in `persistence/accounts.py`, and drop it in `sql/prometheus_accounts.sql` the way entry 91 dropped the limit columns (`DROP COLUMN IF EXISTS`). Apply it to the local container only if the repo's own setup script is safe to run again on a database that has accounts (read it first); otherwise leave the running database as it is, since the code no longer needs the column either way, and say so in the report. Never touch the three sample accounts' rows by hand.
- **Strict** (the web page's checkbox) now means exactly what Jacob said it means: only what the Critic approves trades and a neutral recommendation never passes. `web/settings.run_settings(strict, neutral)` returns only whether a neutral recommendation counts as approved. Update `design/ui_v1_spec.md` 5.3 and `HowToRun.txt`.
- **A neutral recommendation** (entry 91) is unchanged: with the setting on and Strict off it counts as approved; Auto trades it, Ask asks.

### 3.2 The denied recommendations are returned to the user

The Account Manager tells the user, in its own voice, right after it decides: each denied recommendation with the Critic's verdict and reason.

```
      DENIED by the Critic, not traded:
        APA    score 49 (F): The risk is medium (29.8), above the low level a conservative profile accepts (risk_above_profile)
        YUM    score 84 (B): a person must look before this can trade: evidence_not_reviewed
```

- In the session this is part of what is printed under the decision node (`session.decision_lines`); the web page's feed shows the same lines, and its work-order table and recommendation panel already show the verdict and the blockers.
- `run_finished` / the session's verdict for a run where everything was denied says so in its first reason, with the reasons.

### 3.3 Not in this part

The Trader's question about the sells that pay for a purchase (RaiseFunds, `--raise-funds`; `require_user_approval` in `configs/trader_v1.json`) is the Account Manager's own business and is not on the session's or the page's path. It is left as it is, and listed for Jacob in the report.

## 4. Part C: the Advisor and the Critic never touch the holdings

### 4.1 The Advisor

**It is given no portfolio.** `Advisor.generate_recommendations(order, case, today, top_k, only_subject)`: the `portfolio` argument goes, from the method, from `recommend.generate`, from `candidates.collect_candidates`, and from every caller (`langgraph/desk.py`, `application/recommendations.py`, `scripts/capture_advisor_fixture.py`, the tests). A test reads the source of `prometheus/agents/advisor/` and of the prompts and fails if it imports `Portfolio`, takes a portfolio, or says "held", "holding", "sell" or "reduce" to the model (amend `tests/test_advisor_no_size.py`, which today checks that only `positions` is read).

What goes with it:
- **Step 2, candidates** (`candidates.py`): no held candidates, no `origin: "held"`, no `held` tag, no room taken from the new ideas, no company name taken from a holding.
- **Step 3, the plan prompt** (`prompts/advisor_v1.py`): no "held by the user", no "the user already holds it".
- **Step 9, the holdings pass** (`recommend.py`, `ranking.holding_action`, `add_floor`): removed. Step 9 is only the selection of the new ideas. A symbol the user holds and names is an ordinary candidate, and the Advisor does not know it is held.
- **Step 10, the plan** (`trade_plan.py`): the `family=None` path (a plan with an entry zone and nothing else, for a `reduce`) goes.
- **Step 11, the thesis** (`thesis.py`): the digest and the allowed numbers no longer hold an action other than buying, or any holding number.
- **Step 12, the record** (`record.py`): every recommendation is a recommendation to buy. `Action.action` is `"open"` and nothing else (`add`, `hold`, `reduce`, `avoid` go). `Action.held_shares`, `cost_basis`, `unrealized_pct` go. `Candidate.held` goes. `PortfolioFit` goes, name and all: `sector_weight_after` and `correlation_to_portfolio_63d` were never computed and would need the user's portfolio, and its `note` (how this run's own picks sit together: "three names are one bet") moves to a key that does not say portfolio, `concentration: {note}`, beside the `concentrated` flag the ranking already has. The sort "a reduction first, then additions" goes.
- **No portfolio information on the record, at all** (Jacob). No key of a version 3 record or candidate holds the words portfolio, held, holding, cost_basis or unrealized; a test reads the model's fields and a real new record for them.
- The ledger's cooldown (`ranking.cooldown_of`) stays: it is about what was recommended before, not about what is held. It still reads old ledger lines whose action is `add`.

**Old records still load, and a record about a holding does not.** New records are `schema_version: 3`. The real captures in `tests/fixtures/session_real/` are version 2 and all of them are `open`, not held. The record models read a version 2 record by dropping the retired keys when they hold the not-held values (`held_shares` 0, `cost_basis` and `unrealized_pct` null, the two `portfolio_fit` numbers null, `held` false) and moving `portfolio_fit.note` to `concentration.note`, and **refuse** a record whose action is not `open` or that says something is held. The captures are not edited.

**What each may write on the work order** (Jacob: they "only add recommendations and update the recommendations section with the scoring, the actual recommendation, the status, and other metadata"):

| Agent | Writes | Never writes |
|---|---|---|
| Advisor | `order.recommendations` (the recommendation, its score and its metadata), `order.candidates` (what it scored and did not recommend), `order.advisor_runs`, the order's status | anything about the account, the holdings, a size, a sale; `reviews`, `trade_decisions`, `trades`, `direct_orders` |
| Critic | `order.reviews`, `order.critic_runs`, the status it puts on each recommendation (`recommendation["critic"]`: approved or denied, the score, the reason), the order's status | the same; and it never changes what the Advisor wrote in a recommendation |

A test runs each on a real capture and fails if any other part of the work order changed.

### 4.2 The Critic

**It judges recommendations to buy, and nothing else.** A record that is not one is not reviewed: it is a `record_defect` ("the Critic does not judge a holding or a sale"), as any record that fails to load is today.

What goes (all of it existed for `reduce` or `add`):
- `value.py`: the `adding_above_cost` warning (the one place the Critic read `action.cost_basis`), `value.add_above_cost` in `configs/critic_v1.json` and its config model, and "a reduction is not judged".
- `risk.py`, `chart.py`: "not judged for a reduction".
- `timing.py`, `integrity.py`, `evidence.py`, `claims.py`, `recheck.py`: the `opening = action in ("open", "add")` switches and what they guarded for a reduction (zone part 1.0, stance direction "bearish" for a reduce, the gate check skipped, the fresh sources skipped, the decisive metrics of every passed family).
- `scoring.summarize`, `report.py`: "reduction of", "sell the holding", "add to the position", "the whole holding".
- `prompts/critic.py`: the model is told "another agent recommends to buy X", never another action word.
- `review.py` still copies the action (`open`) onto the review, because the decisions read it.

The Critic's own rule that it is never given the portfolio (`tests/test_critic.py`) stays and now covers the record too: a test fails if a Critic module reads `held_shares`, `cost_basis`, `unrealized_pct` or `held`.

### 4.3 The Account Manager, the work order, the Trader

The words `reduce` and `add` stay in ONE place: the Account Manager's own decisions for the user's own orders (`direct.py`: a direct sell is `reduce`, a direct purchase of something held is `add`), because the Account Manager knows the holdings. They no longer arrive from a recommendation.

- `decisions.py`: no `reduce` branch on the Critic's path (a sale of the whole holding, "a sale is never traded automatically", the sell wording of the card and the proposal). Every review is a purchase and is sized.
- `work_order.py`: `verify_trades` / `_trade_reasons` accept a **sale only for the user's own direct order** (a decision with `origin: "direct"`) or a funding sale of RaiseFunds; a sale tied to a recommendation is refused. `is_neutral_candidate` no longer reads `held`: the reason it was there was the holdings pass. A company the user named that the Advisor scored neutral counts as approved whether or not the user already holds some (the Account Manager sizes the purchase, and knows).
- `trader/tickets.py`: the branch that cuts a recommended sale to the shares sellable today goes; a sell ticket is a direct order's (whole or not at all) or a funding sale's.
- `langgraph/desk.py`: the Advisor node passes no portfolio. The Data Clerk node still refreshes the bars of the holdings when asked to fetch (`bars_for`): that is for the Account Manager's prices of what is held (the simulated clock prices a holding from its stored bar), and the comment says so.
- `application/recommendations.py`: the "YOUR HOLDINGS (reviewed before the new ideas)" section of the report goes, with what fed it.
- `sql/seed_test_users.sql` is not changed (the samples' rows and counts are pinned).

## 4a. Part D: the money is spread round-robin in Ask mode too

**What happened** (the real session `tests/fixtures/session_real/d_pyrett_names_ask.json`, 2026-11-15): four purchases named with no number and $10,000 to spend, Ask mode. The Account Manager sized them "as much as the cash allows, in rank order": F took 839 shares ($9,997.52) and GM, PG and GOOGL were `no_funds`. The same request in Auto mode (`d_pyrett_names_auto.json`) was spread round-robin: F 23, GM 19, PG 18, GOOGL 17.

**The rule (Jacob):** when the user did not say how to spread the money over several purchases, it is spread **round-robin, in both modes**: one share of each purchase in turn, in rank order, until the funds are used. "Split it evenly" still gives equal dollars. Never everything into the first.

**Where:** the two calls that choose the default today, `AccountManager._size_direct` and `decisions.decide_trades` (both say `balance or ("round_robin" if mode == "auto" else None)`, and `None` is "as much as possible"); `sizing.py` (its own default and module text); `clarify.py` and its prompt (the how-much loop tells the model "use allocation as_much_as_possible unless they said how to spread it", and says "the first company first, as far as your cash allows").

- The default allocation is `round_robin` wherever a default is chosen, whatever the mode. A user who answers the how-much question with a total and no way of spreading it gets round-robin.
- `as_much_as_possible` is no longer chosen by anything when there are several purchases. Remove it as an allocation if nothing needs it. The real captured replies of the how-much model (`tests/fixtures/session_real/clarify_ask_mode.json`) say `"allocation": "as_much_as_possible"`: they must still be read, as "the user did not say how to spread it", which is now round-robin. The captures are not edited.
- An amount the user gave for EACH company ("10 shares each") is still each company's own cap. The neutral purchases keep their own rule (equal dollars, entry 91).
- Ask mode is otherwise entry 79, unchanged: the sizes are made first, the user sees the whole proposal and approves each trade, and nothing is re-split after an answer.
- Nothing here is a limit: the funds covering the order is still the only check.

**Tests:** on Jacob's request with the real prices of that session, Ask mode sizes F, GM, PG and GOOGL as Auto mode did (23, 19, 18, 17) and none is `no_funds`; the research path in Ask mode (`bob_jones_oil.json`'s approved purchases with a budget and no "evenly") is round-robin; "split evenly" is still equal dollars; a real how-much reply that says `as_much_as_possible` over several purchases is sized round-robin. Tests that pin "Ask mode: as much as the cash allows, the first company first" are amended, each with a line saying why.

## 4b. Part E: a neutral recommendation is an approval, in full

**What happened** (the real session `tests/fixtures/session_real/auto_oil_no_second_mark.json`, Auto mode, the setting on): the request was "Buy the most amount of oil stocks shares from the **3** most recommended companies. Balance the purchased stocks evenly among the 3." The Advisor recommended EOG, CVX and COP; the Critic approved EOG and CVX and denied COP. Then **six** neutral companies (CNQ, OXY, BP, META, VLO, FANG) were each decided `yes_auto` as well: eight purchases for a request that asked for three. The two approved ones were sized over all the money, the neutral ones were sized over the same money again, and at the Trader CNQ got one share and the other five were rejected for cash.

**The rule (Jacob):** with the setting on (`ALLOW_NEUTRAL_RECOMMENDATIONS`, which the page's "Neutral recommendations as passes" checkbox sets for the run), a neutral recommendation is an Approval, in both modes, **whoever named the company**. Three things follow.

1. **Every company the Advisor scored neutral counts, not only one the user typed.** `work_order.is_neutral_candidate` drops its `origin == "subject"` condition: a neutral stance and no failed gate is all it needs (a failed gate is still a refusal, and a bearish stance still trades nothing). `neutral.py`, `account_manager.decide_trades` and `verify_trade`'s permission for a neutral decision say "a company the Advisor scored neutral", not "a company the user named".
2. **The number the user asked for is kept.** When the request said how many companies (the form's `count`: "the 3 most recommended"), at most that many are bought: the Critic's approved recommendations first, in rank order, then the neutral companies, the Advisor's best score first, until the number is reached. A denied recommendation does not count. A neutral company beyond the number gets no decision, and the Account Manager tells the user it was neutral and not needed ("3 companies were asked for"). When the request gave no number, nothing is cut (as today). The desk passes the number to `AccountManager.decide_trades`.
3. **They are sized together.** An approved purchase and a neutral one are the same thing to the Account Manager: ONE call of `sizing.size_purchases` over all of them, in the order of point 2 (so "evenly among the 3" is three equal amounts, and the default is one round-robin over all of them, Part D), not one sizing for the approved and a second for the neutral over the same money. The neutral decision keeps `origin: "neutral"` and its own permission check; what goes is its separate sizing.

Ask mode asks the user about each of them, one set of questions, as it does today. Auto mode trades them and asks nothing.

**Tests:** the real session above decided again: three purchases (EOG, CVX, then CNQ, the best neutral score), each about a third of the money, none rejected for cash, and OXY, BP, META, VLO and FANG reported as neutral and not needed; the same with the setting off (two purchases, no neutral); `pam_smith_neutral.json` still gives its two neutral purchases; a request with no number cuts nothing; a neutral candidate whose origin is not `subject` counts (use a real one). Tests that pin "only a company the user named" are amended, each with a line saying why.

## 4c. Part F: no denial for a company with no six-month average

**The rule (Jacob):** remove it. A company with fewer than 126 stored daily bars has no six-month average, and the Critic's `value_not_assessable` finding asked for a person, which since Part A is a denial: a recent IPO could never trade.

**Where:** `critic/value.py`. `value_not_assessable` no longer asks for a person (`requires_human` goes from it), so it denies nothing. It stays as a plain note on the review that the price was not judged against a six-month average, and the value score is worked out as it already is when a measure is unknown. Amend the tests that pin the denial (the real PG_open review with no six-month average is approved at 89 again), the Critic spec's amendment and the design-choice entry of Part A (say there which findings still ask for a person, and that this one does not).

**Not changed, on purpose:** `same_model_family` still asks for a person, so a Critic whose model is from the same producer as the Advisor's denies. Jacob: the Critic is a different model on purpose, so that its verdict is independent. Today's pair (GLM and gpt-oss) is fine.

## 4d. Part G: an even split spends what is left over (whole shares)

**What Jacob said** (2026-11-15, confirming Part E point 3): *"Yes, try to split the money evenly as much as possible. ... Split evenly will try to split the holdings so the balance of each holding is as close as possible. ... $30000 split evenly is $10000 for stock x, $10000 for stock b, $10000 for stock c and the remainder is again divided equally as close as possible."* Asked whether that means fractional shares now, he chose: **whole shares now, fractional shares next** as their own change (`design/fractional_shares_spec.md`, not part of this spec).

**What the code does today** (`sizing._split_evenly`): the equal amount is the pot divided by the number of purchases; each purchase takes the whole shares that amount buys; what is left over is NOT spent. Three purchases can each end a share short of their third and the rest stays as cash.

**The rule.** An even split has two steps.

1. **The equal amount, as today.** `target` is the pot over the purchases in the split, never more than the smallest amount any of them was asked for; a purchase whose equal amount does not buy one share is dropped (`no_room`) and the others are balanced again; each purchase takes the whole shares `target` buys. Nothing here changes.
2. **New: the leftover is spent one share at a time, the smallest amount first.** The leftover is the pot less the cost of the shares of step 1. Each turn, look at the purchases still in the split that have fewer shares than the user asked for and whose NEXT share costs no more than the leftover; the one with the SMALLEST amount so far takes one share (a tie: rank order). Repeat until no purchase can take a share. What is then left (less than any share that could still be taken) stays as cash, and the note says how much.

Details:

- **Money is counted in cents the way the round robin counts it**: a share's cost is `trade_amount(n + 1, price) - trade_amount(n, price)`, in `Decimal`, so a purchase's cost is exactly `trade_amount(shares, price)` and the total never passes the pot. When step 1's own rounding already reaches the pot there is no leftover and step 2 takes nothing.
- **Why not "divide the leftover by N again and repeat"**: in whole shares an equal slice of a small leftover buys only the cheapest stock, so repeating the division feeds the cheapest purchase and makes the split LESS even. "The smallest amount first" is that second division done as closely as whole shares allow: it gives the leftover to whichever purchase is furthest behind and can still take a share.
- **Step 2 does not run when the equal amount was held down by a purchase's smaller ask** (`binding: "even_split"`): the user asked for less than the pot, and the rest is not the split's to spend. In every other case a purchase never passes the shares the user asked for (`wanted`), as today.
- **A purchase dropped in step 1 stays dropped.** It is not looked at again in step 2.
- **`Sized`** keeps its fields: `target_dollars` and `of` are step 1's equal amount and how many shared it. `binding` is worked out as today. The notes say what step 2 did: how many more shares this purchase took from the leftover (when it took any), and what was left at the end and why nothing more fits.
- **The default stays round-robin** (Part D). `even` is chosen only by the user's words (the request's "split it evenly", or the answer to the how-much question), as today. The neutral purchases are in the same call (Part E point 3), so they are treated the same.
- **No limit is added.** The funds covering the order is still the only check.
- **Fractional shares are not in this part.** Every share count stays a whole number.

**Check the page's own example.** The request box's default text is "Buy an even mixture of 3 most recommended stocks in the pizza restaurants using $3000 of my available money". Make ONE live call of `AccountManager.understand_request` on it (the standing yes of entry 84 covers it: no fetch, no account, nothing written) and see that it is read as `allocation: "even"`, `count: 3`, `budget: 3000`. Save the model's real reply beside the other captured forms and pin it in a test. If the model reads it as `default`, mend the request prompt's sentence about `allocation` (add "an even mixture", "equal amounts"); do not loosen the code's check.

**Tests** (real prices and real captures only): on real prices, the total is never above the pot; after the split no purchase that is under its ask could take another share with what is left; every purchase has at least the shares step 1 alone gave it; a split whose equal amount was set by a smaller ask is exactly what it was; a dropped purchase is still dropped. The real session of Part E decided again (EOG, CVX, CNQ, "evenly among the 3") gives three amounts as close as whole shares allow and says what is left. Tests that pin the share counts of an even split whose leftover now buys another share are amended, each with a line saying why. Do not add a test for each boundary value.

**Documents:** a new entry at the end of `docs/design_choices.md` (the next free number: problem, decision, the alternative "divide the leftover by N again" and why it is less even, the alternative "fractional shares now" and that Jacob put it next), the module text of `sizing.py`, the Account Manager paragraph of `AGENTS.md`, `docs/SESSION_NOTES.md`, and `HowToRun.txt` where it describes "split evenly".

## 4e. Part H: one default for every run, neutral-only runs too

**What Jacob said** (2026-11-15, on the worker's note that a run of neutral companies alone kept entry 91's equal dollars): *"use round-robin for neutral-only runs too. Use split evenly if the user specifies it in the prompt and this will override the round-robin if the prompt specifies it."*

**The rule.** There is ONE rule for how the money is spread, whatever the purchases are (approved, neutral, or both) and whatever the mode:

- **Round-robin** when the user did not say how to spread it.
- **Split evenly** (Part G) when the user's own words ask for it: in the request, or in their answer to the how-much question. It overrides the round-robin.

Nothing else chooses an allocation. This amends the sentence of Part D that said the neutral purchases keep equal dollars, and entry 91's "equal dollars" for a neutral purchase.

**Where:** `decisions.decide_trades` (the default it passes to `_size`: today `balance or ("even" if only_neutral and not wanted else "round_robin")`, which becomes `balance or "round_robin"`), the module text of `neutral.py` ("the one place the neutral kind has its own default" goes), the Account Manager paragraph of `AGENTS.md`, a dated amendment at the end of entry 91, and the new design-choice entry of Part G (or one of its own).

**Tests:** the real session `pam_smith_neutral.json` (MU and AMD, both neutral, no split asked) is sized round-robin; its expected shares are worked out from the capture's own prices and the test says what changed and why (it was MU 32, AMD 46 under equal dollars). The same session with "split evenly" in the request still gives equal dollars. Other tests that pin equal dollars for a neutral-only run with no split asked are amended, each with a line saying why.

## 5. Tests

Real data only; no new stand-in agent or model; no test file is deleted.

- **Amend, with a line saying why,** every test that pins the old rules. The map (file and test names; line numbers may have moved):
  - Advisor: `test_ranking.py` (the holdings pass reduces, adds), `test_advisor_v1.py` (a holding with a red flag is a reduce; a holding with nothing to say is a hold), `test_advisor_sndk_pg.py` (GM is an add; a held name as good as the new ideas is an add; its `replay()` passes the sample portfolio), `test_advisor_no_size.py` (the Advisor reads nothing of the portfolio but its positions; a reduction has an entry zone), `test_candidates.py` (a portfolio holding MCD and PG), `test_trade_plan_real.py` (the golden file's `|reduce|` keys), `test_recommendations.py` (the holdings are reviewed in their own section), `test_desk_critic.py` (the Advisor runs with the portfolio).
  - Critic: `test_critic_value.py` (a reduction is not judged; adding above cost), `test_critic_chart.py`, `test_critic_risk.py`, `test_critic_feedback.py`, `test_critic_evidence.py`, `test_critic.py` (tampering with `add`), `tests/simple_critic.py`.
  - Account Manager, work order, Trader: `test_account_manager_decisions.py` (a reduce is the whole holding and is escalated; the mark), `test_account_manager_balance.py`, `test_neutral_recommendation.py` (held excluded), `test_work_order_trader.py`, `test_trader_tickets.py`, `test_trader_broker.py`, `tests/trader_helpers.py`, `test_session_replay.py` (the record's keys), the web tests of `run_settings`.
  - `test_direct_order.py`: a direct sell is still `reduce`. It must still pass.
- **The Advisor's real replays** (`tests/fixtures/advisor_real/`) were captured with the sample portfolio (MSFT, TSLA, GM held), so GM came out as an `add`. Replayed with no portfolio they give other results. Work the new expected values out from the replay itself, and say in the test what changed and why. Do not edit the captures.
- **New tests**, each on real data: a review with a `requires_human` finding is denied with `needs_human_review` and its score is not capped (use a real review that has one, or the real evidence check that raises it); Auto mode trades every approved review of `bob_jones_oil.json` and asks nobody (the approver is never called); the denied one (APA) is in what the Account Manager tells the user, with the Critic's reason; a version 2 record loads, and the same record made to say it is held, or to `reduce`, does not; the Critic refuses to review a record that is not a purchase; a sale tied to a recommendation fails `verify_trades`.
- The whole suite stays green apart from the known failure named in `docs/SESSION_NOTES.md`.

## 6. Documents

- `docs/design_choices.md`: new entries at the end, one for each part (the next free numbers): problem, decision, alternatives, consequences. The alternatives Jacob turned down are part of the record: asking the user in Auto mode; keeping a second mark; an automatic sale; an Advisor that knows the holdings but only labels a purchase `add`.
- Amend at their end (a dated note, the old text left as it is): `design/advisor_v1_final_spec.md` (9.7, the record), `design/week5_advisor_implementation.md` (the holdings pass, step 9), `design/critic_v1_final_spec.md` (the verdict, the reduction rules, `adding_above_cost`), `design/how_to_code_critic.md`.
- `AGENTS.md` (the Advisor, Critic and Account Manager paragraphs), `README.md`, `HowToRun.txt`, `docs/SESSION_NOTES.md`, `design/ui_v1_spec.md` (5.3, the Strict row), and the module texts of every file changed.

## 7. Order of work

1. **Part A** (small, stands alone).
2. **Part B** (the decision table, the mark, the denied list).
3. **Part D** (the default way of spreading the money; small, in the files Part B already touches), then **Part F** (one finding) and **Part E** (neutral is an approval in full). These three were added after A, B and C were finished.
3a. **Part G** (the even split spends its leftover; `sizing.py` only), after Part E. Added after Jacob confirmed Part E point 3. Then **Part H** (one default for every run: a neutral-only run is round-robin too).
4. **Part C**, in this order: the Critic refuses what is not a purchase; the Advisor without the portfolio; the record's version 3 and the reader of version 2; the Account Manager, work order and Trader; the reports.
5. The documents, the whole suite, then one live session in Auto mode on a throwaway account (no fetch) to see a real run end to end with nothing asked.

Each part leaves the suite green before the next begins.

## 8. Open, for Jacob

1. ~~Whether `add` stays~~ Settled by Jacob the same day: `add` goes with `reduce`, and so does anything else of the Advisor that manages holdings.
2. **A neutral company the user already holds** now counts as approved like any other named company (section 4.3).
3. **The Trader's question about funding sells** in Auto mode (section 3.3) is unchanged.
4. ~~Whether approved and neutral purchases are sized together~~ Settled by Jacob the same day: yes, and "evenly" means amounts as close as possible (Part G).
5. **Fractional shares** come next, as their own spec and change (`design/fractional_shares_spec.md`). Until then an even split is as close as whole shares allow.
6. **A purchase whose equal amount does not buy one share is still dropped** from an even split (the rule of entry 80, unchanged in Part G). Fractional shares end the need for it.
