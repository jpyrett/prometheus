# Ignore Critic: skip the Critic step for one run (design choices entry 148)

Jacob, 2026-11-23: *"The ability to Ignore (Skip) Critic step. Change the main LangGraph workflow and add a new Checkbox: Ignore Critic with defaulted to false. This will allow the User to continue with denied Trade order if the recommendation was approved yet the Critic denied it. ... When "Ignore Critic" is off (normal behavior), the user will not be asked to buy these denied recommendations."*

Jacob's answers to the four questions (same day):

| Question | Answer |
|---|---|
| ON lets a recommendation the Critic would deny trade; OFF keeps "a denial is final" | Yes |
| Does the Critic still run with the box on? | **No: skip the `critic` node.** The desk goes `advisor -> account_manager_decision` |
| Ask vs Auto with the box on | **Ask mode asks** (each purchase gets the usual card), **Auto trades** (`yes_auto`, nothing asked: entry 104 stands) |
| Which denials may be overridden | Every one. Because the Critic does not run, there are no denials. Every Advisor recommendation counts as approved |

## 1. Rules

| # | Rule |
|---|---|
| IC1 | **One setting, one run, default OFF.** The page has a checkbox **"Ignore Critic"** on the request card, unticked by default, sent as `ignore_critic` (a missing value is false). The CLI session has `--ignore-critic`. Nothing is saved to the account, and nothing is read from or written to `.env`. |
| IC2 | **OFF changes nothing.** Every existing test passes unchanged and no printed line changes. The desk, the decisions and `verify_trade` behave exactly as today. |
| IC3 | **ON changes the graph's ROUTE, not its nodes.** `after_advisor` sends a recommended order to `account_manager_decision` instead of `critic` when the run's state says `ignore_critic`. The `critic` node is not run, so neither the Critic's model nor its Finnhub re-asks are called, and `guard.trust` (the Trust Score that follows the Critic) is not run either. The node stays in the graph for the runs that use it. A desk built without `with_critic` ignores the flag. A direct order already skips the Critic and is unchanged. "Research only" still ends after the Advisor. |
| IC4 | **Every Advisor recommendation counts as approved.** Each record on `order.recommendations` (`action: "open"`, schema 3) gets a trade decision with `origin: "advisor"` (new constant `work_order.ADVISOR`, beside `NEUTRAL` and `DIRECT`), `review_id: None`, no review, and its own `valid_until_utc` (now + `configs/account_manager_v1.json` section `ignore_critic.valid_minutes`, the same default as `neutral.valid_minutes`). Its `reason` says that the Critic was skipped by the user's "Ignore Critic" setting and gives the Advisor's score. A record whose action is not `open` gets no purchase (entry 105, as today). |
| IC5 | **Ask asks, Auto trades.** Auto: `yes_auto`, nothing asked. Ask: `pending_user` and a card for each, in the SAME proposal as the neutral purchases. The card says plainly that the Critic did NOT review this recommendation (Ignore Critic is on) and shows the Advisor's score and thesis. With nobody to answer it stays `pending_user` (no answer is never a yes). |
| IC6 | **Sizing, count, limits and money are unchanged.** The Advisor's purchases and the neutral ones are sized TOGETHER in the one `sizing.size_purchases` call (entry 107), round robin unless the words say "evenly". The number asked for is kept: the Advisor's recommendations first, by rank, then the neutral ones by score. `max_price`/`min_price` drop a purchase outside the limit (no decision, the same notes). The pot, dollars orders, the handshake, fractions, typed shares (`shares_of`/`shares_each`), RaiseFunds funding (entry 142/143) and Decimal money (entry 112) all apply exactly as to a Critic-approved purchase. No size limit, risk rule or liquidity rule is added (entries 91, 93, 94). |
| IC7 | **The price.** A Critic-approved purchase is priced from the Critic's deal price. With no Critic, the Account Manager prices it the way it prices a neutral purchase (its own quote through `DataClerk.request_quotes`, converted at the door `AccountManager._prices_of`). A recommendation with no current price is `no_system_denied` ("no current price"), as for a neutral one. |
| IC8 | **The word alone permits nothing.** `work_order.verify_trade` accepts `origin: "advisor"` ONLY when ALL of these hold: the order's `ignore_critic` is true (a new work-order field, default False, set by the desk when it routes around the Critic and written with the order), the Advisor's own record with that `record_id` and symbol is on `order.recommendations` with `action: "open"`, there is NO review of it on `order.reviews` (if the Critic did review it, its verdict stands, so a denial can never be relabelled), the decision is a purchase, and its own `valid_until_utc` has not run out. `verify_trades` counts these purchases in the pot like any other. |
| IC9 | **Strict and Ignore Critic.** Strict means "only what the Critic approves trades", so the two cannot both be on. On the page, ticking Ignore Critic unticks and disables Strict, with the title "Ignore Critic: the Critic does not run". The server enforces the same rule: `run_settings` returns `strict=False` whenever `ignore_critic` is true. The Neutral box keeps its own meaning. |
| IC10 | **Shown, not hidden.** The run's options event (`run_started.options`) carries `ignore_critic`. The work order view (`views.py`) shows these decisions with origin "advisor" and the words "Critic skipped". The session's report prints a line "IGNORE CRITIC: the Critic did not review these recommendations (the user's setting)" before the decisions. LangSmith/Langfuse metadata carries `ignore_critic`, and the saved Proving session (`data/proving/sessions/`) records the flag on the order, so the Track Record can tell these runs apart. |
| IC11 | **Out of scope (open items for Jacob).** The Proving Ground's Time Machine does not get the box. The offline evaluator `denied_never_traded` is unchanged (there is no denial to trade). |

## 2. What changes, module by module

- `prometheus/common/work_order.py`: `ADVISOR = "advisor"`, the field `ignore_critic: bool = False`, `verify_trade` (IC8), `verify_trades` (the pot is unchanged, but make sure an `advisor` decision is not refused for having no review), and a helper `advisor_record_of(order, record_id)` if there isn't one.
- `prometheus/langgraph/desk.py`: `request["ignore_critic"]` (or a `DeskState` key) is read in `after_advisor` (IC3). The order's `ignore_critic` is set, and its status moves from `recommended` to `reviewed` so that `decide_trades` accepts it (mirror the neutral-only case at `decisions.py:321`). Update the module docstring's graph picture with the new route.
- `prometheus/agents/account_manager/decisions.py`: `decide_trades(..., advisor_purchases=...)` or an equivalent. The decisions are built with `origin: "advisor"` (IC4, IC5) and sized with the neutral ones in the ONE call (IC6). There is a card builder for Ask mode (IC5). `denied_lines` is unchanged.
- `prometheus/agents/account_manager/account_manager.py`: `AccountManager.decide_trades` builds the advisor purchases (prices per IC7) when the order says `ignore_critic`, and keeps the count rule (IC6).
- `configs/account_manager_v1.json`: section `ignore_critic` with `valid_minutes` (and its loader).
- `prometheus/application/session.py`: `--ignore-critic` passes the flag into the desk request, and the report line (IC10).
- `prometheus/application/web/settings.py` (`RunSettings.ignore_critic`, the Strict rule of IC9), `app.py` (`ignore_critic` read from the body, `defaults.ignore_critic: False`), `desk_run.py` (into the desk's request, the options event, the tracing metadata), `views.py` (IC10), `replay.py` (replays of old captures are unchanged).
- `ui/src/components/RequestCard.tsx` (the box, IC9), `ui/src/api/types.ts`, and the WorkOrderTable label for origin `advisor`. Run the UI's own tests and update `App.test.tsx`/`RequestCard.test.tsx` for the new field. The new body field is the only change to a printed or expected value there.
- `docs/design_choices.md`: entry 148 (problem, decision, alternatives, consequences). It records that this is a user's explicit override of entry 104's "a denial is final" and of the Critic's place in the desk, ONLY for a run where the box is ticked. `AGENTS.md`: one sentence in the Critic bullet. `HowToRun.txt`: the CLI flag.

## 3. Tests (real data, no stand-in agents for new tests)

- `tests/test_ignore_critic.py`:
  - The graph: with the flag the `critic` node is never run (record the node names from `stream(["tasks"])`) and `account_manager_decision` is. Without it, the route is the old one.
  - The decisions from a REAL Advisor capture (an existing `tests/fixtures/session_real/*.json` with recommendations, its reviews removed, `ignore_critic=True`): Auto gives `yes_auto` for each with `origin: "advisor"`, sized round robin with Decimal money. Ask with an approver that says no gives `no_user_denied`. Ask with none gives `pending_user`.
  - `verify_trade`: accepts an `advisor` decision. It refuses one when the order's `ignore_critic` is False, when there is a review beside it (including a DENIED review: a denial is never relabelled), when the record is not on `order.recommendations`, and when the approval has expired.
  - The count: count 3 with 2 Advisor recommendations and 2 neutral gives the 2 first, then 1 neutral.
  - Settings: `run_settings(strict=True, ..., ignore_critic=True).strict` is False.
- The web tests: `POST /api/runs` with `ignore_critic` reaches `run_started.options`. A missing field is false.
- The whole Python suite (the one deselected test as usual) and the UI tests pass. No existing expected value changes, except the request body in the two UI tests.
- A REAL capture after the code is in, allowed under the standing yes for paid runs that test the desk: one Auto session on a throwaway account with `--ignore-critic --no-fetch --cleanup --save tests/fixtures/session_real/auto_ignore_critic.json`. The capture is checked by a test (no Critic review on the order, the decisions are `origin: "advisor"`, the trades are within the pot).

## 4. Order of work

work_order (IC8) -> decisions + AccountManager (IC4 to IC7) -> desk route (IC3) -> session CLI -> web settings/app/desk_run/views -> UI box -> tests -> docs entry 148 -> the real capture.
