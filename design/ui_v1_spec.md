# Prometheus UI v1: the final spec

A web front end for the Prometheus desk. One page: the user's portfolio and profile, a request in their own words, the six stages of the desk lighting up as the five agents work, the Account Manager's words as they arrive, the stocks on the work order, and the recommendation behind each one.

- **Visual reference:** `design/ui_v1_mockup.html` (open it in a browser; the theme button switches dark and light). Its `:root` tokens and its CSS are the source of truth for the look. When this text and the mockup disagree about a colour or a spacing, the mockup wins; when they disagree about behaviour, this text wins.
- **Written by** Fable 5.2 as the design advisor, 2026-11-15, for Jacob. **Built by** a separate coding worker.
- **Status:** final, and built (2026-11-15: `ui/`, `prometheus/application/web/`, design choices entries 99 and 100). What the two checkboxes do (section 5.3) was confirmed by Jacob the same day. **Amended the same day (Jacob):** a **Mode** switch, Ask / Auto, on the request card, hooked to the Account Manager (section 5.3, "Mode"); it replaces the Trades control of the Profile card.

Read `AGENTS.md` first. Everything in it applies to this work.

---

## 1. What this is for

A person who is not at a terminal should be able to see two things at a glance: **what is in my account** and **what the desk is doing with my request right now**. The UI adds no trading logic. It is a window on what `prometheus/langgraph/desk.py` already does, driven the same way `prometheus/application/session.py` drives it.

**Goals**
1. Show off the back end: the LangGraph workflow, the five agents, the hand-offs, the human in the loop.
2. Show the portfolio before, during and after, from the real Postgres account book.
3. Stay simple: one page, no routing, no login, one user at a time.

**Not in v1:** live trading (the Trader is simulated, always), charts of prices, run history, several users at once, live market prices in the portfolio (section 12), a mobile layout beyond "it stacks and stays usable".

---

## 2. The decisions, and why

| # | Decision | Why |
|---|---|---|
| D1 | **React 19 + Vite + TypeScript** in `ui/`. No UI kit, no state library, no router. Plain CSS with the mockup's tokens. | The class recommends React. One page with one reducer does not need more, and hand-written CSS is what makes it look designed and not assembled. |
| D2 | **FastAPI** in `prometheus/application/web/`, in the same Python process as the agents. | The agents are Python objects; the bridge should be a thin layer over `build_desk`, not a second system. FastAPI is the conventional choice and `uvicorn` is already a dependency. |
| D3 | **Server-Sent Events** for server to browser (`GET /api/runs/{id}/events`), plain **POST** for browser to server. No WebSocket. | The traffic is one-way except for a handful of answers. SSE is plain HTTP, reconnects by itself, replays from an event id, and can be read with `curl`. |
| D4 | **Progress comes from LangGraph itself**: `graph.stream(..., stream_mode=["tasks", "values"])`. `tasks` says a node started and ended; `values` gives the work order after each node. `desk.py` is not changed. | The graph already knows. No hook has to be added to the 2,774-test core for the stepper to work. |
| D5 | **Langfuse is optional and off by default** (section 7). It is a second consumer of the same run, switched on by keys in the environment. LangSmith needs no code at all. | The UI must work with no account anywhere. Tracing is for looking inside a run afterwards, not for drawing the stepper. |
| D6 | **The UI answers the Account Manager's questions** (confirm, how much, approve a trade) with cards in the Activity panel. Both trade modes work: *Ask me first* and *Auto-trade*. | In Ask mode nothing trades without a yes, so a UI that cannot answer could never trade. It is also the human-in-the-loop part of the demo. |
| D7 | **The UI uses its own account** in the account book: created from the Profile form on first use, remembered by number. A sample account (one with a `seed_key`) can be looked at and never written. | `tests/test_account_book.py` pins the three samples' counts. |
| D8 | **A replay mode** (`--replay <capture.json>`): the server plays a real captured session through the same event pipeline and the same view code. No model, no quote, no database. | The UI can be built, tested, screenshotted and demoed for free, and the tests use real data, as the repo's rule asks. |
| D9 | **One run at a time**, one process, bound to `127.0.0.1`. | One setting is process-wide (section 5.3), the agents are heavy, and this is a single-user desk. |
| D10 | A third checkbox: **Refresh market data first (paid)**, **on by default** (Jacob, 2026-11-15: *"Make this on by default"*; it was off and folded away in the first build). It is in plain sight beside the other two, never in a fold. | The Advisor's freshness gate refuses stale bars. Without a refresh, a demo on old data recommends nothing. Because it is on unless unticked, a live Go! calls the paid data APIs and writes the data files: an option that spends money by default must be visible, and the page says "paid" on it. The API's own default stays off: a client that does not send `fetch` fetches nothing. |
| D11 | **Portfolio value is shares × average cost** in v1, and says so. | It is what the account book knows without a paid quote. Live prices are a follow-up (section 12). |

---

## 3. Architecture

```
 Browser (React, ui/)                         Python process (cmu env)
┌───────────────────────────┐   POST /api/runs            ┌──────────────────────────────────────────────┐
│ RequestCard  ── Go! ──────┼────────────────────────────▶│ prometheus/application/web/                  │
│ ProfileCard ──────────────┼── /api/accounts/... ───────▶│   app.py      FastAPI routes                 │
│                           │                             │   runs.py     RunManager: one run, a thread, │
│ Pipeline, Activity,       │◀── SSE /api/runs/{id}/events│               an event log, open questions   │
│ WorkOrder, Recommendation,│   stage · log · question ·  │   desk_run.py the live run ──┐               │
│ Portfolio                 │   work_order · portfolio ·  │   replay.py   a capture ─────┤ same events   │
│                           │   run_finished              │   views.py    order -> rows  │               │
│ QuestionCard ── answer ───┼── POST .../answers ────────▶│   tracing.py  callbacks      ▼               │
└───────────────────────────┘                             │        build_desk(...).stream(["tasks","values"])
                                                          │        account_manager -> data_clerk -> advisor
                                                          │        -> critic -> account_manager_decision -> trader
                                                          │        (the real agents of session.build_parts)
                                                          │        AccountManager <-> Postgres account book
                                                          └──────────────────────────────────────────────┘
```

- The desk runs in a **worker thread** (`graph.stream` is blocking). It appends events to the run's log under a `threading.Condition`. The SSE route reads the log from any index and waits on the condition, with a comment line every 15 s to keep the connection open.
- When the desk needs the user (`confirm`, `clarify`, `approver`), the worker thread emits a `question` event and blocks on that question until `POST /answers` arrives, the run is stopped, or the timeout passes (600 s by default). No answer is never a yes: it returns `None`, exactly what the terminal versions return when nobody is there.
- In production mode FastAPI also serves the built UI (`ui/dist`) at `/`, so the demo is one command and one port. In development Vite serves the UI on 5173 and proxies `/api` to 8000.

---

## 4. The back end: `prometheus/application/web/`

| File | What it holds |
|---|---|
| `__init__.py`, `__main__.py` | `python -m prometheus.application.web [--host 127.0.0.1] [--port 8000] [--replay FILE] [--replay-speed 1.0] [--question-timeout 600]` |
| `app.py` | `create_app(...)`: the routes of 4.1. No logic beyond validation and calling the others. |
| `runs.py` | `Run` (id, status, the event log, open questions, stop flag) and `RunManager` (start, get, the one-run lock, the last few finished runs kept in memory). Knows nothing about agents. |
| `desk_run.py` | The live run (4.3): builds the real parts, the desk and the request as `session.run_session` does, streams, emits. |
| `replay.py` | The replay of a capture (4.6). |
| `views.py` | Pure functions, no I/O: `account_view`, `work_order_view`, `stage_states`. The only place that knows the shape of a recommendation, a review, a decision and a trade. |
| `settings.py` | The two checkboxes as run settings (5.3). |
| `tracing.py` | The callbacks handed to the graph (section 7). |

Reuse, do not copy: `session.build_parser`, `check_args`, `build_parts`, `with_overrides`, `NODE_WORDS`, `decision_lines`, `status_lines`, `trade_lines`, `use_of_money_lines`, `why_not`, `changed`, `holdings_of`, `money`; `direct.direct_section`; `neutral.neutral_lines`; `critic.report.DECISION_WORDS`; `recommendations.KEY_NUMBERS`; `trading_clock.clock_note`. **Do not change `session.py` or `desk.py`.** If something there truly has to change, stop and say so in the report.

### 4.1 The API

All under `/api`. JSON in and out. A refusal is `{"error": "<a message for the user>"}` with 4xx; the message names the thing and never a value from `.env` or a connection detail (the same rule as `desk.saved_account`).

| Method and path | Body | Answer |
|---|---|---|
| `GET /api/health` | | `{ok, mode: "live"\|"replay", book: "ok"\|"<kind of failure>", clock_note, defaults: {prompt, strict: false, neutral: true, fetch: true}, limits: {prompt_chars: 600}, active_run: id\|null}` |
| `POST /api/accounts` | `{name, age, risk, mode, cash}` | 201, the account view (4.5) |
| `GET /api/accounts/{id}` | | the account view; `{id}` is an account number |
| `PATCH /api/accounts/{id}` | any of `{name, age, risk, mode}` | the account view |
| `POST /api/accounts/{id}/funds` | `{action: "add"\|"remove", amount: "500.00"}` | the account view |
| `POST /api/runs` | `{account_id, prompt, strict, neutral, fetch, mode}` | 202 `{run_id}`; 409 when a run is active. `mode` (`"ask"` or `"auto"`, optional) is what the Mode switch shows: when it differs from the account's saved trade mode the server first saves it through the Account Manager (`users.set_trade_mode`), then starts the run, so a run always uses the mode on the screen. |
| `GET /api/runs/{run_id}/events` | header `Last-Event-ID` or `?after=<seq>` | `text/event-stream` (4.2), from the start or from after `seq` |
| `POST /api/runs/{run_id}/answers` | `{question_id, answer}` (`true`/`false` for confirm and approve, a string for how-much) | 204; 409 when the question is closed |
| `POST /api/runs/{run_id}/stop` | | 202. Open and later questions answer "no answer"; the desk stops before its next stage. A trade already placed stays. |
| `GET /api/runs/{run_id}/records/{key}` | | the raw records of one work-order row: `{recommendation, review, decision, trade, candidate, direct_order}` (whichever exist) |

- `amount` and `cash` are **strings** (`parse_money` refuses a float). Every other check is the book's own: `clean_display_name`, `clean_age`, `clean_risk_level`, `clean_trade_mode`, `parse_money`. A `ValueError` becomes a 422 with its message.
- Writes to an account with a `seed_key` are refused with 409: "sample accounts are read-only here". A run on one is refused the same way.
- In replay mode the account routes answer the replay's account and refuse every write with 409.
- CORS: only `http://localhost:5173` and `http://127.0.0.1:5173`, and only when started with `--dev`.

### 4.2 The event stream

Each SSE message is `id: <seq>` / `event: <type>` / `data: <json>`. `seq` counts from 1 inside a run. A client that connects late gets every event from the start, so a page reload in the middle of a run redraws everything.

| `type` | `data` |
|---|---|
| `run_started` | `{run_id, mode: "live"\|"replay", account_id, prompt, options: {strict, neutral, fetch}, trade_mode, started_utc, stages: [{key, label, agent, agent_label}]}` |
| `stage` | `{key, state: "busy"\|"waiting"\|"done"\|"skipped"\|"failed", node, seconds}` |
| `log` | `{agent: "account_manager"\|"data_clerk"\|"advisor"\|"critic"\|"trader"\|"system"\|"you", level: "info"\|"good"\|"warn"\|"error", text, block: bool}`. `text` may hold several lines; `block: true` is a report block (kept in monospace with its line breaks). |
| `question` | `{question_id, kind: "confirm"\|"clarify"\|"approve"\|"missing_company", title, text, facts: [lines], n, of, symbol, shares, est_cost, cash, proposal: [...], scorecard, typed, reason, kept: [{typed, symbol, company, how}], wording: "model"\|"code"}` (the fields the kind has) |
| `question_closed` | `{question_id, answer, by: "you"\|"timeout"\|"stopped"}` |
| `work_order` | the work-order view (4.4), whole, each time. The client replaces what it has. |
| `portfolio` | the account view (4.5), plus `changes: {SYMBOL: +shares}` and `cash_change` against the portfolio at the start of the run |
| `activity` | `{calls, tokens_in, tokens_out, model, node}`: running totals of the model calls of this run (section 7) |
| `run_finished` | `{status: "traded"\|"no_trade"\|"looked"\|"stopped"\|"failed", headline, reasons: [...], path: [...], seconds: {...}, total_seconds}`. **Always preceded by a `log`** with the headline and, under it, every reason the headline does not hold (green, yellow or red by the status): the feed never just stops. |

### 4.3 The live run (`desk_run.py`)

Do what `session.run_session` does, with emitters where it prints:

1. `args = session.build_parser().parse_args(["--account", str(account_id), "--prompt", prompt, *(["--fetch"] if fetch else [])])`, then the run settings of 5.3 on it. `parts = session.build_parts(args)` inside the run thread, fresh for every run (the throwaway ledgers are what make a run repeatable). Emit a `log` "starting the agents" before and the time it took after.
2. Connect a manager to the account; read the holder, the profile (`with_overrides`) and the portfolio. Emit `portfolio`.
3. `build_desk(clerk_factory, advisor_factory, funds_factory=new_manager, with_critic=True, critic_factory, with_trader=True, trader_factory, approver=web_approver, store_factory, confirm=web_confirm, clarify=web_clarifier)` and the same `request` as the session: `{"ask": {"text", "default_count": 3, "fetch"}, "account": {"account": account_id, "profile": profile}, "trade": {"mode": "paper", "book": True, "model": True}}`. `new_manager` sets `manager.screen` to an emitter of `log` events with `agent: "account_manager"` (what the Account Manager tells the user while it trades).
4. `for mode, chunk in graph.stream(initial_state, stream_mode=["tasks", "values"], config={"callbacks": tracing.callbacks(run)})`:
   - `tasks`, a start: the node's stage becomes `busy`.
   - `tasks`, a result: the stage becomes `done` or `failed` (4.3.1), with its seconds.
   - `values`: for each new name in `state["path"]` emit the `NODE_WORDS` line with its seconds as a `log` from that node's agent, then what the session prints under it (`decision_lines`, the "Data Clerk found" lines, `direct_section`). Then emit `work_order` (4.4). While the plan exists and the order has no recommendations yet, the rows come from the companies the Data Clerk found.
   - Check the stop flag after each chunk; when set, leave the loop.
   - **Verify the exact shape of a `tasks` chunk on the installed LangGraph (1.2.17) with a three-node toy graph before building on it.** If it does not give a clean start and end per node, fall back to `stream_mode=["debug", "values"]`.
5. While the `trade` stage is busy, read the account from the book about once a second and emit `portfolio` when it changed, so each fill lands in the table as it is booked. Stop polling when the stage ends.
6. At the end emit, as `log` blocks, the session's own reports in its order: `status_lines`, `neutral_lines`, `trade_lines`, `use_of_money_lines`; then the final `portfolio`; then `run_finished` with the session's verdict: `traded` (the portfolio changed), `looked` (a look only), `no_trade` with `why_not(...)` as `reasons`, `stopped`, or `failed`.
7. **A request the desk refuses at its first node says why, whether or not a work order was made** (added 2026-11-15 after Jacob's first live run: "Buy as much stock in: F, GM, Procter and gamble and Google. And Chick-fil-a." The Data Clerk answered "Google: Company Name not found." and "Chick-fil-a: Company Name not found.", a direct order is placed as typed or not at all, so the desk ended at `account_manager` with a work order and `errors`, and the page showed nothing). The state's `errors` make `work_order` `failed` ("stopped here"); the run ends `no_trade` with the headline "Nothing was placed: the order could not be made as you typed it.", the Data Clerk's own reasons, and for a direct order the request to clarify (`direct.CLARIFY`, design choices entry 102): "Please clarify your request and send it again: name each stock by its ticker or by the listed company's name." A request that could not be read ends the same way (`failed`, the reason, the same sentence). The Trader's and the money's reports are not printed: nothing was priced. **Amended by entries 101 and 102 (2026-11-15):** Google is now found (as Alphabet Inc.), and a stock that is not listed no longer refuses the whole order: Auto mode drops it without a question and says so, Ask mode asks (the question below); the run still ends this way when nothing was found, a lookup failed, two orders are for one stock, or the user said no or did not answer.
8. Any exception inside the thread ends the run with `failed` and a message that gives the kind of failure only (`type(error).__name__`) unless it is a `ValueError` or `RuntimeError` of our own. The one-run lock is always released, and the settings of 5.3 are always restored.

**The three callables the desk is given**
- `web_confirm(intent)`: always emits a `log` "understood: " + `intent.describe()`. In Ask mode it then asks a `confirm` question and returns its answer. In Auto mode it returns `True` without asking (the CLI's `--yes`).
- `web_approver(card)`: an `approve` question from the card (`symbol`, `shares`, `est_cost`, `action`, `n`, `of`, `cash`, `proposal`, `text` as the scorecard, the card's own `question` when it has one). Returns `True`, `False` or `None`. A card with `kind: "missing_company"` (a stock the Data Clerk did not find, Ask mode, entry 102) is a `missing_company` question instead (`desk_run.missing_payload`: `typed`, `reason`, `n`, `of`, the Account Manager's sentence as `text` and `question`, what stays on the order as `facts` and `kept`, and who worded it as `wording`); its feed line is "continue without Chick-fil-a" or "stop the order: no to going on without Chick-fil-a".
- `web_clarifier(card)`: a `clarify` question (`question`, `n`, `of`, `facts`); returns the typed text or `None`. It has `available = lambda: True` (a person is at the screen).
- While a question is open the busy stage is `waiting`; when it closes it is `busy` again.

#### 4.3.1 Stages

| key | Label | Agent | Nodes |
|---|---|---|---|
| `work_order` | Create Work Order | Account Manager | `account_manager` |
| `gather` | Gather Data | Data Clerk | `data_clerk` |
| `recommend` | Create Recommendations | Advisor | `advisor` |
| `critique` | Critique · Approve / Deny | Critic | `critic` |
| `prepare` | Prepare Trading Order | Account Manager | `account_manager_decision`, `account_manager_direct` |
| `trade` | Trade · Place Order | Trader | `trader` |

Six stages, five agents: the Account Manager has two.

- A stage is `failed` when its node raised; or it is `work_order` and the state has `errors`; or it is `gather` and the order is `data_failed`; or the node added an entry to `order.errors` that says an agent "could not" start, decide or prepare. Otherwise it is `done`.
- When a later stage starts, or the run ends having gone further, every earlier stage that never ran is `skipped` (a direct order skips `gather`, `recommend`, `critique`; a neutral pass skips `critique`).
- Stages after the last one that ran stay `pending`; the UI shows them as "not reached".

### 4.4 The work-order view

```jsonc
{ "request_id": "8a7460ffc3c2", "status": "decided", "kind": "research",   // or "direct"
  "understood": "buy the 3 most recommended 'oil' companies; ...",
  "rows": [ {
      "key": "XOM", "symbol": "XOM", "company": "ExxonMobil", "rank": 1,
      "group": "recommended",          // recommended | neutral | direct | scored | found
      "advisor":  { "score": 73, "stance": "bullish", "reason": "trend: blended score 73 ..." },
      "critic":   { "approved": true, "score": 86, "grade": "B" },                    // or null
      "decision": { "will_trade": "yes_auto", "words": "yes (AUTO)", "action": "open", "shares": 8, "est_cost": 1364.72, "origin": null },   // or null
      "trade":    { "status": "filled", "side": "buy", "shares": 8, "price": 170.6753, "amount": 1365.4 },                                  // or null
      "detail":   { ... }              // 4.4.1
  } ] }
```

Rows, in this order:
1. **direct**: each of `order.direct_orders`, joined to its decision and its trade by `record_id`.
2. **recommended**: each of `order.recommendations` by `ranking.rank`, joined by `identity.record_id` to `order.reviews`, `order.trade_decisions` and `order.trades`.
3. **neutral**: each decision with `origin: "neutral"`, on the candidate with its symbol (so it leaves group `scored`).
4. **scored**: the rest of `order.candidates` that are not held, by `rank` ("not recommended: " + `reason`).
5. **found**: only before the Advisor has answered: the companies the Data Clerk found (symbol and company, nothing else yet).

A trade that matches no row gets a row of its own. `words` is `DECISION_WORDS[will_trade]`.

#### 4.4.1 `detail`, by where it comes from

| Section in the panel | Fields | Source on the record |
|---|---|---|
| Header | score, stance, rank, of how many, strategy | `score`, `stance`, `ranking.rank`, count of scored symbols, `trade.family` |
| Thesis | text | `thesis.thesis` |
| Bull case, bear case | `[{text, ref}]` | `thesis.bull_case`, `thesis.bear_case` |
| Evidence | `[{source, direction, quote, verified, counted, reason}]` | `signals` |
| How the score was built | `[{name, weight, value, available, contribution}]` | `blend.components` (`news`, `performance`, `historical`, `filings`), `blend.weights` |
| Key numbers | `[{label, value}]`, formatted by the server | `features[metric].value` through `recommendations.KEY_NUMBERS` |
| Trade plan | entry zone, stop, target, limit, order style, time stop, horizon, invalidation, notes | `trade.entry_zone`, `stop_price`, `target_price`, `limit_price`, `order_style`, `time_stop_date`, `horizon_bars`, `invalidation`, `notes` |
| Critic review | approved, score, grade, summary, `dimensions: [{name, label, points, weight}]`, blockers, warnings, risk, valid until | `reviews[]`: `approved`, `score`, `grade`, `summary`, `dimensions`, `blockers`, `warnings`, `risk`, `valid_until_utc` |
| Order and fill | the decision (words, shares, cost, reason, mode, balance), the trade (status, reason, fill) | `trade_decisions[]`, `trades[]` |
| Raw record | fetched when opened | `GET /api/runs/{id}/records/{key}` |

A `scored` row has the header, the reason it was not recommended, and its strategies (`families`: name, passed, score). A `direct` row says "Your own order: no recommendation is made and the Critic is not asked", then the order, the decision and the fill. A `neutral` row says the Advisor found no strategy and that a neutral recommendation counts as approved, then the decision and the fill. A section with no data is left out, never shown empty.

### 4.5 The account view, and the profile

```jsonc
{ "account": { "id": 1042, "name": "Bob Jones", "age": 67, "risk": "conservative", "mode": "auto", "sample": false, "read_only": false },
  "positions": [ { "symbol": "XOM", "name": "ExxonMobil", "shares": 8, "cost_basis": 170.6753, "value": 1365.40 } ],
  "cash": 13634.60, "invested": 1365.40, "total": 15000.00, "as_of_utc": "2026-11-13T20:32:14Z" }
```

`value` is `round(shares × cost_basis, 2)`; `invested` is their sum; `total` is `invested + cash`.

The routes go through the **Account Manager**, not the store (the user talks to the Account Manager only): `store.create_account` then `manager.add_funds` for a new account; `manager.users.set_name`, `set_age`, `set_trade_mode`; `manager.set_risk_level`; `manager.add_funds`; `manager.list_portfolio`. Building an `AccountManager` makes a model client and calls nothing.

**Remove funds is new** (the book has deposits only). Add:
- `AccountStore.remove_funds(account_id, amount, when=None) -> Decimal` beside `add_funds`: lock the account row, require it active, refuse an amount above the cash ("the account has $X in cash: $Y cannot be removed"), write one `cash_ledger` row of kind `withdrawal` with a negative amount and the balance that follows. The schema and its trigger already allow it; no SQL changes.
- `AccountManager.remove_funds(amount, when=None) -> Decimal`, which calls it.
- This is not risk management and not a limit on a trade. It is the cash check the book already makes.

### 4.6 Replay (`replay.py`)

`--replay tests/fixtures/session_real/bob_jones_oil.json` (any capture made with `session --save` that has a `path` and an `order`).

- `POST /api/runs` ignores the prompt and plays the capture: `run_started`, then for each node of the capture's `path` a `busy` stage, a pause (`--replay-speed`; 0 means none), `done`, the `NODE_WORDS` line and a `work_order` built by `views.work_order_view` from a copy of the order **cut back to what that node had written** (after `data_clerk`: the subject only; after `advisor`: recommendations and candidates; after `critic`: reviews; after the decision: `trade_decisions`; after `trader`: trades). Same view code as a live run.
- The account is the capture's: `trader_runs[-1].portfolio_before` at the start and `portfolio_after` at the end; the risk level from a review's `risk.tolerance_stated`, the mode from a decision's `mode`. A direct-order capture with no trader run starts from what it has.
- The seconds shown are the replay's own pacing. The header says **REPLAY** and "a recorded session: no model is called".
- No keys, no network, no database.

---

## 5. The front end: `ui/`

### 5.1 Stack and layout of the code

React 19, Vite, TypeScript (strict). Dependencies: `react`, `react-dom`, `@fontsource-variable/inter`, `@fontsource-variable/jetbrains-mono` (bundled: the demo must work with no network). Dev: `vite`, `@vitejs/plugin-react`, `typescript`, `vitest`, `@testing-library/react`, `jsdom`. Nothing else without a reason written in the report. Icons are inline SVG.

```
ui/
  package.json            "packageManager": "npm@10.9.2"
  vite.config.ts          proxy /api -> http://127.0.0.1:8000
  index.html
  src/
    main.tsx  App.tsx
    api/      client.ts (fetch)   events.ts (EventSource)   types.ts (the shapes of section 4)
    state/    runReducer.ts (events -> RunState, pure)   useRun.ts   useAccount.ts
    components/
      Header.tsx  Pipeline.tsx  RunBox.tsx  Gears.tsx
      PortfolioCard.tsx  ProfileCard.tsx
      RequestCard.tsx  ActivityFeed.tsx  QuestionCard.tsx
      WorkOrderTable.tsx  RecommendationPanel.tsx
      ui/  Badge.tsx  Button.tsx  Card.tsx  Check.tsx  Meter.tsx  ScoreDial.tsx  Segmented.tsx  RiskSlider.tsx
    styles/   tokens.css  base.css  (component CSS beside each component, or in components.css)
    format.ts             money, numbers, times
    fixtures/             an event stream recorded from the server's replay of a real capture (generated, never typed by hand)
```

The repo's `.gitignore` is a Python one: it ignores folders named `lib`, `build`, `dist`, `parts`, `var`, `env`. Do not use those names for source folders. Add `ui/node_modules/` to `.gitignore`.

`npm` is not installed on this machine; `corepack npm@10.9.2 <command>` works (and `corepack pnpm@9.15.9`). The newest pnpm does not start under this corepack.

### 5.2 The page

Desktop first (1440 × 900), one screen, no page scroll: the page is the height of the window and the cards scroll inside.

```
┌ header: logo · PROMETHEUS / Agentic trading desk ··· PAPER TRADING pill · clock pill · theme · account chip ┐
├ DESK PIPELINE: six stages on a track, left to right ─────────────────────────── run box: gears, timer ──────┤
├──────────────────┬────────────────────────────────────────┬─────────────────────────────────────────────────┤
│ PORTFOLIO        │ YOUR REQUEST                           │ WORK ORDER                                      │
│ table + totals   │ text box · checkboxes · Mode · Go!     │ the stocks on the order, one row each           │
├──────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ PROFILE          │ ACTIVITY                               │ RECOMMENDATION                                  │
│ name age risk    │ the Account Manager's words, live;     │ the selected stock: thesis, cases, score,       │
│ funds            │ question cards pinned at the bottom    │ plan, the Critic's review, the order and fill   │
└──────────────────┴────────────────────────────────────────┴─────────────────────────────────────────────────┘
      me                         my request                              the desk's work
```

Columns `330px · minmax(420px, 1fr) · 520px`, gap 16, page padding 20. Under 1240 px the right column drops below as two cards side by side; under 860 px everything stacks and the page scrolls.

### 5.3 The components, against Jacob's list

**(1) Portfolio** `PortfolioCard`
- Columns: Symbol (the ticker, the company name under it), Shares, Value. One row per holding, then a **Cash** row.
- Foot: **Total invested**, **Funds available**, and **Account value** in bold.
- Above the table, a thin bar: invested against cash, with the two percentages.
- After a run changes something: the row gets a soft green wash and a badge (`+8 NEW`, `+8`, `−5`), and the hint under the table says what the cash was before. These stay until the next Go!.
- Empty: "No holdings yet. Add funds and ask the desk to buy." The Cash row and the totals still show.

**(4) Profile** `ProfileCard`
- Name (text), Age (number, 18 to 120), **Risk level**: a slider with three stops, Conservative · Moderate · Aggressive, on a green-to-red track with the chosen word in bold. **Funds**: one amount box with **Add** and **Remove**, and under it "Simulated money. No bank is connected."
- A change to name, age or risk saves on blur or on release (PATCH), with a small "Saved" tick; a refusal shows the server's message under the field and puts the old value back. Add and Remove clear the box and update the portfolio at once.
- First visit (no account remembered): the same card says **Open an account**, with Starting funds in place of the Funds row and one button, **Create account**; the account is opened in the mode the Mode switch shows. Go! is disabled until there is an account. The account number is kept in `localStorage`.
- The account chip in the header opens a small menu: the account number, *Switch account* (type a number) and *New account*.
- Disabled while a run is active, and always for a sample or replay account (with the reason as a tooltip).

**(2) (6) (7) (8) Request** `RequestCard`
- A text box, 600 characters at most (the back end's `MAX_TEXT`), with a counter. Its starting text is exactly:
  `Buy an even mixture of 3 most recommended stocks in the pizza restaurants using $3000 of my available money`
- **Clear** empties the box and puts the cursor in it.
- Three small checkboxes, all in plain sight (no fold): **Strict recommendations** (off), **Neutral recommendations as passes** (on) and **Refresh market data first (paid)** (**on**; Jacob, 2026-11-15), with the tooltip "Fetches fresh bars, news and filings with the paid APIs and writes the data files. Untick to use the stored data." In replay mode the third is shown unticked and disabled ("a recorded session: nothing is fetched"). `POST /api/runs` keeps its own default of no fetch when the field is missing: only the page turns it on.
- **Mode** (Jacob, 2026-11-15): a slider switch on the right of the card, above Clear and Go!: `MODE  Ask (●━) Auto`. **Off is Ask** (the default: you confirm what was understood and approve each trade; nothing trades without your yes). **On is Auto** (the desk trades what the Critic approves without asking; it still asks when a review wants a person to look). The word of the chosen side is bold; the track is blue (`--info`) when on. See the mockup.
  - It IS the account's trade mode, and it is the only control for it (the Profile card has none). Flipping it saves at once through the **Account Manager**: `PATCH /api/accounts/{id}` `{mode}`, which calls `AccountManager.users.set_trade_mode`. The switch moves first; if the save is refused it moves back and the server's message shows under it. The account chip in the header shows the mode.
  - The run reads the mode from the account, as the session does (the profile the Account Manager holds: `decide_trades`, `decide_direct`, and whether `confirm` asks). Go! also sends `mode`, so a run started a moment after a flip cannot use the old one (4.1).
  - With no account yet the switch still works and holds its own value (Ask); Create account sends it. Disabled while a run is active, for a sample account, and in replay mode (where it shows the capture's mode). `role="switch"`, `aria-checked`, Space and Enter flip it.
  - Tooltip: Ask: "Ask: nothing trades without your yes." Auto: "Auto: the desk trades what the Critic approves without asking."
- **Go!**: the one ember button on the page. Enabled when there is text, an account, and no run. It posts `{account_id, prompt, strict, neutral, fetch, mode}`. While a run is active it shows the small gear and "Running…", and a quiet **Stop** appears beside it. `Ctrl+Enter` in the box is Go!.
- What the checkboxes do (`settings.py`), for one run only; nothing is saved to the account and `.env` is never read or written by this code:

| Checkbox | Run setting |
|---|---|
| Neutral as passes: on | `ALLOW_NEUTRAL_RECOMMENDATIONS=true` in the server's own environment for the length of the run, put back afterwards. The Account Manager already reads it through `rag.config.setting`, environment first. |
| Neutral as passes: off | the same variable set to `false`. |
| Strict: off | Nothing else changes: the neutral box says whether a neutral recommendation counts as approved. (Design choices entry 104: there is no automatic-trading mark any more; Auto mode trades every recommendation the Critic approves whatever the checkboxes say.) |
| Strict: on | **Only what the Critic approves trades: a neutral recommendation never passes**, whatever the other box says; the UI shows that box unticked and disabled with the tooltip "Strict: only what the Critic approves trades". |

  **Confirmed by Jacob, 2026-11-15, in his words: *"The Strict Recommendations is only allow Approved recommendations (no neutral allowed)"* and *"ALLOW_NEUTRAL_RECOMMENDATIONS=true in .env follows that matching checkbox"*. The neutral box decides the setting for the run; the value in `.env` is what a run without the page uses, and the page never edits it. Strict means only what the Critic APPROVES may trade, and a neutral recommendation never passes. All of it is one function, `settings.run_settings(strict, neutral) -> RunSettings`, which returns only whether a neutral recommendation counts as approved (`allow_neutral`), so that changing it is a few lines.

**(3) (9) Pipeline** `Pipeline`, `RunBox`, `Gears`
- Six nodes on a track, left to right, each with its stage name and under it a dot in its agent's colour, the agent's name and the seconds.
- States (see the mockup's "every stage state"): **not reached** (hollow, grey), **busy** (an ember ring turning round the node, a soft glow, the label in ember, the seconds counting up), **waiting for you** (yellow, pulsing, a "?"), **done** (solid green, a tick, the final seconds), **skipped** (dashed, the label struck through), **stopped here** (red, a cross).
- The track between two nodes is green once the left one is done, green-to-ember into a busy node, dashed around a skipped one.
- The run box on the right: two gears turning against each other while a run is active (the indeterminate indicator), the elapsed time, and one line of what is happening ("Critic is reviewing 3 recommendations", "Waiting for your answer"). Idle: a still gear and "Ready". Finished: the headline of `run_finished` and the total time. With `prefers-reduced-motion` the gears and the ring do not turn; the states still differ by colour and icon.
- When `activity` events arrive, a quiet line under the timer: "7 model calls · 12.4k tokens".

**(5) Activity** `ActivityFeed`, `QuestionCard`
- The output field. One line per `log`: the time, the agent with its coloured dot, the text. Monospace. Report blocks keep their line breaks. `good`, `warn` and `error` lines take the green, yellow and red.
- It follows the newest line unless the user has scrolled up; then a "Jump to latest" pill appears.
- An open `question` is a card pinned to the bottom of the panel, with a yellow border, and the panel's header says "Waiting for you":
  - **confirm**: "I understood: … Go ahead?" with *Yes, go ahead* and *No, stop*.
  - **clarify**: the question, the facts in a monospace block, a text box and *Send* (Enter sends). "question 1 of 3".
  - **approve**: "Buy 8 shares of XOM, about $1,364.72 of $15,000.00 cash", the Critic's verdict, *Approve* and *Deny*, and *See the scorecard* which unfolds the card's text. "1 of 2". The first card of a proposal lists the whole proposal above it.
  - **missing_company** (a stock the Data Clerk did not find; Ask mode only, entry 102): the badge "ACCOUNT MANAGER ASKS" ("1 OF 2" when there are several), the Account Manager's sentence (its model's wording, checked by code, or the code's own) with the stock as the user typed it in bold ("**Chick-fil-a** was not found because it is privately held, or not known to be listed; is it OK to continue without it ...?"), *OK, continue without it* and *Stop the order*, and under them, in a monospace block, what stays on the order ("Still on your order: F: Ford Motor Co (F) ... Google: Alphabet Inc. (GOOGL), the listed owner of what you typed"). No answer means the order is not placed. The work order table then shows the stocks that were kept.
  - After an answer the card becomes one line in the feed ("You: approved XOM").
- Before the first run: "Nothing yet. The Account Manager will report here as the desk works."

**(10) Work order** `WorkOrderTable`
- Header: the request id, and the order's status as a badge.
- Columns: # · Stock (ticker, company under it) · Advisor (score, stance badge) · Critic (`APPROVE 86 · B` or `DENY 49 · F`) · Order (`Buy 8`, the dollars under it) · Trade (`FILLED`, `PLACING…`, `REJECTED`, `NO TRADE`, `PENDING YOU`).
- Rows fill in as the desk goes: found, then scored, then reviewed, then decided, then traded. A `scored` row is dimmed and its last three columns are one cell with the reason.
- Clicking a row selects it (an ember bar on its left) and shows it in the Recommendation panel. Arrow keys move the selection. When recommendations first arrive, the first row is selected unless the user already chose one.

**(11) Recommendation** `RecommendationPanel`
- For the selected row, the sections of 4.4.1 in that order, each only when it has data: header with the score dial; Thesis; Bull case and Bear case side by side; Evidence; How the score was built (four bars, 40 / 30 / 20 / 10); Key numbers; Trade plan (six small tiles; the note "prices and dates only: the Advisor sets no size"); Critic review (verdict, score, grade, six bars, blockers in red, warnings in yellow); Order and fill; **Raw record** (a fold; fetched when opened, shown as formatted JSON).
- Before the Advisor has answered: "Pick a stock" / "The Advisor has not answered yet."

**Header** `Header`
- The mark and the name. Pills: **PAPER TRADING · SIMULATED BROKER** always; the clock note when `clock_note()` is not empty; **REPLAY** in replay mode. The theme button (dark, light; first visit follows the system; the choice is remembered). The account chip.
- If `health.book` is not `ok`, a red bar under the header with the message and "run scripts/setup_postgres.sh".

### 5.4 State

One reducer, `runReducer(state, event) -> state`, pure and tested: the stages, the log, the open question, the work order, the portfolio, the activity totals, the result. `useRun` opens the `EventSource` and dispatches; on a reconnect it passes the last `seq`. On page load, if `health.active_run` is set, it attaches to that run and everything redraws from the replayed events.

The account (`useAccount`) is separate: loaded from `GET /api/accounts/{id}`; a `portfolio` event of a run replaces its portfolio.

### 5.5 The look

Take the tokens and the component CSS from the mockup. In short:

- **Feel:** a trading terminal that a person, not a trader, can read. Deep ink navy, quiet cards with hairline borders, numbers in a monospace face with tabular figures, one warm colour for action.
- **Ember** (`--brand`, orange to amber) is for exactly two things: the Go! button and whatever is busy right now. Nothing else is orange.
- **Green** is good money and done. **Red** is a denial, a rejection, a loss. **Yellow** is only "this waits for you". **Blue** is information. **Grey** is neutral.
- Each agent has one hue (`--am`, `--clerk`, `--advisor`, `--critic`, `--trader`) used only as the small dot beside its name and for the Critic's bars.
- Type: Inter 13 px for text, JetBrains Mono for every ticker, number, time and log line. Card titles are 11 px capitals, letter-spaced.
- Radius 12 for cards, 8 for controls, pills for badges. A 4 px spacing grid. Cards have 16 px padding.
- Motion is short and means something: the busy ring, the gears, a 600 ms green wash on a row that changed, the blinking cursor on the last busy line. Nothing else moves.
- Both themes must pass WCAG AA for text.

### 5.6 Accessibility and small things

Every control has a label. The pipeline is an ordered list with `aria-current` on the busy stage. The feed is `role="log"`; a question moves focus to its first button. Everything works from the keyboard. Money is `Intl.NumberFormat("en-US", {style: "currency", currency: "USD"})`. Times are local, `HH:MM:SS`. Long company names end in an ellipsis with the full name as the title.

---

## 6. Running it

`./run_ui.sh` at the repo root (a thin script, like `test_prometheus.sh`):

```
./run_ui.sh                      build the UI if ui/dist is missing or older than ui/src, start the server on http://127.0.0.1:8000
./run_ui.sh --replay             the same, replaying tests/fixtures/session_real/bob_jones_oil.json (free: no keys, no database)
./run_ui.sh --replay FILE        another capture
./run_ui.sh --dev                the API on 8000 and Vite on 5173 with hot reload
```

It uses `~/miniconda3/envs/cmu/bin/python` and `corepack npm@10.9.2`. Starting the server costs nothing. **Pressing Go! in live mode is a paid run** (OpenRouter and Finnhub) that writes the UI's own account in Postgres; and because "Refresh market data first (paid)" is on by default, it also calls the paid data APIs and writes the data files unless that box is unticked. The script prints that in two lines when it starts in live mode.

---

## 7. Observability

Three layers, each optional above the first:

1. **The stepper and the feed** (always on): LangGraph's own `tasks` and `values` stream (D4). Nothing to configure.
2. **Model activity** (on by default, no account needed): `tracing.ActivityHandler`, a LangChain `BaseCallbackHandler` passed in `config={"callbacks": [...]}` to `graph.stream`. On each chat-model end it adds the call and its tokens to the run's totals and emits `activity`. Check with the first live run that callbacks given to the graph do reach the model calls the agents make inside their nodes; if they do not, leave the counter out of the UI and say so in the report. Do not change an agent to make it work.
3. **Langfuse** (off unless configured): when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are in the environment, `tracing.callbacks(run)` also returns Langfuse's LangChain callback handler, with the run id as the trace's session and the tags `prometheus-ui`, the trade mode, `strict`, `neutral`. When the keys are missing or the package is not installed it returns none and logs one line at start-up. The health route says `tracing: "langfuse" | "off"` and the header shows a small "Traced" dot when on. **Look up the current Langfuse Python SDK before writing the import**; its module paths have changed between major versions. Read the keys with `rag.config.setting`, never by opening `.env`.
   - Add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` to `.env.example` with a comment (Langfuse Cloud has a free tier; it can also be self-hosted with Docker). Jacob adds the real ones to `.env` himself.
   - With tracing on, the prompts, the model replies and the account's figures of a run are sent to that Langfuse server. Say so in `HowToRun.txt`.
   - **LangSmith instead** needs no code: `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in the environment and LangGraph traces itself. One line in `HowToRun.txt`.

---

## 8. Tests and proof

Rules of the repo that bind here: real data only; no stand-in agent or model in a test; no test that repeats another's boundary value; leave no `__pycache__` or `.pytest_cache`.

**Python** (`tests/test_web_ui.py`, and `tests/test_web_views.py` if it reads better):
1. `views.work_order_view` on the real captures: `bob_jones_oil.json` (three recommendations: XOM approved 86 B and filled, APA denied 49, EOG approved 76 and filled; the scored rest), `pam_smith_neutral.json` (the neutral rows and their `origin`), `bill_gates_direct_order_funded.json` (a direct row; no Advisor or Critic section).
2. The replay through the real app (`fastapi.testclient.TestClient`, `--replay-speed 0`): the event types in order, the six stages ending `done`, the portfolio before ($15,000.00, no holdings) and after ($12,367.41, 8 XOM and 8 EOG), `run_finished` `traded`. A direct-order capture: `gather`, `recommend`, `critique` are `skipped`.
3. `RunManager` questions with no agent anywhere: a thread asks, the answer route answers; a timeout returns `None`; stop closes it; a second answer is refused.
4. `settings.run_settings` for the four combinations, with the marks read from `configs/critic_v1.json`, and the environment variable restored after a run that raises.
5. The account routes on a **throwaway** account in the local Postgres, created and deleted by the test as `tests/test_trader_handshake.py` does, skipped when the book is not reachable: create, read, change name, age, risk and mode, add funds, remove funds, remove more than the cash (422), a write to a sample account (409).
6. `AccountStore.remove_funds` beside the existing add-funds tests.
7. A test that reads the source of `prometheus/application/web/` and fails if it opens `.env` or imports an agent's scoring code.

**Front end** (`vitest`): the reducer over the recorded real event stream (stages, rows, portfolio changes, the result); the request card (the default text, Clear, the checkbox defaults, strict disables neutral, the payload of Go!); the question card for each kind.

**The whole suite stays green:** `~/miniconda3/envs/cmu/bin/python -m pytest -q --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo`.

**Seen with eyes.** Headless Chrome is installed: `google-chrome --headless=new --no-sandbox --hide-scrollbars --window-size=1440,900 --screenshot=out.png URL`. Against the replay server, take and look at: idle with the default prompt; a stage busy in the middle of a run; a question card open; finished with a row selected and the recommendation showing; the same in the light theme; 1100 px and 800 px wide. Put them beside the mockup and fix what differs. Keep the final set in `docs/ui_screenshots/`.

**One real run** (allowed by Jacob's standing yes for paid runs that test the desk, `AGENTS.md`; no fetch, a throwaway account that is deleted afterwards): a direct order such as "Buy 2 shares of MSFT" in Auto mode through the live server, driven with `curl` or a short script. It proves the `tasks` stream on the real graph, the Postgres booking and the portfolio events. If that passes, **one** run of the default pizza prompt in Auto mode. No more than three paid runs in all. Report what each cost in time and what it showed. If a key is missing or the market clock refuses the trade, report that and stop; do not work around it.

---

## 9. Documents to update in the same change

- `docs/design_choices.md`: **a new entry at the end** (the next number after 98): the UI. Problem, decision, alternatives (Streamlit or Gradio; WebSocket; a hook inside `desk.py`; polling), consequences. A second entry for `remove_funds` if it reads better apart. Do not edit an old entry.
- `README.md` (the layout: `ui/`, `application/web/`, `run_ui.sh`), `HowToRun.txt` (a section: how to run the UI, replay, dev, tracing), `AGENTS.md` (where things are; and under "Never without asking": pressing Go! in live mode), `docs/SESSION_NOTES.md` (what is open).
- `requirements.txt`: `fastapi` (and `langfuse` only if it is really imported), added lines only; no existing version is touched.
- `.env.example`: the three Langfuse names. `.gitignore`: `ui/node_modules/`.

---

## 10. Order of work

1. **Views and replay** (free): `views.py` on the real captures, `runs.py`, `replay.py`, `app.py`, their tests.
2. **The UI against the replay**: tokens, layout, every component, the reducer, screenshots beside the mockup.
3. **Accounts**: the routes, `remove_funds`, the Profile and Portfolio cards against the local Postgres on a throwaway account.
4. **The live run**: `desk_run.py`, questions, settings, the portfolio polling; the one real run.
5. **Observability**: the activity counter; Langfuse behind its keys.
6. **Documents**, the full test suite, the final screenshots.

**Done means:** every item of Jacob's list (1 to 11) works in replay mode with no keys; a live run goes end to end from the browser; the whole test suite passes; the screenshots match the mockup; the documents are updated; nothing is committed (Jacob reviews first).

---

## 11. Rules that are easy to break here

- Never read, print or edit `.env`. Never touch `data/historical/` or `data/processed/`. Delete no file of the repo. Change no existing version in `requirements.txt`.
- No risk management anywhere: no size limit, no liquidity rule. The UI adds no rule about what may trade.
- Keep the model out of anything code can check. The UI shows what the agents wrote; it scores nothing and words nothing.
- The Trader is simulated. No control in the UI may suggest that a real order is sent; the PAPER TRADING pill is always there.
- Do not write to the three sample accounts.
- No commit and no push.

---

## 12. Open, and next

1. ~~"Strict recommendations" waits for Jacob's word~~ Confirmed (5.3). (Settled 2026-11-15, entry 104: there is no automatic-trading mark: Auto mode trades everything the Critic approves without a question whether Strict is on or off, and Strict is only the neutral rule.)
2. Live market value in the portfolio (a Refresh prices button through `AccountManager.request_prices`; Finnhub quotes cost calls).
3. Run history (persisting work orders is already an open item in `docs/SESSION_NOTES.md`).
4. Showing the agents' inner hand-offs (the Critic asking the Data Clerk again; the Trader's handshake with the Account Manager on every trade) as their own marks in the pipeline. The Langfuse trace has them; the stepper does not.
5. Limit orders, stop-loss orders and the monitoring mode are back-end TODOs (`design/week6_updated_design.md`); the UI has no control for them.
