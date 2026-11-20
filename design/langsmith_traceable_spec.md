# LangSmith `@traceable` spans for the agents (entry 115)

Jacob, 2026-11-20: *"add @traceable to the handshake and sizing now. Add it to everywhere where LLM and agents are used and tracing is beneficial for generating a trace."*

Entry 114 turned LangSmith tracing on (`prometheus/common/langsmith_tracing.py`, agent `prometheus` / `local`). LangGraph and LangChain already trace every desk node, every chat-model call and the Data Clerk's tools: the first real session trace held one root run (`LangGraph`), 68 runs and 14 model calls. What a trace does NOT show is the plain-Python work inside a node: which agent asked which other agent for what (the Critic asking the Data Clerk again), the Advisor's and the Critic's steps, the sizing, and above all the account handshake (approve, fill, book), which is the "data integrity" metric of the week 6 submission. This spec adds named spans for that work. It changes NO behaviour.

## 1. Rules

| # | Rule |
|---|------|
| R1 | One helper, `prometheus/common/trace.py`, is the ONLY module that imports `langsmith` for spans. Every other module imports `traced` from it. (The Trader's import tests forbid some modules; `prometheus.common.trace` is not one of them. Do NOT decorate `agents/trader/record.py` or `agents/trader/config.py`: their import whitelist test allows no new module.) |
| R2 | `traced(name, run_type="chain")` returns a decorator built on `langsmith.traceable`, with `process_inputs` / `process_outputs` from R4. |
| R3 | NEST ONLY, NEVER A ROOT. The wrapper calls the plain function when `langsmith.run_helpers.get_current_run_tree()` is None (no trace is open: a test, a CLI step before the graph starts, `--account`), and the traced one otherwise. Checked on 2026-11-20: inside a LangGraph node the current run tree is the node's run; outside the graph it is None. So a decorated function never makes a root trace of its own, the online judge (filter "is trace") keeps judging only the desk's root run, and nothing at all is sent when tracing is off. |
| R4 | SMALL, SAFE PAYLOADS. Inputs and outputs go through `trace.summarize(value)`: a `WorkOrder` becomes `{"id", "kind"/request summary, "symbols", counts of companies, articles, recommendations, reviews, trade_decisions, trades}` (never its data: the articles, filings and press releases are already in the node's own state); a pydantic model is `model_dump()` (Python mode) and then summarized; a `Decimal` stays a Decimal (the SDK writes it); dicts and lists are summarized recursively with a cap (lists longer than 20 items keep 20 and add `"…": n_more`; strings longer than 2,000 characters are cut with a marker). Anything that is not data (a model, a clerk, a store, a connection, an approver, a callable, a ledger, an `out`/`say` writer) becomes its type name only (`"<AccountStore>"`): never its `repr`, which could hold a DSN or a key. `self` is dropped by the SDK already. |
| R5 | No behaviour change. The decorated function returns the same value and raises the same exception; no argument is changed; no generator, coroutine or LangGraph node function is decorated (the nodes are traced by LangGraph already); no `float(...)`, `round(` or float literal is added to an account-side module (`tests/test_no_float_in_the_account.py`). |
| R6 | Names say what the step is, `agent.step` in snake case (`account_manager.approve_trade`, `critic.review_recommendations`). In the Advisor's and the Critic's modules a span name and its summary must not contain the words that their source-reading tests forbid (`portfolio`, `held`, `holding`, `cost_basis`, `unrealized`, … : read `tests/test_advisor_no_size.py` and `tests/test_critic.py` before naming). |
| R7 | `run_type`: `"chain"` for an agent step, `"tool"` for a Data Clerk request that fetches or looks up (`request_quotes`, `request_bars`, `request_symbols`, `request_verification`), `"parser"` is not used, `"llm"` is NEVER used (the model calls are already `llm` runs from LangChain; a second `llm` run would double the token counts). |

## 2. Where (module by module)

Decorate the PUBLIC agent methods and the code-only steps a person would look for in a trace. Skip trivial accessors (`config`, `model`, `clerk`, `ledger`, `profile`, `users`, `connect`) and the account book's CRUD that the session calls before the graph (R3 makes them no-ops there anyway, so leave them out).

- **Account Manager** (`agents/account_manager/`): `AccountManager.understand_request`, `decide_trades`, `decide_direct`, `clarify_sizing`, `name_orders`, `name_the_stocks`, `request_symbols`, `raise_funds`, `raise_funds_many`, and the Trader's doors `trading_portfolio`, `approve_trade`, `preview_trade`, `record_fill`, `apply_earlier_fills`, `notify_trades`; module functions `sizing.size_purchases`, `sizing.funds_check`, `handshake.decide_approval`, `decisions.decide_trades`, `neutral.purchases_for`, `missing.ask_about_each`, `clarify.run_loop`, `direct.decide_direct`. The handshake spans must show, in their outputs, the approved shares and dollars, the cash before and after (`record_fill`'s `Booking`), and the refusal reason when there is one: that is what the summary of their return value gives without extra code, check it on a real call.
- **Data Clerk** (`agents/data_clerk/data_clerk_agent.py`): `DataClerk.request_data`, `request_quotes`, `request_symbols`, `request_bars`, `request_metrics`, `request_verification`, `find_companies`, `refresh_data`.
- **Advisor** (`agents/advisor/`): `Advisor.generate_recommendations`, `recommend.generate`, and in `calls.py` `ask_json` and `ask_object` named by their `run["stage"]` (one span per model step: `advisor.plan`, `advisor.thesis`, the Critic's evidence call) — use `traced` with a name function or set the span name from the stage inside the wrapper; keep `invoke_with_limit` undecorated (it runs in its own thread with the copied context).
- **Critic** (`agents/critic/`): `Critic.review_recommendations`, `review.review_order`.
- **Trader** (`agents/trader/`): `Trader.execute_trades`, `execute.execute`, `execute.fill_order`; `work_order.verify_fill` and `verify_trades` in `common/work_order.py` (the output checks of each fill and of the whole run).
- **RaiseFunds** (`reasoning/raise_funds.py`): `search` (the tree), `choose` (one model expansion), `score_plan` (the evaluator), `answer`.
- **Web and runner:** nothing. The root stays the LangGraph run.

If a listed function turns out to be a generator, a graph node, called thousands of times per run, or decorated in a way that breaks a source-reading test, leave it out and say so in the report.

## 3. Tests (few, real)

`tests/test_trace.py`:
1. Off, no trace open: a decorated function returns the same value and raises the same exception, and `summarize` is never called (patch-free: count with a real function that records its calls).
2. R3 with tracing ON and the endpoint at a dead local address (`LANGSMITH_ENDPOINT=http://127.0.0.1:9`, a key `"x"`; monkeypatched env, `configure()` NOT called, nothing reaches a network): inside a real one-node `StateGraph` the decorated function runs as a child (its run tree's parent is the node), outside the graph it creates no run (`get_current_run_tree()` stays None, the function still runs).
3. `summarize` on a REAL work order from `tests/fixtures/session_real/` (`auto_even_split_decimal.json` or another capture with trades): no article text, the counts and symbols are right, Decimals stay Decimals, a store-like object (use the real `AccountStore` class without connecting, or any real non-data object of the project) becomes its type name.

Keep `tests/test_langsmith_tracing.py` as it is. The whole suite must pass with the same pre-existing failures only (since Postgres was recreated on 2026-11-20 there should be none apart from the deselected demo test; report every failure with its cause).

## 4. Docs

New entry 115 at the end of `docs/design_choices.md` (problem, decision, alternatives: decorate everything / the nodes / only the handshake; consequences: payload size, what is sent, R3). One paragraph under LANGSMITH TRACING in `HowToRun.txt`. A line in `AGENTS.md`? No: Jacob edits AGENTS.md.

## 5. Order of work

1. `common/trace.py` and its tests. 2. Account Manager and handshake. 3. Data Clerk, Advisor, Critic, Trader, RaiseFunds. 4. Full suite. 5. Docs. 6. Report: the list of decorated functions, anything left out and why, test results.
