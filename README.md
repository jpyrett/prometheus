# Prometheus

Self-learning multi-agent trading bot — CMU Agentic AI Capstone (Fall 2026).

## Layout

```
capstone/
├── design/                 # Capstone design docs (keep as source of truth)
├── labs/                   # Course notebooks and lab scripts — port snippets from here
│   └── module3_rag/        # RAG guide, advisor_with_without.py; other/ = one-off lab scripts
├── prometheus/             # Application source
│   ├── application/        # Orchestration / use-case entrypoints
│   │   ├── harness/        # How you run & drive the multi-agent loop
│   │   └── web/            # The web UI's bridge (FastAPI + Server-Sent Events): app.py routes, desk_run.py the live run, replay.py a recorded session, views.py what the page shows, runs/settings/tracing/accounts/emitter (design choices entry 99; design/ui_v1_spec.md)
│   ├── agents/             # Data Clerk, Advisor, Critic, Account Manager, Trader
│   │   ├── data_clerk/     # data_clerk_agent.py (DataClerk: the LLM agent; `request_metrics` calculates numbers), data_toolbox.py (DataToolbox: plain Python, no LLM), bars.py / technicals.py / fundamentals.py / metrics_catalog.py (the numbers)
│   │   ├── critic/         # critic.py (Critic: approve or deny each recommendation), review.py (its nine steps), integrity/recheck/evidence/timing/risk/value/scoring/report.py, config.py, record.py (design choices entries 56 to 59)
│   │   ├── trader/         # trader.py (Trader: executes the permitted trades against a SIMULATED broker, paper or backtest; a model with four guarded tools, then a code sweep), execute.py (its ten steps), tickets/broker/simulator/record/config/report.py, replay.py (the free replay command); design choices entries 73 to 75, design/trader_v1_final_spec.md
│   │   ├── advisor/        # advisor.py (v0 `advise`: a score from checked quotes; v1 `generate_recommendations`: ranked, audited recommendations; recommend.py is its twelve steps, config/families/blend/gates/quality/ranking/trade_plan/events/thesis/record its parts; guide: design/week5_advisor_implementation.md)
│   │   └── account_manager/ # account_manager.py (AccountManager: the LLM agent behind RaiseFunds, a Tree of Thought; holds the user's profile; the Postgres account book: funds, holdings, risk, gain and counts), decisions.py (the Critic's verdicts become trade decisions: Auto trades what is approved and asks nothing, Ask asks; the denied are returned to the user, entry 104), sizing.py (how the user's money is divided: the cash and the budget, round-robin in Auto, an even split or the largest order; NO risk management and no size limit: the only check is that the funds cover the order, entry 91; every threshold in configs/account_manager_v1.json), neutral.py (a neutral recommendation is approved with ALLOW_NEUTRAL_RECOMMENDATIONS, entry 91), missing.py (a stock the Data Clerk did not find: dropped in Auto mode, asked about in Ask mode with a question the model words and code checks, entry 102)
│   ├── mcp/                # Local FastMCP servers (tools / resources / prompts)
│   ├── rag/                # Embeddings, vector store, retrieval
│   ├── reasoning/          # raise_funds.py: the Tree of Thought for what to sell to buy a stock (a LangGraph; design/week4_tot_guide.md)
│   ├── prompts/            # Prompt templates (per agent / use case)
│   ├── langchain/          # LangChain pieces you port from labs
│   ├── langgraph/          # desk.py: the top-level graph, account_manager -> data_clerk -> advisor (-> critic -> account_manager_decision with --critic; -> account_manager_funds with --raise-funds; -> trader with --trade; --account uses a saved account)
│   ├── tools/              # Local Python tools (calc, sentiment, search, …)
│   ├── models/             # OpenRouter client wiring + domain JSON schemas
│   ├── memory/             # Short / long-term memory (later modules)
│   ├── evals/              # Hallucination, brittleness, cost/quality/latency
│   ├── persistence/        # accounts.py: the Postgres account book (cash, holdings, trades, recommendation counts)
│   └── common/             # Shared non-agent helpers (work_order.py: the work order every agent reads; money.py: the doors, the types and the ONE JSON encoder of the account's `decimal.Decimal` money and prices (entry 112); shares.py: the one door of a share quantity, a Decimal with at most 6 decimals (a whole one has none), and a bool or a string is refused; the pot of a run and its orders in dollars are in work_order.py (`decision_pot`, `decision_dollars`); user_profile.py: risk tolerance and Ask/Auto; user_account.py: the holder's name, account number and screen fields; market_calendar.py: trading days and the trading window)
├── configs/                # Non-code config (portfolios/ and profiles/jacob_pyrett.json: the static sample portfolio and user profile; advisor_v1.json, critic_v1.json, account_manager_v1.json (the sizing), trader_v1.json: every threshold; market_calendar.json)
├── sql/                    # prometheus_accounts.sql and seed_test_users.sql (the account book), pg_hba.conf
├── docker-compose.yml      # Postgres 18 for the account book, bound to 127.0.0.1 (scripts/setup_postgres.sh)
├── data/
│   ├── samples/            # Small explainability samples for demos
│   ├── raw/                # Downloaded / API dumps (local)
│   └── processed/          # Cleaned data, vector indexes (local)
├── scripts/                # Thin runners (raise_funds_offline_demo.py: RaiseFunds with invented quotes and a scripted LLM, no cost)
├── tests/                  # pytest (FastMCP tools + later app tests)
├── main.py                 # App entry: LocalLLM + FastMCP smoke / HITL
├── ui/                     # The web page (React 19 + Vite + TypeScript, no UI kit): `cd ui && corepack npm@10.9.2 run build`; tests with vitest; design/ui_v1_spec.md, the look: design/ui_v1_mockup.html, screenshots: docs/ui_screenshots/
├── run_ui.sh               # `./run_ui.sh --replay` the page over a recorded session (free); `./run_ui.sh` live (Go! is a PAID run); `--dev` for Vite with hot reload
├── test_prometheus.sh      # Main script: prompts for symbol / term / source / from_days_ago, runs the desk (a LangGraph passing one work order); pytest stays in tests/
├── test_prometheus.sh      # the desk by parameters; `--session`: a user session end to end (create a user, add money, a request in your own words, the real agents on the LangGraph desk, the portfolio before and after: entry 78)
├── test_recommendations.sh # Recommendations end to end: a symbol, a list, a company, a brand or a phrase ("Pizza", 'MCD, YUM, "Burger King"') to a ranked report; Advisor on gpt-oss-120b (design choices entry 53); with the Critic, and with --trade the Trader (simulated)
├── AGENTS.md               # Instructions for Cursor agents: how to run, where things are, what never to run unasked
├── HowToRun.txt            # Copy-paste shell commands
├── pytest.ini              # asyncio_mode=auto for FastMCP Client tests
├── requirements.txt        # Curated runtime deps (reproducible install)
├── .env.example            # Safe template (committed)
└── .env                    # Your keys (gitignored)
```

Keep layers thin: put new code in the folder that matches its job. Prefer porting from `labs/` into one of these folders rather than growing notebook-only logic.

## Working with Cursor agents

Open the project in Cursor (editor or Agents Window; in the editor run **Open Agents Window** from the Command Palette). Use a **local** agent: it runs in this checkout, so it sees the `cmu` conda env, `.env` and `data/`. A worktree or cloud agent starts from a fresh git checkout without the git-ignored `.env` and `data/` (see [HowToRun.txt](HowToRun.txt), section 0).

**[AGENTS.md](AGENTS.md)** is read by every agent chat automatically. It holds the facts an agent would otherwise have to rediscover: the Python to use, the one test that fails on purpose, where each part lives, and what must never be run without asking (anything that calls a paid API or writes the news files, and anything touching `.env`). Keep it short; when you rename or move a thing, update it with the other docs. Cursor does not read a file called `Cursor.md`.

## How to run

See **[HowToRun.txt](HowToRun.txt)** for the full command list. Short version:

```bash
conda activate cmu
pip install -r requirements.txt   # once / when deps change
cp .env.example .env              # once; set LOCAL_MODEL / keys

python main.py --skip-llm         # MCP only (fast)
python main.py                    # elicit symbol → fetch_quote → LocalLLM
python main.py --symbol TEST      # demo quote price 999.99; skip elicitation
./test_prometheus.sh --source press_releases --company "Apple Inc" --yes   # press releases (the company's own announcements) by company name; or --symbol AAPL
./test_prometheus.sh --source filings --company "Apple Inc" --yes         # the company's SEC filings (10-K, 8-K, 10-Q ...), asked for the same way; or --symbol AAPL
./test_prometheus.sh              # prompts for symbol, term, source (news, press_releases or filings) and from_days_ago (7; press releases and filings 365), then runs the desk: account_manager -> data_clerk -> advisor
./test_prometheus.sh --legacy     # the old demo (LocalLLM + elicitation)

pytest tests/fastmcp_tools.py -q -k 'not forced_failure'
fastmcp inspect prometheus/mcp/server.py
```

Notes:

- `LOCAL_MODEL` in `.env` (default `Qwen/Qwen3-4B`) loads via `prometheus.models.llm.LocalLLM`.
- Leading `$` on symbols is fine (`$AAPL` → `AAPL`). Symbol `TEST` is a temporary hardcoded quote fixture.
- Do not expect `python prometheus/mcp/server.py` to stay up alone — stdio MCP needs a client; `main.py` uses in-process `connect_client()`.

## Conda + reproducible deps

Cursor is pointed at the course env (`cmu`):

```
/home/jpyrett/miniconda3/envs/cmu/bin/python
```

Activate in a terminal:

```bash
conda activate cmu
```

**Rule:** when you install a package for Prometheus, also add it to `requirements.txt` (direct deps only — never dump a full `pip freeze`).

```bash
conda activate cmu
pip install -r requirements.txt
# torch is platform-specific — install separately (see comments in requirements.txt)
```

Same on DGX Spark: recreate or sync the `cmu` env, install torch/CUDA for that box first, then `pip install -r requirements.txt`.

Uncomment LangChain / LangGraph / OpenAI lines in `requirements.txt` when you start using those labs.

## Testing (pytest)

We use **pytest** (+ **pytest-asyncio**) for unit tests and in-process FastMCP `Client(server)` tests. Config lives in `pytest.ini` (`asyncio_mode = auto`).

```bash
conda activate cmu
pip install -r requirements.txt   # includes pytest / pytest-asyncio
pytest                            # from the repo root
```

FastMCP tools: `tests/fastmcp_tools.py` (imports `prometheus.mcp.server`).

### FastMCP client connection (Part 4) + caching (quotes)

Prefer `connect_client()` from **`prometheus/mcp/client.py`** (host-side; `server.py` does not import it):

- **`mode="legacy"`** — Client setting (not `server.run()`). Needed for elicitation (HITL); use with in-process *or* stdio hosts. Sampling and roots are deprecated — prefer host LLMs and tool args / resource URIs / config.
- **`cache=False`** — **Never set `cache=True` or will break the FastMCP `market://{symbol}/quote` resource.**
- Server: `cache_ttl=None` in `server.py`. Stdio: `python prometheus/mcp/server.py`; host connects via `client.py`.

## Environment variables

```bash
cp .env.example .env   # already created; edit .env with real keys
```

For local `main.py` runs: set `LOCAL_MODEL` in `.env` (see `.env.example`).  
For later OpenRouter agents: set `OPENROUTER_API_KEY` in `.env`.

## Two “harnesses” (do not confuse them)

### 1) Cursor + OpenRouter — coding / agent harness (IDE)

This is how *you* develop with Cursor calling OpenRouter models while you write Prometheus by hand.

1. OpenRouter API key: https://openrouter.ai/keys  
2. Cursor → Settings → Models → API Keys  
3. Enable **OpenAI API Key**, paste the OpenRouter key (`sk-or-…`)  
4. Enable **Override OpenAI Base URL** and set:

   `https://openrouter.ai/api/v1/cursor`

5. Add model IDs you want in the model picker (OpenRouter model slugs).

That routes Cursor’s agent/chat through OpenRouter. It is **not** the trading bot runtime.

### 2) Prometheus application harness — running the bot

**Today:** run `python main.py` (see [HowToRun.txt](HowToRun.txt)). That wires:

- `prometheus/models/llm.py` — local HF CausalLM (`LOCAL_MODEL`)
- `prometheus/application/host.py` — elicit → `fetch_quote` → prompt → LLM
- `prometheus/mcp/` — FastMCP tools / resources / prompts (in-process Client)

**Later** under `prometheus/application/harness/`:

- Load `.env` (OpenRouter + data keys)
- LangGraph / LangChain multi-agent desk
- Use cases: Case 1 ($N), Case 2 (trend phrases), Case 3 (“What’s Happening?”)

Suggested growth order (manual coding):

1. `models/` — local LLM done; OpenRouter chat model done (`models/chat.py`, Part 6 of the RAG guide)  
2. `mcp/` + `tools/` — FastMCP quote path in place; the Data Clerk's tools done (`tools/data_clerk.py`) and served over MCP (`mcp/data_clerk_server.py`, Part 8); Redis next  
3. `prompts/` + one agent in `agents/` — the Data Clerk agent (Part 7) and an Advisor v0 with a score calculated from checked quotes (Part 8: `agents/advisor/advisor.py`, `prompts/advisor.py`) done; the work order is in `common/work_order.py`  
4. `langgraph/` — wire agents into a graph  
5. `application/harness/` — richer CLI / multi-case entry  

`rag/` is done through retrieval and citations (loaders, store, ingest, retriever, citations; see `labs/module3_rag/hpw_to_do_langchain_rag.md`). The RAG build is done through Part 8: the Data Clerk agent (a budgeted LangGraph loop with a citation check and defences against hidden instructions), its MCP server, the work order, and an Advisor v0. The desk graph (`langgraph/desk.py`, Part 9) wires the Account Manager, the Data Clerk and the Advisor together. **Week 4 (design/week4.md, design choices entry 34) added press releases as a second source:** a work order can ask for them by symbol and/or company name (the symbol first, then the name, exact matches only, in code before any model runs), they are fetched into `<SYMBOL>_PRs.txt`, indexed in their own collection and searched with a new `retrieve_press_releases` tool. **Filings (Week 4, part 2; `design/week4.md` sections 11 to 21, design choices entry 35) are the third source, asked for the same way:** `source="filings"` with its own action `search_filing` (design choices entry 37), by symbol and/or company name, resolved by the same code with the same messages, the Account Manager passes them on the work order (`--source filings`, also on `./test_prometheus.sh`), one Document per section of a 10-K or 8-K and one short Document for each filing the API gives no text for (most of them), in their own collection and searched with `retrieve_filings`. The 10-Q has no text in the paid API, so `fetch_filings` also takes it from SEC EDGAR (design choices entry 36, `design/week4.md` sections 22 to 29; it needs `SEC_USER_AGENT` in `.env`, a name and a contact e-mail, and is skipped and reported without it). The code, the tests (1353 passed before part 7) and mutation checks (38, 34, 9, 32 and 12, all caught) are done; **the 10-Q text of AAPL's 3 and META's 4 10-Qs was fetched into `data/filings` on 2026-10-30 (with Jacob's yes); the index and the desk run have not been made** (they need a yes). A tool of the Data Clerk, `fetch_filings_recommendation(symbol)` (entry 38, `design/week4.md` part 5), reads the stored filings and returns evidence for the Advisor (the balance sheet for both periods, buybacks, dividends, share count, and exact quotes about dilution and executive compensation) and **scores nothing**: the Advisor scores the sentiment in a later step. The Data Clerk puts the results of each source (news, press releases, filings: which numbered sources and which are cited) and that filing evidence on the work order (`data.results`, `data.filing_evidence`, design choices entry 39), so the next agent reads them from the order. **Every order carries all three sources by design (Week 4, part 7, `design/week4.md` sections 49 to 54, design choices entry 40):** after the model's loop the Data Clerk's code searches the corpora the model did not search and reads the stored filing evidence (stored text only, nothing fetched; `data.sweep` says who searched what, `data.results[...]["swept"]` which numbers code added), and **the Advisor reads `data.results` and `data.filing_evidence`**: a signal may cite an evidence item (`"source": "E3"`), which code checks (the item exists for that company, the quote is in it, the direction is the one its type fixes) before it counts toward the calculated score. The tests are 1380 passed (19 mutation checks, all caught); the Advisor was not run with the real model. The Critic, the Trader and a full Account Manager are next, and so are the other work-order actions (`fetch_quote`, `calculate_financials`, `search_socials`: the Data Clerk's `verify_request` still refuses them). Later: `memory/`, `evals/`, paper-trading hooks.

## Agents (trading desk)

| Folder target        | Role |
|----------------------|------|
| `agents/data_clerk/` Data Clerk | Retrieve / filter / aggregate; call tools. `data_clerk_agent.py` is the agent; `data_toolbox.py` is its toolbox. Built (search news, press releases and filings) |
| `agents/advisor/` Advisor | Recommend opportunities; memory + tools. Version 0 (`advise`) and version 1 (`generate_recommendations`, week 5) built. It recommends companies TO BUY and a plan of prices and dates; no amount (entry 80), and it is given no portfolio: it never reads the user's cash or holdings and writes no sale, addition or hold (entry 105) |
| `agents/critic/` Critic | Approve or deny each recommendation (Critic v1, `design/critic_v1_final_spec.md`, entries 56 to 59): asks the Data Clerk again for the numbers, a quote and the sources and compares; judges the company's risk against the user's risk tolerance, expiry (the market's trading window) and value (against the six-month average); one model call per recommendation (Z.ai GLM 5.3) that only labels quotes; a score from 0 to 100. It does not size and is never given the portfolio (entry 80); a review that wants a person to look is DENIED (entry 103); it judges recommendations to buy and nothing about the holdings (entry 105). Built |
| `agents/account_manager/` Account Manager | Cash, T+1, positions, approve trades. RaiseFunds, the user's profile, `decide_trades` (the Critic's verdicts become permissions: an approval trades in Auto mode with nothing asked and is asked about in Ask mode; a denial is returned to the user, entry 104) and `sizing.py` (how much to buy or sell, from the user's money alone: no risk management and no size limit, entries 80 and 91) and `neutral.py` (a neutral recommendation counts as approved with `ALLOW_NEUTRAL_RECOMMENDATIONS`, entry 91) are built; as nodes in `langgraph/desk.py`. The Postgres account book (funds, holdings, booked trades) and the Trader's only door: its prices, its portfolio, and the owner of the funds while it trades: it approves each trade and records each fill as it happens (`handshake.py`, entry 81): a SELECT, one transaction under row-level locks on the whole portfolio, a SELECT again, and a trade that fails is shown to the user and never retried (entry 82); in Ask mode it asks how much to buy when the request did not say (`clarify.py`, entry 85); a request that names the trade ("Buy 50 shares of MSFT", "of Microsoft") is a direct order: no recommendation, no Critic; it asks the Data Clerk for the company name and symbol and appends them to the work order, prices and decides it here (a named company is confirmed in Ask mode; no profile limit applies) and the Trader places it as typed or not at all (`direct.py`, entries 87 and 88) |
| `agents/trader/` Trader | Executes the permitted trades against a SIMULATED broker (Trader v1, `design/trader_v1_final_spec.md`, entries 73 to 75): `paper` fills at the current quote, `backtest` at the next session's open from the stored daily bars and follows each purchase to its stop, target or time stop. There is no live trading. A model (gpt-oss-120b, temperature 0) works through four tools that cannot name a symbol, a side or a number of shares; code places what it leaves and checks everything. Reaches the world only through the Account Manager and holds no cash of its own: every trade is approved and recorded by the Account Manager, one at a time (entry 81). `desk --critic --trade`, `./test_recommendations.sh --trade`, and the free replay `python -m prometheus.agents.trader.replay`. Built; no live run yet. Future Alpaca: not built |

## Intentional non-goals for this scaffold

- No full-env lockfile (`pip freeze`) — keep `requirements.txt` curated; pin tighter later if Spark installs drift
- `labs/` stays read-only reference; copy patterns into `prometheus/` yourself
