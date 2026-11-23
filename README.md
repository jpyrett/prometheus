# Prometheus: an Agentic Stock Trading Desk

Prometheus is a multi-agent stock trading desk built for the CMU Agentic AI capstone (Fall 2026). Five agents in the roles of a human trading desk take a request in plain English and carry it out together: "Buy the 3 most recommended oil stocks with $3,000", "Buy 10 shares of MSFT", "Show the news for AAPL". Trading is **simulated**: no real broker is ever called.

| Agent | Role |
|---|---|
| **Account Manager** | The supervisor. Reads the request, owns the account (Postgres), decides how much to buy, and asks the user questions in Ask mode. |
| **Data Clerk** | Gathers prices, news, press releases and SEC filings, searches them (RAG over Chroma) and calculates about 80 metrics in code. |
| **Advisor** | Scores and ranks the candidates and writes each recommendation with a thesis and a trade plan. |
| **Critic** | Re-checks every recommendation on a different model and approves or denies it with reasons. |
| **Trader** | Places only the trades the Account Manager permitted, one at a time, on a simulated broker. |

A Security Guard screens every request, LangSmith traces every step, and the **Proving Ground** replays the desk on past days to measure whether its recommendations made money.

- **Final report:** [`design/final_capstone_report.pdf`](design/final_capstone_report.pdf) ([Markdown](design/final_capstone_report.md))
- **Architecture diagram:** [`design/final_presentation_arch.png`](design/final_presentation_arch.png)
- **Screenshots:** [`design/presentation/screenshots/`](design/presentation/screenshots/) (light) and [`screenshots_dark/`](design/presentation/screenshots_dark/)
- **Every command:** [`HowToRun.txt`](HowToRun.txt) · **Every design decision:** [`docs/design_choices.md`](docs/design_choices.md)

---

## Quick start for reviewers

There are two ways to run it:

| | **A. Replay (recommended first)** | **B. Live** |
|---|---|---|
| What it does | The real web page plays back a recorded real session, stage by stage | The real agents run your own request |
| Cost | Free | Paid API calls (OpenRouter, Finnhub; Massive if you refresh data) |
| Needs | Python + Node.js | Python + Node.js + Docker + API keys |
| Time to set up | About 10 minutes | About 30 minutes |

### 1. Prerequisites

- **Linux or macOS** (on Windows, use WSL2 with Ubuntu).
- **Git**.
- **Miniconda or Anaconda** (for Python 3.13): https://docs.conda.io/en/latest/miniconda.html
- **Node.js 20 or 22** (LTS), which includes `corepack`: https://nodejs.org. Check with `node --version` and `corepack --version`.
- **Live mode only:** Docker with the Compose plugin (`docker compose version`), and `openssl`.
- About 3 GB of free disk space (Python packages, the page's packages, and the local embedding model).

### 2. Get the code and install the Python packages

```bash
git clone https://github.com/jpyrett/prometheus.git
cd prometheus

conda create -n cmu python=3.13 -y
conda activate cmu
pip install -r requirements.txt
```

`pip` installs PyTorch as a dependency of `sentence-transformers` (the local embedding model). A CPU is enough; no GPU is needed.

> Run every command below from the repository root, with `conda activate cmu` done in that terminal.

### 3A. Replay a recorded session (free: no keys, no database, no network)

```bash
./run_ui.sh --replay
```

The first start installs the page's packages and builds the page (1 to 2 minutes). Then open **http://127.0.0.1:8000** and press **Go!**. The page plays a real recorded session: "Bob Jones" asks for oil stocks, and you can watch the six stages run. The Account Manager reads the request, the Data Clerk gathers the data, the Advisor recommends, the Critic approves or denies, the Account Manager decides, and the Trader trades. The work order, the Critic's verdicts, the trades and the updated portfolio appear as the run finishes. Stop the server with **Ctrl+C**.

Other recorded sessions can be replayed the same way:

```bash
./run_ui.sh --replay tests/fixtures/news_real/news_tsla_week_fetched.json        # "What's Tesla doing the last week?": the News pane
./run_ui.sh --replay tests/fixtures/quote_real/quote_tsla.json                    # "Whats the current price of: TSLA?": the price in the output
./run_ui.sh --replay tests/fixtures/session_real/bill_gates_direct_order.json     # a direct order: "Buy ... shares of ..."
./run_ui.sh --replay tests/fixtures/session_real/ant_raise_funds_ask.json         # selling holdings to pay for a purchase (RaiseFunds)
./run_ui.sh --replay tests/fixtures/session_real/auto_ignore_critic.json          # Auto mode with "Ignore Critic"
```

Every file in `tests/fixtures/session_real/` that holds a whole session can be replayed. The **Proving Ground** tab shows the recorded backtests of `tests/fixtures/proving_real/`, and **Watch it again** replays them for free.

### 3B. Run it live (paid)

**Step 1: API keys.** Copy the settings template and fill in your keys. The comments in the file say which key is needed for what.

```bash
cp .env.example .env
```

| Setting in `.env` | Needed for | Where to get it |
|---|---|---|
| `OPENROUTER_API_KEY` | Every live run (all the agents' models) | https://openrouter.ai/keys (add a few dollars of credit) |
| `FINNHUB_API_KEY` | Every live run (quotes, company lookup, news) | https://finnhub.io/register |
| `POLYGON_API_KEY` | Only "Refresh market data first" (daily bars, news, press releases). Use your **Massive** key: Massive is the renamed Polygon.io | https://massive.com |
| `SEC_USER_AGENT` | Only fetching SEC 10-Q text: your name and e-mail | (no sign-up) |
| `LANGSMITH_API_KEY` | Optional tracing. **Leave it empty if you have none** | https://smith.langchain.com |

Leave every other setting as it is. One that matters: outside US market hours (9:30 to 16:00 Eastern on trading days), the Critic denies every recommendation because the market is closed. To try the desk in the evening or at the weekend, set `IGNORE_MARKET_HOURS=true` in `.env`.

**Step 2: The account database (Postgres in Docker).**

```bash
bash scripts/setup_postgres.sh
```

This starts a Postgres 18 container on `127.0.0.1:5432`, creates the schema and three sample accounts, and writes generated passwords to `secrets/` (which git ignores). Running it again keeps all the data.

**Step 3: Build the search index (free, once, a few minutes).** The repository has the stored news, press releases and filings as text. The search index is built from them on your machine. The first run downloads the small MiniLM embedding model.

```bash
python -m prometheus.rag.ingest --all
python -m prometheus.rag.ingest --press-releases --all
python -m prometheus.rag.ingest --filings --all
```

**Step 4: Start the page.**

```bash
./run_ui.sh
```

Open **http://127.0.0.1:8000**:

1. **Create an account** in the profile card: a name, an age, a risk level and starting funds (simulated money).
2. Choose **Ask** mode (you approve every trade) or **Auto** mode (the desk trades what the Critic approves).
3. **Keep "Refresh market data first" ticked** (it is on by default). The 26 GB of historical price data is not in the repository, so the Data Clerk must fetch the bars and news for the companies you ask about. This needs `POLYGON_API_KEY` and costs API calls.
4. Type a request and press **Go!**. Try these:
   - `Show the news for AAPL`
   - `What's the current price of AAPL?`
   - `Buy 5 shares of MSFT`
   - `Buy the 3 most recommended oil stocks with $3,000`
   - `What are the best AI stocks? Spend $2,000 split evenly`
5. In Ask mode, answer the question cards (how much, approve each trade). Results appear in the work order, the portfolio and the activity log.

The Trader fills on a **simulated** broker at the live quote. No real money moves.

**Command-line alternative** (no web page): one whole session, which creates a throwaway account and deletes it afterwards.

```bash
python -m prometheus.application.session --name "Test User" --cash 10000 --mode auto \
  --prompt "Buy the 2 most recommended oil stocks with \$3,000" --fetch --yes --cleanup
```

### 4. Run the tests

```bash
python -m pytest -q --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo
cd ui && corepack npm@10.9.2 ci && corepack npm@10.9.2 test && cd ..   # page tests (ci installs the packages once)
```

- The tests are free: no network call is made (a guard in `tests/conftest.py` blocks any fetch).
- The one test that is deselected fails on purpose (a course demo).
- Tests that use the local Postgres skip themselves when it is not running.
- On the developer's machine the full suite passes: 2,749 passed and 35 skipped, plus 198 page tests.
- Many tests read local data that is not in the repository: the 26 GB of historical daily bars (`data/historical/`), the search index and the saved Proving Ground runs. On a fresh clone most of them skip themselves, and about 17 fail for want of that data. On a fresh clone of this repository: 2,020 passed, 747 skipped, 17 failed (in about 3 minutes).

### 5. Troubleshooting

| Problem | Fix |
|---|---|
| `ERROR: no Python found` from `./run_ui.sh` | Run `conda activate cmu` first, or `PYTHON=/path/to/python ./run_ui.sh --replay` |
| `corepack: command not found` | Install Node.js 20 or 22 (corepack ships with it), or run `npm install -g corepack` |
| Port 8000 is in use | `./run_ui.sh --replay --port 8100`, then open http://127.0.0.1:8100 |
| The page says the account book cannot be opened | Live mode needs Postgres: run `bash scripts/setup_postgres.sh` (Docker must be running) |
| Every recommendation is denied with `market_closed` | The market is closed: set `IGNORE_MARKET_HOURS=true` in `.env` |
| "nothing is stored yet" / no companies found | Build the index (Step 3), or tick "Refresh market data first" |
| `... is not set in .env` | Fill in that key in `.env` (see the table in Step 1) |

---

## Where to look

| Path | What it is |
|---|---|
| `prometheus/langgraph/desk.py` | The LangGraph that connects the agents: one work order passed from agent to agent |
| `prometheus/agents/` | The five agents: `account_manager/`, `data_clerk/`, `advisor/`, `critic/`, `trader/` |
| `prometheus/reasoning/raise_funds.py` | The Tree-of-Thought search for which holdings to sell |
| `prometheus/security/` | The Security Guard: the YAML policy, the code rules, the guard model, the router |
| `prometheus/proving/` | The Proving Ground: backtests of the real desk, the track record, the live watch |
| `prometheus/evals/`, `scripts/run_offline_evals.py` | Offline evaluation: the red-team and session datasets, the evaluators, the Opus judge |
| `prometheus/application/web/`, `ui/` | The web page: FastAPI with Server-Sent Events, and React + TypeScript |
| `configs/` | Every threshold (`advisor_v1.json`, `critic_v1.json`, `account_manager_v1.json`, `trader_v1.json`, `security_v1.yaml`) |
| `tests/`, `tests/fixtures/` | The tests and the recorded real sessions they use |
| `design/`, `docs/design_choices.md` | The specifications and the numbered design decisions (149 entries) |

---

## Developer notes

The notes below are kept from the build. They describe the full layout and the course-lab tools (the local Hugging Face model, FastMCP, and the Cursor setup).


### Layout

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
│   ├── proving/            # The Proving Ground (design choices entries 126 to 132; design/proving_ground_spec.md): did the recommendations work? engine.py (the Time Machine: the real desk replayed at past days, fills at the next open, exits by the Advisor's plan), speed.py (fast, the default, and faithful), outcomes.py / portfolio.py / stats.py / splits.py / record.py (pure: the scoring, the test's own money, the statistics, the result), track.py (the Track Record of the saved real sessions), live.py (the paper account against its plans: it watches, it never sells), coverage.py, memo.py, tracing.py (LangSmith), store.py (results under data/proving/); run with ./test_proving.sh
│   ├── security/           # Agentic security v1 (design choices entry 116; design/agentic_security_spec.md): policy.py (loads configs/security_v1.yaml), rules.py (the code rules), scorer.py (the LLM Risk Scorer: role guard, GUARD_MODEL), router.py (act: BLOCK/REDACT_OR_REVIEW/ALLOW_AND_FLAG/ALLOW, and the Trust Score), guard.py (the doors: the user's request, the retrieved documents, the messages to the user), audit.py (logs/security_audit.jsonl), rate.py. SECURITY risk only: it never refuses or resizes a trade
│   ├── evals/              # Offline evaluation (design choices entry 117; design/offline_evaluation_spec.md): datasets.py (the red-team set and the real session captures as LangSmith examples), evaluators.py (code evaluators: attack_blocked, false_positive, within_pot, cash_math, ...), judge.py (the Opus risk judge: openevals, role judge, JUDGE_MODEL; prompts/risk_judge.py is Jacob's rubric), targets.py; run with scripts/run_offline_evals.py (--dry-run is free)
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
├── configs/                # Non-code config (portfolios/ and profiles/jacob_pyrett.json: the static sample portfolio and user profile; advisor_v1.json, critic_v1.json, account_manager_v1.json (the sizing), trader_v1.json: every threshold; security_v1.yaml (the security policy: thresholds, patterns, messages) and security/red_team_v1.jsonl (the red-team set); market_calendar.json)
├── sql/                    # prometheus_accounts.sql and seed_test_users.sql (the account book), pg_hba.conf
├── docker-compose.yml      # Postgres 18 for the account book, bound to 127.0.0.1 (scripts/setup_postgres.sh)
├── data/
│   ├── samples/            # Small explainability samples for demos
│   ├── raw/                # Downloaded / API dumps (local)
│   └── processed/          # Cleaned data, vector indexes (local)
├── scripts/                # Thin runners (raise_funds_offline_demo.py: RaiseFunds with invented quotes and a scripted LLM, no cost; run_offline_evals.py: the offline evaluation, entry 117, --dry-run is free)
├── tests/                  # pytest (FastMCP tools + later app tests)
├── main.py                 # App entry: LocalLLM + FastMCP smoke / HITL
├── ui/                     # The web page (React 19 + Vite + TypeScript, no UI kit): `cd ui && corepack npm@10.9.2 run build`; tests with vitest; design/ui_v1_spec.md, the look: design/ui_v1_mockup.html, screenshots: docs/ui_screenshots/
├── test_proving.sh         # The Proving Ground by command: track, coverage, index, show (free); live --account N (quotes); time-machine (PAID: the models, never a fetch)
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

### Working with Cursor agents

Open the project in Cursor (editor or Agents Window; in the editor run **Open Agents Window** from the Command Palette). Use a **local** agent: it runs in this checkout, so it sees the `cmu` conda env, `.env` and `data/`. A worktree or cloud agent starts from a fresh git checkout without the git-ignored `.env` and `data/` (see [HowToRun.txt](HowToRun.txt), section 0).

**[AGENTS.md](AGENTS.md)** is read by every agent chat automatically. It holds the facts an agent would otherwise have to rediscover: the Python to use, the one test that fails on purpose, where each part lives, and what must never be run without asking (anything that calls a paid API or writes the news files, and anything touching `.env`). Keep it short; when you rename or move a thing, update it with the other docs. Cursor does not read a file called `Cursor.md`.

### How to run

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

### Conda + reproducible deps

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

### Testing (pytest)

We use **pytest** (+ **pytest-asyncio**) for unit tests and in-process FastMCP `Client(server)` tests. Config lives in `pytest.ini` (`asyncio_mode = auto`).

```bash
conda activate cmu
pip install -r requirements.txt   # includes pytest / pytest-asyncio
pytest                            # from the repo root
```

FastMCP tools: `tests/fastmcp_tools.py` (imports `prometheus.mcp.server`).

#### FastMCP client connection (Part 4) + caching (quotes)

Prefer `connect_client()` from **`prometheus/mcp/client.py`** (host-side; `server.py` does not import it):

- **`mode="legacy"`** — Client setting (not `server.run()`). Needed for elicitation (HITL); use with in-process *or* stdio hosts. Sampling and roots are deprecated — prefer host LLMs and tool args / resource URIs / config.
- **`cache=False`** — **Never set `cache=True` or will break the FastMCP `market://{symbol}/quote` resource.**
- Server: `cache_ttl=None` in `server.py`. Stdio: `python prometheus/mcp/server.py`; host connects via `client.py`.

### Environment variables

```bash
cp .env.example .env   # already created; edit .env with real keys
```

For local `main.py` runs: set `LOCAL_MODEL` in `.env` (see `.env.example`).  
For later OpenRouter agents: set `OPENROUTER_API_KEY` in `.env`.

### Two “harnesses” (do not confuse them)

#### 1) Cursor + OpenRouter — coding / agent harness (IDE)

This is how *you* develop with Cursor calling OpenRouter models while you write Prometheus by hand.

1. OpenRouter API key: https://openrouter.ai/keys  
2. Cursor → Settings → Models → API Keys  
3. Enable **OpenAI API Key**, paste the OpenRouter key (`sk-or-…`)  
4. Enable **Override OpenAI Base URL** and set:

   `https://openrouter.ai/api/v1/cursor`

5. Add model IDs you want in the model picker (OpenRouter model slugs).

That routes Cursor’s agent/chat through OpenRouter. It is **not** the trading bot runtime.

#### 2) Prometheus application harness — running the bot

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

**Build notes from Weeks 3 and 4 (retrieval and the three sources).** These record the state of the build at that time. The Critic, the Trader, the full Account Manager and everything after them came later: see the Quick start above and `docs/design_choices.md`.

- **RAG.** `rag/` is done through retrieval and citations: loaders, store, ingest, retriever and citations (guide: [`labs/module3_rag/how_to_do_langchain_rag.md`](labs/module3_rag/how_to_do_langchain_rag.md)).
- **Through Part 8.** The Data Clerk agent (a budgeted LangGraph loop with a citation check and defences against hidden instructions), its MCP server, the work order, and an Advisor v0.
- **The desk graph** (`langgraph/desk.py`, Part 9) wires the Account Manager, the Data Clerk and the Advisor together.
- **Press releases, the second source** (Week 4; `design/week4.md`, design choices entry 34):
  - A work order can ask for them by symbol and/or company name: the symbol first, then the name, exact matches only, in code before any model runs.
  - They are fetched into `<SYMBOL>_PRs.txt`, indexed in their own collection and searched with the `retrieve_press_releases` tool.
- **Filings, the third source** (Week 4, part 2; `design/week4.md` sections 11 to 21, design choices entries 35 and 37):
  - They are asked for the same way: `source="filings"` with its own action `search_filing`, by symbol and/or company name, resolved by the same code with the same messages. The Account Manager passes them on the work order (`--source filings`, also on `./test_prometheus.sh`).
  - Each 10-K or 8-K becomes one Document per section. Each filing the API gives no text for (most of them) gets one short Document. They go in their own collection and are searched with `retrieve_filings`.
  - The paid API has no 10-Q text, so `fetch_filings` takes it from SEC EDGAR (design choices entry 36, `design/week4.md` sections 22 to 29). This needs `SEC_USER_AGENT` in `.env` (a name and a contact e-mail); without it the 10-Q is skipped and reported.
  - The 10-Q text of AAPL's 3 and META's 4 10-Qs was fetched into `data/filings` on 2026-10-30.
- **Filing evidence for the Advisor.** The Data Clerk's tool `fetch_filings_recommendation(symbol)` (entry 38, `design/week4.md` part 5) reads the stored filings and returns evidence: the balance sheet for both periods, buybacks, dividends, share count, and exact quotes about dilution and executive compensation. It **scores nothing**; the Advisor scores the sentiment later.
- **Results on the work order.** The Data Clerk puts each source's results (which numbered sources there are, and which are cited) and the filing evidence on the work order (`data.results`, `data.filing_evidence`, entry 39), so the next agent reads them from the order.
- **Every order carries all three sources** (Week 4, part 7; `design/week4.md` sections 49 to 54, entry 40):
  - After the model's loop, the Data Clerk's code searches the corpora the model did not search and reads the stored filing evidence. This uses stored text only and fetches nothing. `data.sweep` says who searched what, and `data.results[...]["swept"]` which numbers code added.
  - The Advisor reads `data.results` and `data.filing_evidence`. A signal may cite an evidence item (`"source": "E3"`). Code checks that the item exists for that company, the quote is in it, and the direction is the one its type fixes, before it counts toward the score.
- **Tests at the time.** 1,353 passed before part 7 and 1,380 after, and every mutation check was caught (38, 34, 9, 32, 12 and 19).

### Agents (trading desk)

| Folder target        | Role |
|----------------------|------|
| `agents/data_clerk/` Data Clerk | Retrieve / filter / aggregate; call tools. `data_clerk_agent.py` is the agent; `data_toolbox.py` is its toolbox. Built (search news, press releases and filings) |
| `agents/advisor/` Advisor | Recommend opportunities; memory + tools. Version 0 (`advise`) and version 1 (`generate_recommendations`, week 5) built. It recommends companies TO BUY and a plan of prices and dates; no amount (entry 80), and it is given no portfolio: it never reads the user's cash or holdings and writes no sale, addition or hold (entry 105) |
| `agents/critic/` Critic | Approve or deny each recommendation (Critic v1, `design/critic_v1_final_spec.md`, entries 56 to 59): asks the Data Clerk again for the numbers, a quote and the sources and compares; judges the company's risk against the user's risk tolerance, expiry (the market's trading window) and value (against the six-month average); one model call per recommendation (Z.ai GLM 5.3) that only labels quotes; a score from 0 to 100. It does not size and is never given the portfolio (entry 80); a review that wants a person to look is DENIED (entry 103); it judges recommendations to buy and nothing about the holdings (entry 105). Built |
| `agents/account_manager/` Account Manager | Cash, T+1, positions, approve trades. RaiseFunds, the user's profile, `decide_trades` (the Critic's verdicts become permissions: an approval trades in Auto mode with nothing asked and is asked about in Ask mode; a denial is returned to the user, entry 104) and `sizing.py` (how much to buy or sell, from the user's money alone: no risk management and no size limit, entries 80 and 91) and `neutral.py` (a neutral recommendation counts as approved with `ALLOW_NEUTRAL_RECOMMENDATIONS`, entry 91) are built; as nodes in `langgraph/desk.py`. The Postgres account book (funds, holdings, booked trades) and the Trader's only door: its prices, its portfolio, and the owner of the funds while it trades: it approves each trade and records each fill as it happens (`handshake.py`, entry 81): a SELECT, one transaction under row-level locks on the whole portfolio, a SELECT again, and a trade that fails is shown to the user and never retried (entry 82); in Ask mode it asks how much to buy when the request did not say (`clarify.py`, entry 85); a request that names the trade ("Buy 50 shares of MSFT", "of Microsoft") is a direct order: no recommendation, no Critic; it asks the Data Clerk for the company name and symbol and appends them to the work order, prices and decides it here (a named company is confirmed in Ask mode; no profile limit applies) and the Trader places it as typed or not at all (`direct.py`, entries 87 and 88) |
| `agents/trader/` Trader | Executes the permitted trades against a SIMULATED broker (Trader v1, `design/trader_v1_final_spec.md`, entries 73 to 75): `paper` fills at the current quote, `backtest` at the next session's open from the stored daily bars and follows each purchase to its stop, target or time stop. There is no live trading. A model (gpt-oss-120b, temperature 0) works through four tools that cannot name a symbol, a side or a number of shares; code places what it leaves and checks everything. Reaches the world only through the Account Manager and holds no cash of its own: every trade is approved and recorded by the Account Manager, one at a time (entry 81). `desk --critic --trade`, `./test_recommendations.sh --trade`, and the free replay `python -m prometheus.agents.trader.replay`. Built; no live run yet. Future Alpaca: not built |

### Intentional non-goals for this scaffold

- No full-env lockfile (`pip freeze`) — keep `requirements.txt` curated; pin tighter later if Spark installs drift
- `labs/` stays read-only reference; copy patterns into `prometheus/` yourself
