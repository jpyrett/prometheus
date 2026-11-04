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
│   │   └── harness/        # How you run & drive the multi-agent loop
│   ├── agents/             # Data Clerk, Advisor, Critic, Account Manager, Trader
│   │   ├── data_clerk/     # data_clerk_agent.py (DataClerk: the LLM agent), data_toolbox.py (DataToolbox: plain Python, no LLM)
│   │   ├── advisor/        # advisor.py (Advisor v0: a score calculated from checked quotes)
│   │   └── account_manager/ # account_manager.py (AccountManager: the LLM agent behind RaiseFunds, a Tree of Thought)
│   ├── mcp/                # Local FastMCP servers (tools / resources / prompts)
│   ├── rag/                # Embeddings, vector store, retrieval
│   ├── reasoning/          # raise_funds.py: the Tree of Thought for what to sell to buy a stock (a LangGraph; design/week4_tot_guide.md)
│   ├── prompts/            # Prompt templates (per agent / use case)
│   ├── langchain/          # LangChain pieces you port from labs
│   ├── langgraph/          # desk.py: the top-level graph, account_manager -> data_clerk -> advisor (-> account_manager_funds with --raise-funds)
│   ├── tools/              # Local Python tools (calc, sentiment, search, …)
│   ├── models/             # OpenRouter client wiring + domain JSON schemas
│   ├── memory/             # Short / long-term memory (later modules)
│   ├── evals/              # Hallucination, brittleness, cost/quality/latency
│   └── common/             # Shared non-agent helpers (work_order.py: the work order every agent reads)
├── configs/                # Non-code config (portfolios/jacob_pyrett.json: the static sample portfolio; models, agent YAML, later)
├── data/
│   ├── samples/            # Small explainability samples for demos
│   ├── raw/                # Downloaded / API dumps (local)
│   └── processed/          # Cleaned data, vector indexes (local)
├── scripts/                # Thin runners (raise_funds_offline_demo.py: RaiseFunds with invented quotes and a scripted LLM, no cost)
├── tests/                  # pytest (FastMCP tools + later app tests)
├── main.py                 # App entry: LocalLLM + FastMCP smoke / HITL
├── test_prometheus.sh      # Main script: prompts for symbol / term / source / from_days_ago, runs the desk (a LangGraph passing one work order); pytest stays in tests/
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

pytest tests/fastmcp_tools.py tests/fastmcp_quotes.py -q -k 'not forced_failure'
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
| `agents/advisor/` Advisor | Recommend opportunities; memory + tools. Version 0 built |
| `agents/` Critic     | Reason / rank / approve or deny. Not built |
| `agents/` Account Manager | Cash, T+1, positions, approve trades. Only the request half exists, as a node in `langgraph/desk.py` |
| `agents/` Trader     | Paper trade / backtest; future Alpaca. Not built |

## Intentional non-goals for this scaffold

- No full-env lockfile (`pip freeze`) — keep `requirements.txt` curated; pin tighter later if Spark installs drift
- `labs/` stays read-only reference; copy patterns into `prometheus/` yourself
