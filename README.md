# Prometheus

Self-learning multi-agent trading bot — CMU Agentic AI Capstone (Fall 2026).

## Layout

```
capstone/
├── design/                 # Capstone design docs (keep as source of truth)
├── labs/                   # Course notebooks — manually port snippets from here
├── prometheus/             # Application source (empty shells for now)
│   ├── application/        # Orchestration / use-case entrypoints
│   │   └── harness/        # How you run & drive the multi-agent loop
│   ├── agents/             # Data Clerk, Advisor, Critic, Account Manager, Trader
│   ├── mcp/                # Local FastMCP servers (tools / resources / prompts)
│   ├── rag/                # Embeddings, vector store, retrieval
│   ├── reasoning/          # CoT / critique / ranking helpers
│   ├── prompts/            # Prompt templates (per agent / use case)
│   ├── langchain/          # LangChain pieces you port from labs
│   ├── langgraph/          # LangGraph graphs / state machines
│   ├── tools/              # Local Python tools (calc, sentiment, search, …)
│   ├── models/             # OpenRouter client wiring + domain JSON schemas
│   ├── memory/             # Short / long-term memory (later modules)
│   ├── evals/              # Hallucination, brittleness, cost/quality/latency
│   └── common/             # Shared non-agent helpers
├── configs/                # Non-code config (models, agent YAML, later)
├── data/
│   ├── samples/            # Small explainability samples for demos
│   ├── raw/                # Downloaded / API dumps (local)
│   └── processed/          # Cleaned data, vector indexes (local)
├── scripts/                # Thin shell/python runners you’ll add later
├── tests/                  # pytest (FastMCP tools + later app tests)
├── pytest.ini              # asyncio_mode=auto for FastMCP Client tests
├── requirements.txt        # Curated runtime deps (reproducible install)
├── .env.example            # Safe template (committed)
└── .env                    # Your keys (gitignored)
```

Keep layers thin: put new code in the folder that matches its job. Prefer porting from `labs/` into one of these folders rather than growing notebook-only logic.

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

- **`mode="legacy"`** — Client setting (not `server.run()`). Needed for sampling / roots / elicitation; use with in-process *or* stdio hosts.
- **`cache=False`** — **Never set `cache=True` or will break the FastMCP `market://{symbol}/quote` resource.**
- Server: `cache_ttl=None` in `server.py`. Stdio: `python prometheus/mcp/server.py`; host connects via `client.py`.

## Environment variables

```bash
cp .env.example .env   # already created; edit .env with real keys
```

Minimum to start LLM work: set `OPENROUTER_API_KEY` in `.env`.

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

This is what lives under `prometheus/application/harness/` once you start coding:

- Load `.env` (OpenRouter + data keys)
- Start / connect to **local FastMCP** tools under `prometheus/mcp/`
- Build / invoke the LangGraph (or LangChain) graph
- Drive use cases: Case 1 ($N), Case 2 (trend phrases), Case 3 (“What’s Happening?”)

Suggested growth order (manual coding):

1. `models/` — OpenRouter chat client from `.env`  
2. `mcp/` + `tools/` — local FastMCP tools (reuse Lab 1.2 patterns)  
3. `prompts/` + one agent in `agents/`  
4. `langgraph/` — wire agents into a graph  
5. `application/harness/` — CLI or script entry that runs one use case end-to-end  

Later: `rag/`, `memory/`, `evals/`, paper-trading hooks.

## Agents (trading desk)

| Folder target        | Role |
|----------------------|------|
| `agents/` Data Clerk | Retrieve / filter / aggregate; call tools |
| `agents/` Advisor    | Recommend opportunities; memory + tools |
| `agents/` Critic     | Reason / rank / approve or deny |
| `agents/` Account Manager | Cash, T+1, positions, approve trades |
| `agents/` Trader     | Paper trade / backtest; future Alpaca |

## Intentional non-goals for this scaffold

- No full-env lockfile (`pip freeze`) — keep `requirements.txt` curated; pin tighter later if Spark installs drift
- `labs/` stays read-only reference; copy patterns into `prometheus/` yourself
