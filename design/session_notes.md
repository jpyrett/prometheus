# Session notes — Capstone Prometheus MCP (handoff)

**Date:** 2026-10-18  
**Project:** Capstone Prometheus (multi-agent trading bot)  
**Repo:** `/run/media/jpyrett/980pro/CMU-AgenticAI/capstone`  
**Goal:** Manually port Lab 1.2 MCP into `prometheus/mcp/` (not auto-copy notebook).

Open this file in the next chat (`@design/session_notes.md`) so context is not lost.

---

## Current architecture decisions (locked)

| Topic | Decision |
|--------|----------|
| MCP framework | **FastMCP 4** (`fastmcp` package, PrefectHQ) on **mcp ≥ 2** — not old `MCPServer` |
| Layout | `prometheus/mcp/server.py` = server only; `prometheus/mcp/client.py` = host Client; **server must not import client** |
| Client defaults | `mode="legacy"`, **`cache=False`** (never `cache=True` — breaks live quotes) |
| Server cache | `FastMCP(..., cache_ttl=None)` — no server-side cache hints |
| Quotes | Resource `market://{symbol}/quote` — stub now; **Redis later**; no response caching |
| Prompts vs resources | Resource = load data into context; Prompt = user intent template (**do not name URIs/tools in prompt text**) |
| LangGraph | Above MCP; owns LLM calls — **not** MCP sampling |
| Deps / env | `requirements.txt`; prefer **cmu conda** for DGX Spark reproducibility |
| Tests | `pytest` + `connect_client()` fixtures in `tests/` |

### Part 5 primitives (notebook → product)

| Primitive | Status | Why |
|-----------|--------|-----|
| **Elicitation** | **Kept** | HITL still current. Server: `enter_symbol`; Client: `elicitation_handler` (auto `symbol=AAPL` for tests) |
| **Logging** | **Kept (demo)** | `noisy_fetch_quote` → `ctx.info`/`ctx.warning` around fetch; client `log_handler`. Note: MCP logging capability deprecated 2026-07-28 (SEP-2577) — still useful for demos/tests |
| **Progress** | **Kept** | `report_progress` tool → `ctx.report_progress`; client `progress_handler` |
| **List-changed** | **Kept (catalog only)** | tools / prompts / static resources via `ping_catalog` + `PrometheusMessageHandler`. **Not** `resources/updated` for quotes |
| **Cancellation** | **Kept (client hook)** | `on_cancelled` in message handler; host can `client.cancel(request_id)` |
| **Sampling** | **Removed** | Deprecated 2026-07-28; summarize/LLM on LangGraph host, not `ctx.session.create_message` |
| **Roots** | **Removed** | Deprecated; use **tool args / resource URIs / config** (already: `market://…`, `config://server`, tool params) |

---

## Key files

```
main.py                    # App entry: LocalLLM + PrometheusHost MCP smoke
prometheus/models/llm.py   # Lab 1.1 AutoTokenizer / CausalLM / chat() / show()
prometheus/application/host.py  # PrometheusHost — tools/prompts/resources + LLM
prometheus/mcp/server.py   # tools/resources/prompts; elicitation + logging/progress/list-changed
prometheus/mcp/client.py   # connect_client(); elicitation + log/progress/message handlers
prometheus/mcp/__init__.py
tests/fastmcp_tools.py     # initials + enter_symbol + notification demos
tests/fastmcp_quotes.py    # quote resource / cache-related tests
labs/.../model_context_protocol_solution.ipynb  # FastMCP 4 lab (reference)
labs/.../introduction_to_large_language_models_solution.ipynb  # LLM chat() source
design/main.md             # mermaid architecture
design/choices_week2.txt   # design scratch
requirements.txt           # mcp>=2, fastmcp>=4, transformers, pytest, …
```

### Server tools / resources / prompts (today)

- **Tools:** `fetch_quote`, `verify_fetch_quote`, `enter_symbol` (HITL), `noisy_fetch_quote` (logging), `report_progress` (progress), `ping_catalog` (list-changed)
- **Resources:** `config://server`, `market://{symbol}/quote` (stub)
- **Prompts:** `get_market_data` (briefing intent only — no URI names)

### Client API

```python
from prometheus.mcp.client import connect_client
async with connect_client() as c:
    ...
# aliases: client = connect_client
# CLIENT_MODE = "legacy", CLIENT_CACHE = False
# Handlers: elicitation, log, progress, PrometheusMessageHandler (list-changed + cancel)
```

---

## Tests (last known good)

```bash
conda run -n cmu python -m pytest tests/fastmcp_tools.py tests/fastmcp_quotes.py -q -k 'not forced_failure'
# → 19 passed, 2 skipped, 1 deselected
```

Note: `test_forced_failure_known_demo` is intentional red demo — deselect with `-k 'not forced_failure'`.

---

## Git status at handoff (uncommitted / dirty)

Expect roughly:

- Modified: `design/choices_week2.txt`, lab notebook, possibly README
- Untracked: `prometheus/__init__.py`, `prometheus/mcp/__init__.py`, `.vscode/extensions.json`
- Deleted: `tests/.gitkeep`
- Active work under `prometheus/mcp/*`, `tests/fastmcp_*`, `design/*`

**Do not commit unless Jacob asks.**

---

## App entry (LLM + MCP)

```bash
conda activate cmu
python main.py --skip-llm          # MCP only (fast)
python main.py                     # load LOCAL_MODEL + briefing
python main.py --symbol MSFT --thinking
```

## Next work (suggested order)

1. **Redis-backed quotes** — replace stub in `quote()`; keep `cache=False` / `cache_ttl=None`
2. **Trading HITL** — evolve `enter_symbol` → richer elicit schemas (e.g. confirm size/side) + real UI handler (not auto-AAPL)
3. **Tool-calling loop** — map MCP `list_tools` schemas into chat template `tools=` (Qwen3)
4. **LangGraph host** — wire agents above MCP; LLM sampling lives there
5. **More resources** — from `choices_week2.txt` (portfolio, news, risk docs, etc.)
6. Commit when Jacob requests; update `design/main.md` mermaid if Redis lands

---

## Hard rules to preserve

1. Never `cache=True` on Client for quote path.
2. Never import `client.py` from `server.py`.
3. Prompt text must not name MCP URIs/tools.
4. Prefer tool args / resource URIs / config over Roots.
5. Prefer host LLM over MCP Sampling.
6. Manual port from lab — keep Prometheus naming and trading domain.

---

## Quick paste for new chat

> Continue Capstone Prometheus from `@design/session_notes.md`.  
> Stack: FastMCP 4 + `main.py` / `PrometheusHost` / `LocalLLM`, cache=False, elicitation + notifications, sampling/roots removed.  
> Run: `python main.py --skip-llm` or full LLM smoke. Next: Redis / tool-calling loop / LangGraph.
