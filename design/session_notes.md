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
| **Elicitation** | **Kept** | HITL still current. Server: `pick_number`; Client: `elicitation_handler` (auto `favorite_number=24` for tests) |
| **Sampling** | **Removed** | Deprecated 2026-07-28; summarize/LLM on LangGraph host, not `ctx.session.create_message` |
| **Roots** | **Removed** | Deprecated; use **tool args / resource URIs / config** (already: `market://…`, `config://server`, tool params) |

---

## Key files

```
prometheus/mcp/server.py   # FastMCP tools/resources/prompts; pick_number elicitation
prometheus/mcp/client.py   # connect_client(); elicitation + log handlers; no sampling/roots
prometheus/mcp/__init__.py
tests/fastmcp_tools.py     # initials tools + pick_number elicitation test
tests/fastmcp_quotes.py    # quote resource / cache-related tests
labs/.../model_context_protocol_solution.ipynb  # FastMCP 4 lab (reference)
design/main.md             # mermaid architecture
design/choices_week2.txt   # design scratch
requirements.txt           # mcp>=2, fastmcp>=4, pytest, …
```

### Server tools / resources / prompts (today)

- **Tools:** `add_initials_between`, `verify_add_initials_between`, `pick_number` (elicitation demo)
- **Resources:** `config://server`, `note://{topic}`, `market://{symbol}/quote` (stub)
- **Prompts:** `review_code`, `get_market_data` (briefing intent only — no URI names)

### Client API

```python
from prometheus.mcp.client import connect_client
async with connect_client() as c:
    ...
# aliases: client = connect_client
# CLIENT_MODE = "legacy", CLIENT_CACHE = False
```

---

## Tests (last known good)

```bash
conda run -n cmu python -m pytest tests/fastmcp_tools.py tests/fastmcp_quotes.py -q -k 'not forced_failure'
# → 16 passed, 2 skipped, 1 deselected
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

## Next work (suggested order)

1. **Redis-backed quotes** — replace stub in `quote()`; keep `cache=False` / `cache_ttl=None`
2. **Trading HITL** — evolve `pick_number` → real elicit schemas (e.g. confirm size/side) + real UI handler (not auto-24)
3. **LangGraph host** — wire agents above MCP; LLM sampling lives there
4. **More resources** — from `choices_week2.txt` (portfolio, news, risk docs, etc.)
5. Commit when Jacob requests; update `design/main.md` mermaid if Redis lands

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

> Continue Capstone Prometheus MCP from `@design/session_notes.md`.  
> Stack: FastMCP 4, `prometheus/mcp/server.py` + `client.py`, cache=False, elicitation kept, sampling/roots removed.  
> Next: Redis quotes / trading HITL / LangGraph as needed.
