# AGENTS.md

Prometheus: a multi-agent stock trading desk (CMU Agentic AI capstone). Jacob is the developer and reviews every change. Start with `README.md` (layout), `HowToRun.txt` (commands) and `design/jpyrett_capstone_tradingbot.pdf` (the design). Do not copy them here.

## Run things

- Python is the `cmu` conda env: `/home/jpyrett/miniconda3/envs/cmu/bin/python`. The system `python` does not exist and `python3` has no pytest.
- Tests (from the repo root): `~/miniconda3/envs/cmu/bin/python -m pytest -q --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo`. That one test fails on purpose.
- New dependency: add it to `requirements.txt` too (direct dependencies only, never a `pip freeze`).

## Where things are

- `prometheus/langgraph/desk.py` is the top-level LangGraph: `account_manager -> data_clerk -> advisor`, one work order as its state.
- `prometheus/agents/data_clerk/data_clerk_agent.py` is the Data Clerk agent (`DataClerk`: LLM, tools, a loop, a budget, a citation check). Its call for the other agents is `request_data(work_order)`.
- `prometheus/agents/data_clerk/data_toolbox.py` is the toolbox it uses (`DataToolbox`): plain Python, no LLM. It is not an agent.
- `prometheus/agents/account_manager/account_manager.py` is the Account Manager agent (`AccountManager`, an LLM like the Advisor): `raise_funds` is the Tree of Thought for what to sell to buy a stock; the tree is `prometheus/reasoning/raise_funds.py`; the sample portfolio is `configs/portfolios/jacob_pyrett.json`; the record of how it was built is `design/week4_tot_guide.md`. Its prices come from `DataClerk.request_quotes`.
- `prometheus/tools/data_clerk.py` wraps the toolbox as tools; `prometheus/prompts/` holds the prompts; `prometheus/common/work_order.py` is the work order.
- `labs/module3_rag/` has the RAG guide and `advisor_with_without.py`. One-off lab scripts are in `labs/module3_rag/other/`. `labs/` is course reference; port code into `prometheus/`, do not grow it.

## Never without asking Jacob

- Run anything that calls a paid API or writes the news files: `--allow-fetch`, a symbol run of `test_prometheus.sh`, `e2e_symbol.py`, `compare_models.py`, `agent_reliability.py`, `injection_check.py`, `with_without_retrieval.py`, and a real run of `prometheus.agents.account_manager.account_manager` (Finnhub quotes and OpenRouter calls). `scripts/raise_funds_offline_demo.py` uses fakes and is safe. Tests use fakes and are always safe, except the ones marked `live` (`tests/test_raise_funds_complex_live.py`), which call the paid APIs and are skipped unless `PROMETHEUS_LIVE_TESTS=1`: never set it without Jacob's yes.
- Fetch filings, or the text of a 10-Q: `fetch_filings` (paid Massive) and `DataToolbox.fetch_filing_text` (free, but it downloads MBs from sec.gov with Jacob's contact in the User-Agent and appends to `data/filings/`). `SEC_USER_AGENT` lives in `.env`, which you never edit.
- Read, print or edit `.env` (it holds keys), or touch `data/historical/` (26 GB, re-downloadable) or `data/processed/` (saved evidence the docs cite).
- Delete files, or change `requirements.txt` versions.

## Rules for changes

- Keep a rename or a move free of behaviour changes. Update every import, test and doc in the same change: `docs/design_choices.md`, `HowToRun.txt`, `README.md`, `design/week3_e2e.md`, `labs/module3_rag/how_to_do_langchain_rag.md`.
- A design decision gets a new numbered entry at the end of `docs/design_choices.md` (problem, decision, alternatives, consequences). Do not rewrite an old entry to look as if it had always said the new thing.
- Keep an LLM out of anything a program can check: scores, citations and budgets are calculated in code (see entries 24 and 26).
- Leave no `__pycache__` or `.pytest_cache` behind. Keep `.md` and `.pdf` files.
