# Design choices

Short records of decisions made while building `DataAgent` and the RAG / LangGraph layer on top of it: what was chosen, why, and what it costs. Entries 1 to 9 are the DataAgent; entries 10 onward are the RAG build (step by step in `labs/module3_rag/hpw_to_do_langchain_rag.md`).

---

## 1. Keyword/trend discovery uses a local market-news store

**Date:** 2026-10-24

**Problem.** Users should be able to search by trend or phrase ("weight loss", "GLP-1") and get
back companies and ticker symbols (for example Eli Lilly, `LLY`), without already knowing the
company. `fetch_news()` cannot do this because both providers need a symbol up front.

**What the providers offer (checked against live responses).**
- **Polygon/Massive `/v2/reference/news`**: the `ticker` filter is optional. Without it the API
  returns market-wide news. Each article carries `tickers`, `keywords` and `insights` (per-ticker
  sentiment plus reasoning). There is no free-text search parameter.
- **Finnhub**: `company-news` requires a symbol. The general/merger feeds (`/news`) need no
  symbol but return only the latest items, and `related` (tickers) is often empty. No free-text search.
- **NewsAPI** has a real `q=` search but is not available (no paid plan), so it is not used.

**Options considered.**
1. *Scan the API on every search.* Simple, but every search would re-download the whole date range
   (about 3,400 articles for 7 days, roughly 500 per day, so about 4 requests per week and about 17
   per month). Cheap now, but it grows with the range and repeats on every search.
2. *Pull once into a local store and search the store.* **Chosen.**
3. *Semantic search over embeddings.* The better long-term answer, but it needs a vector store and
   embedding model and belongs to the later RAG step. The local store is the input it will need.

**Decision.**
- `sync_market_news(from_date)` downloads market-wide Polygon news (every page) and the Finnhub
  general and merger feeds into `{NEWS_OUTPUT_DIR}/market/YYYY-MM-DD.jsonl`, one JSON object per
  article, one file per UTC publish day. The synced range is tracked in `market/_coverage.json`, so
  repeat runs only fetch what is new (with a 1 hour overlap) and articles are de-duplicated by URL.
- `discover(phrase, from_date, top, refresh=True)` runs the sync first (cheap when up to date) and
  then searches only the local files. No API calls happen during the search itself.
- Matching is a whole-word, case-insensitive phrase match over headline, summary, API keywords and
  insight reasoning. Tickers are ranked by number of matching articles; where an article has
  insights, only tickers with their own insight count, so passing mentions do not rank.
- The store is separate from the per-symbol `fetch_news` files because the two have different
  shapes (JSON lines for bulk machine reading versus the splittable text records).
- Progress is shown as a one-line bar (percent, article counts, elapsed time and ETA). Polygon
  results arrive in publish-time order, so the percent is the share of the date range covered.

**Consequences.**
- *Pro:* instant, repeatable searches; the store is reusable as RAG input; fewer API calls.
- *Pro:* the first sync is the only expensive step, and an interrupted sync can be re-run.
- *Con:* disk use grows with the date range, and results are only as fresh as the last sync.
- *Con:* matching is literal. "Weight loss pills" will not find articles that only say "GLP-1" or
  "obesity". Closing that gap (LLM query expansion or embeddings) is future work.
- *Open question:* the Massive plan and terms of service were not checked for limits on bulk
  retrieval or storage. The 10 calls per second pacing is our own conservative choice.

**Bug found while building this.** Polygon's `next_url` cursor does not page reliably: without a
`limit` it falls back to 10 articles per page, and with one it starts repeating the same page (an
earlier run "scanned" 45,670 articles that were mostly repeats; the real 7-day total is about 3,400).
Paging now moves the `published_utc.gte` bound forward 1,000 articles at a time and de-duplicates by
article id.

---

## 2. News bodies come from the APIs, not from scraping

**Date:** 2026-10-24

Scraping article pages returned usable text for about 1 of 89 articles (paywalls, bot blocking,
redirect links), so the scraper and the `<<<BODY>>>` section were removed. The RAG text comes from
API fields instead: the Polygon description, keywords, tickers and per-ticker insights, the
Finnhub summary, related tickers and category. Press-release text (the full release) is handled by
`fetch_press_releases`, see entry 4. The existing BigMessyData news pipeline does the same and never scrapes bodies.

## 3. Both news providers are kept

**Date:** 2026-10-24

Polygon gives the richest data (insights, keywords, tickers). Finnhub adds articles Polygon does not
(about half of the matched Finnhub articles were unique in a test). Press releases are a separate
function (entry 4).
Both are used and de-duplicated by URL.

## 4. Press releases get their own function and file per ticker

**Date:** 2026-10-24

`fetch_press_releases(symbol, from_date=365)` writes `data/press_releases/{SYMBOL}.txt`. It uses the
same Finnhub `/press-releases` endpoint as the BigMessyData `FinnhubSource.fetch_press_releases`,
simplified: synchronous, plain dicts, errors raised instead of swallowed. Reasons for a separate
function and folder: press releases are official, high-signal documents the RAG may want to weight
or retrieve separately from general news, and they need only a symbol and a date range (no company
or keyword filter). The release text is already in the API response, so nothing is scraped.

**Fully separate from news (user requirement).** An agentic tool will call press releases on its
own, so `fetch_press_releases` has its own request, parser, file (`data/press_releases/`) and record
format (`<<<PRESS_RELEASE>>>` / `<<<TEXT>>>` / `<<<END>>>`), and `fetch_news` no longer fetches
press releases at all. The cost is a little duplicated parsing code; the benefit is that changing
either one cannot break the other. The 365-day default is a starting point; Finnhub returned 33
AAPL releases for it, and history limits for longer ranges were not checked.

## 5. SEC filings come from the Massive filings endpoints (no extra account)

**Date:** 2026-10-24

**Goal.** Append 10-K, 10-Q and other SEC filings per ticker for search/RAG:
`fetch_filings(symbol, from_date=365, forms=FILING_FORMS)` writes `data/filings/{SYMBOL}.txt`.
Like press releases it is independent of the news code (own request, parser, file and record format).

**Options checked (live calls with the existing keys).**
- **Massive (Polygon) filings, chosen.** Included in the Stocks plans (free during the early-access beta,
  so the plan or price may change later) and works with the existing key: `/stocks/filings/vX/index`
  (every form type, with SEC URL and accession number), `/stocks/filings/10-K/vX/sections` (clean text of
  Business and Risk Factors) and `/stocks/filings/8-K/vX/text` (Items text of material events). No new account.
- **Finnhub `/stock/filings`.** Works with the existing key, but returns only metadata and SEC links (500
  rows for AAPL, mostly Form 4). Its text-based filings tools (sentiment, similarity index, global filings
  search) are premium. Not used.
- **SEC EDGAR directly (`data.sec.gov`, `sec.gov/Archives`).** Free, no account and no API key, and the only
  source of full 10-Q/10-K text. The SEC requires a `User-Agent` that identifies the requester (name and
  email) and limits requests to 10 per second. Not implemented yet: it needs a contact email from the user.
- **FINRA.** EDGAR holds SEC filings only. FINRA data (for example BrokerCheck, TRACE bond prints) is a
  different system and is not covered by any source above.

**Decision.** One record per filing (`<<<FILING>>>`, `accession`, `symbol`, `form`, `filed`, `period_end`,
`url`, `text_status`, `<<<TEXT>>>`, `<<<END>>>`), appended oldest first and de-duplicated by accession number.
10-K text is Business plus Risk Factors only (not the full annual report) and 8-K text is the Items text.
10-Q has no text endpoint, so it is listed with its SEC URL and `text_status: not_available`. Date-window
paging is used instead of `next_url`; a result set at the page limit raises an error and asks for a smaller
`from_date` (a year of 10-K, 10-Q and 8-K per ticker is far below the limit). The two text endpoints sort on
different columns (`period_end` for 10-K sections, `filing_date` for the others).

Tested result: AAPL, 365 days -> 12 filings (8 8-K and 1 10-K with text, 3 10-Q listed).

**Update (2026-10-24): wider form list, EDGAR direct dropped.**
- Check: META since 2026-07-01 had 66 filings (37 Form 4, 22 Form 144, 2 10-Q, 2 8-K, a PRE 14C filed
  2026-10-21 about the Class B / Zuckerberg control, a 13G/A and a 144/A). Massive and Finnhub list almost the
  same set, so the paid Massive index is complete. The first version missed the PRE 14C because its default
  list was only 10-K, 10-Q and 8-K.
- Default `FILING_FORMS` is now 10-K, 10-Q, 8-K, PRE 14C, DEF 14C, DEF 14A, SCHEDULE 13D, 13D/A, 13G, 13G/A,
  144 and 144/A. Amendments are listed explicitly because the form filter matches exact names. Form 4 is left
  out (about 37 per quarter for META, mostly routine); pass `forms=` to include it. Forms without a text
  endpoint (10-Q, 14C/14A, 13D/G, 144) are stored as SEC link only (`text_status: not_available`).
- Decision: do not use SEC EDGAR direct. The paid Massive subscription is enough, so there is no User-Agent
  or contact email to manage and no second source to maintain. Consequence: no full 10-Q or proxy text; the
  record holds the SEC URL. Revisit only if the full text of those forms becomes needed.
- Tested result: META, 365 days -> 110 filings (87 Form 144, 12 8-K, 4 10-Q, 1 10-K, 1 DEF 14A, 1 PRE 14C,
  3 Schedule 13G, 1 144/A).

## 6. Combined news + filings search, and a Finnhub URL de-duplication fix

**Date:** 2026-10-24

- `DataAgent.search(symbol, terms, from_date=30, refresh=True)` refreshes `fetch_news` (per-ticker, keywords =
  terms) and `fetch_filings`, then searches `news/{SYMBOL}.txt` and `data/filings/{SYMBOL}.txt` and returns
  `{news, filings, filings_in_window}`. A record matches when any term occurs as a whole word. Filings
  without text (10-Q, 14C, 144 ...) match on form and URL only.
- Why per ticker: `discover` searches only the market-wide store (Finnhub general + merger and Polygon), a
  thin sample. The CNBC/Yahoo/Benzinga stories about Zuckerberg and the California billionaire tax were in
  Finnhub's per-ticker META feed (Yahoo, Benzinga) but not in the market store. CNBC itself has few items per
  ticker in Finnhub and none for that story; no extra news source is added.
- Bug fixed: Finnhub article links are redirects of the form `finnhub.io/api/news?id=...`. `_url_key` dropped
  the query, so all of them collapsed to one key and only one Finnhub article was kept per file. `_url_key` now
  keeps the query for finnhub.io. Test: META 7 days -> 19 news items and 1 filing for the Zuckerberg terms
  (before the fix: 2 news). `news/AAPL.txt` and the stored market feed (`news/market`) were written before the
  fix and lack most Finnhub items; re-fetch them to repair.

## 7. CNBC headlines come from public RSS feeds; other sources stay as they are

- **Decision:** `sync_cnbc_news()` reads seven public CNBC RSS feeds (Top News, World, Business, Earnings, Technology, Finance, Investing) and converts each item into the existing news schema when it is read (`api` = `cnbc-rss`, `publisher` = `CNBC`, `category` = feed name). Items go into the market store (`news/market/*.jsonl`), de-duplicated by URL.
- **Why RSS:** it needs no key or account. Alpaca news is Benzinga only and Finnhub carries only a few CNBC items per ticker, so neither covers CNBC stories.
- **Keep what we have (Finnhub, Yahoo, Benzinga):** unchanged.
- **Skipped:** a licensed CNBC data feed (cost and contract).
- **Limits:** RSS gives headline, short description, link, id and time only. There are no tickers, author or image, so those fields are empty. Each feed holds only the latest 30 items, so there is no history: run the sync regularly. Sponsored items are not filtered (the feed does not mark them reliably in the sample).
- **How it is found:** items have no tickers, so `discover` ignores them. `search()` syncs the feeds and matches the search terms against CNBC headline and description text, and lists them with the other news.
- **Terms:** CNBC's terms restrict scraping and repackaging its content. Only headline, short description and link are stored, for internal research. Check the terms before any wider use.
- **Verified:** first sync 210 items fetched, 152 new, 58 duplicates across feeds; a second sync wrote 0. After a full rebuild, `search("META", ["Zuckerberg", "wealth tax", "billionaire tax"], 7)` returned 20 news items (1 from CNBC) and 1 filing. `test_news.py` has a CNBC sync step.

## 8. Monthly performance summary and Financials (top 10 metrics)

**Date:** 2026-10-24

- **`performance_summary(symbol)`:** reads this month's `day_aggs_v1` files (about 320 KB per day, so it is fast) and returns symbol, date range, volume, number of trades, lowest/average/highest price and a normal-distribution (bell curve) built from the daily closes (mean = average of closes, std = sample std dev, 61 points over mean +/- 3 std, plus 1 and 2 sigma ranges). Saved to `data/summaries/<SYMBOL>_<YYYY-MM>.json` for the UI chart. The average is a simple mean of closes, not volume-weighted, and the curve is a normal fit, not the real price distribution.
- **`Financials` class (`agent.financials`):** the 10 most commonly asked metrics from the main bmd `FundamentalTicker`: P/E, EPS, net profit margin, gross margin, operating margin, ROE, debt-to-equity, current ratio, dividend yield, revenue growth. `get(symbol)` returns all of them; each metric also has its own method. One Finnhub `stock/metric` call per request.
- **Decision: use Finnhub's pre-computed fields instead of computing from raw statements.** They are more up to date (Finnhub refreshes them as new filings and prices arrive), and most of the main project's methods already just read these fields. Computing them ourselves would need statement history and would go stale between our own refreshes.
- **Difference from bmd:** debt-to-equity is Finnhub's ratio (0.78), not the x100 percentage the main project computes.
- **Not saved to disk:** `Financials` returns data only; `performance_summary` is the one that writes a file.

## 9. Why keep the large trades and quotes files

- **`day_aggs_v1` and `minute_aggs_v1`** (small) cover charts, summaries and most price questions.
- **`trades_v1` (about 3 GB gz per day) and `quotes_v1` (about 7-11 GB gz per day)** are tick-level data. They are not needed for the current features. They are kept for the planned recommendations agent (after RAG): order-flow and large-trade detection, real bid/ask spread and liquidity, true VWAP, intraday volatility, and slippage/backtest realism that minute bars cannot give.
- **Cost:** about 111 GB on disk for one month. If space is needed, convert to Parquet (`convert_to_parquet`) and delete the CSVs, or keep only the symbols the agent uses. Decide when the recommendations agent is designed; nothing depends on them today.
- **TODO:** chart integration needs these larger trades and quotes files (tick-level charts, spread/volume-at-price). Keep them until that is built.

---

## 10. Vector store: ChromaDB, and the version that works with LangChain

**Date:** 2026-10-24

**Question.** Which ChromaDB version works best with LangChain?

**What decides it.** LangChain never talks to Chroma directly. The `langchain-chroma` package is the bridge, and its metadata says which `chromadb` versions it accepts. So the real choice is the `langchain-chroma` release; pip then picks a matching `chromadb`. Read from PyPI on 2026-10-24:

| `langchain-chroma` | Released | Accepts `chromadb` | LangChain line |
|---|---|---|---|
| 0.2.0 to 0.2.3 | Jan to Apr 2025 | 0.4 to 0.6 (several broken 0.5.x builds excluded) | pre-1.0 |
| 0.2.4 to 0.2.5 | May to Jul 2025 | 1.0.9 or newer | pre-1.0 |
| 0.2.6 | Sep 2025 | 1.0.20 or newer | pre-1.0 |
| 1.0.0 | 17 Oct 2025 | 1.0.20 up to below 2 | 1.x |
| **1.1.0** (latest) | 12 Dec 2025 | **1.3.5 up to below 2** | 1.x |

The table shows two things: Chroma's own 0.x to 1.x jump happened between `langchain-chroma` 0.2.3 and 0.2.4, and the current release pairs with the LangChain 1.x line this project uses. The newest `chromadb` is 1.5.9 (5 May 2026). Its 1.5.3 release was yanked by the authors, so never pin that one.

**Options considered.**
1. `langchain-chroma` 1.1.0 with the newest `chromadb` 1.x it allows. **Chosen.**
2. The old pairing (`chromadb` 0.6, `langchain-chroma` 0.2.3). Rejected: `langchain-chroma` itself moved off that line at 0.2.4 and nothing new is built on it.
3. `langchain_community.vectorstores.Chroma`. Rejected: the pre-1.0 import path. The maintained integration is `langchain-chroma`.
4. `chromadb` alone, without LangChain. Rejected: we would rewrite the retriever and filter plumbing that the LangGraph tools expect.
5. Another store (FAISS, pgvector). Not now: the week 2 design names Chroma for similarity search and Postgres for SQL and full-text search. Revisit pgvector if Chroma's single-process limit (below) hurts.

**Decision.**
- `langchain-chroma>=1.1,<2` and `chromadb>=1.3.5,<2` in `requirements.txt`. The floor is the lowest `chromadb` that `langchain-chroma` 1.1.0 allows. The `<2` caps stop a major upgrade from arriving unannounced.
- Installed and tested on 2026-10-24 (Python 3.13.16, conda env `cmu`): `chromadb` 1.5.9, `langchain-chroma` 1.1.0, `langchain-core` 1.7.0, `langchain` 1.4.7, `langgraph` 1.2.17. `pip check` is clean, and nothing already installed (torch, transformers, numpy, mcp, fastmcp) changed.
- To upgrade later: raise `langchain-chroma` first, let pip choose `chromadb`, then re-run `labs/module3_rag/check_install.py`.
- Embedded mode: `chromadb.PersistentClient`, index in `data/processed/vectorstore` (already git-ignored, and already named in `.env.example`). No server to run.
- **Telemetry off.** Chroma sends anonymous usage events unless told not to (`Settings().anonymized_telemetry` is `True` by default, checked). The client is created with `Settings(anonymized_telemetry=False)`, in one place (`prometheus/rag/store.py`).
- **Cosine distance, chosen at creation.** `collection_configuration={"hnsw": {"space": "cosine"}}`. LangChain picks its relevance formula from the collection's metric. With cosine it is `1 - cosine distance`, a plain similarity (0 for unrelated text). With Chroma's default metric (l2) the scores still work, but unrelated text can score negative (tested: -0.41 on a toy example). Chroma rejects changing the metric once the collection exists (tested), so a different metric means a new collection.
- **The index is derived data.** The DataAgent text files (`news/*.txt`, later `data/filings` and `data/press_releases`) are the source of truth. The Chroma folder can be deleted and rebuilt at any time, so a Chroma upgrade never needs a migration.
- **Re-ingest is safe.** `langchain-chroma` writes with `upsert` (read in its source and confirmed by test: five documents added twice are stored once each), so deterministic chunk ids overwrite instead of duplicating.

**Consequences.**
- *Pro:* no server and one folder. The filters the week 3 design needs work (tested: filter on symbol, filter on date, and both together).
- *Pro:* ranking and scores were identical on GPU and CPU (tested), so results do not depend on which device embeds.
- *Con:* treat embedded mode as single-process. If the FastMCP server and the app both need to write, run Chroma as a server (`chroma run`) and switch the client to `HttpClient`. The LangChain code above the client stays the same.
- *Con:* `chromadb` is heavy. With its dependencies (onnxruntime, grpc, opentelemetry, a Kubernetes client) the install added 60 packages, most of which we never use.
- *Con:* range filters compare numbers only (tested: `$gte` on a text date is rejected), so every chunk carries `published_ts` (epoch seconds) next to the readable `published_utc`.
- *Pro:* list metadata works. An article's `tickers` can stay a list and be filtered with `$contains` or `$not_contains` (tested on 1.5.9, also through `langchain-chroma`), so "news that mentions MSFT" is one filter.
- *Con:* collection names must be 3 to 512 characters from `[a-zA-Z0-9._-]`, so a model name like `sentence-transformers/all-MiniLM-L6-v2` has to be cleaned before it goes into a collection name.

**Correction (2026-10-24, evening).** This entry first said that metadata lists must be flattened to text. I had not tested that, and it is wrong for `chromadb` 1.5.9: lists are accepted and can be filtered with `$contains`. The guide and the Part 3 metadata design were changed to match. It also said that LangChain "needs" the cosine setting to produce relevance scores. That was too strong: scores work with the default metric too, but they can go negative for unrelated text, which is why cosine is still the choice.

## 11. LangChain 1.x and LangGraph 1.x: what is installed and what is left out

**Date:** 2026-10-24

**Decision.** Use only the current 1.x line, each integration in its own package, all capped below 2: `langchain` 1.4.7, `langgraph` 1.2.17, `langchain-core` 1.7.0 (pulled in), `langchain-text-splitters` 1.1.3, `langchain-huggingface` 1.2.3, `langchain-chroma` 1.1.0 (entry 10) and `langchain-openai` 1.7.2.

**Why 1.x only.** Most RAG tutorials online are pre-1.0 (`RetrievalQA`, `langchain.vectorstores`, `langchain_community`), and their imports are the wrong shape for these versions. The reference we follow, LangChain's "Build a custom RAG agent with LangGraph" guide, uses the 1.x paths. The 12 LangChain and LangGraph names it imports all load here (`OpenAIEmbeddings` is replaced by local embeddings), as do the four we add (`Chroma`, `HuggingFaceEmbeddings`, `ChatOpenAI`, `InMemorySaver`).

| Package | Job in Prometheus |
|---|---|
| `langchain-text-splitters` | cut long text into chunks that fit the embedding model |
| `langchain-huggingface` | `HuggingFaceEmbeddings`: runs `sentence-transformers` on this machine |
| `langchain-chroma` | the Chroma vector store and retriever |
| `langgraph` | the graph: state, nodes, edges, `ToolNode`; later checkpointer memory |
| `langchain` | `@tool`, `init_chat_model`, message classes |
| `langchain-openai` | chat-model client for OpenRouter now and a local OpenAI-compatible server later |
| `pyarrow` | imported by `data_agent.py`; was missing from the env |

**Left out on purpose.**
- `langchain-community`: its Chroma and embedding classes are the deprecated pre-1.0 paths.
- `RetrievalQA`-style chains: they retrieve and answer in one step, which merges the Data Clerk and Advisor roles from the week 2 design.
- `beautifulsoup4` and `requests` (the guide's web loader): our text is local files.
- `langgraph-checkpoint-postgres`: later (thread memory for the Account Manager).
- `langchain-mcp-adapters`: cannot be installed next to our FastMCP. It requires `mcp<2`, and `fastmcp` 4.1.1 requires `mcp>=2` (entry 14).
- LangSmith tracing: the `langsmith` package comes with `langchain-core`, but tracing is off unless `LANGSMITH_TRACING=true` is set (checked: off by default). Keep it off. Prompts contain market data and trading logic, and nothing should leave the machine by accident.

**Provisional: chat-model client.** `langchain-openai` pointed at OpenRouter's OpenAI-compatible URL (`OPENROUTER_BASE_URL` in `.env.example`), and later at the local server on the DGX Spark by changing the URL. One client class covers both. The alternative, `langchain-openrouter` (0.2.9), only talks to OpenRouter, so local models would need a second client. Confirm when the chat-model factory is written.

**Two things found while installing.**
- `pyarrow` was not in the `cmu` env, so `import data_agent` failed. It is now in `requirements.txt` (18 is the first release with a Python 3.13 wheel).
- `data_agent.py` sets its base folder to the folder that holds the file, which was the old layout. Create it as `DataAgent(base_dir=<repo root>)`: `news/`, `data/` and `.env` then resolve to the repo root (tested: `_news_dir()` returns `<repo>/news`). No change to `data_agent.py` is needed.

## 12. Where the RAG, tool and agent code lives

**Date:** 2026-10-24 (layout confirmed by Jacob the same day; revised that evening, see the update below)

**Problem.** `prometheus/` already has empty folders for `rag`, `tools`, `langgraph`, `models`, `prompts` and more. The new code needs a home that keeps each piece easy to follow and easy to swap.

**Decision.** Follow the README: one folder per kind of thing, and `agents/` holds the agents, one folder each.

```text
prometheus/
  agents/data_clerk/       the Data Clerk, the one data agent (entry 13)
    data_agent.py            its toolbox: DataAgent, plain Python. Vendored copy, not edited.
    agent.py                 the LLM agent: the agentic-RAG graph (Part 7)
  tools/data_clerk.py      the Data Clerk's tool functions: plain, typed, documented (entry 14)
  mcp/server.py            serves the same functions as MCP tools (Part 8)
  rag/                     RAG library, shared by agents: files -> Documents -> Chroma -> retriever
    config.py                paths and model names (from .env, with defaults)
    loaders.py               news/<SYMBOL>.txt -> LangChain Documents
    store.py                 embeddings + persistent Chroma (the only file that names either)
    ingest.py                load -> split -> embed -> upsert;  python -m prometheus.rag.ingest AAPL
    retriever.py             filtered search -> chunks + scores
    citations.py             numbered sources for the prompt; checks that every citation points at a retrieved chunk (entry 15)
  models/chat.py           chat-model factory (OpenRouter now, local later); llm.py (local HF) stays
  prompts/data_clerk.py    grader, rewrite and answer prompts
  langgraph/desk.py        later: wires the five agents into one graph (Account Manager -> Data Clerk -> ...)
```

Today `agents/data_clerk/` (the toolbox and its test scripts) and `rag/` with `config.py`, `loaders.py`, `store.py` and `ingest.py` exist. The other files are created step by step in the guide.

**Rules behind it.**
1. **One folder per agent under `agents/`,** named after its role in the design: Data Clerk, Advisor, Critic, Account Manager, Trader (the README lists them). The Data Clerk and its toolbox share one folder (entry 13).
2. **`rag/` reads files and never imports `DataAgent`.** The record format in `docs/DataAgent.md` (`<<<ARTICLE>>>` to `<<<END>>>`) is the contract between them, so either side can change alone. Other agents can reuse `rag/` later (for example the Advisor's "similar recommendations").
3. **One module per reason to change.** Loading (file format), the store (embedding model and vector DB) and retrieval (search strategy) change for different reasons, so each is its own module and can be tested alone.
4. **`tools/` holds plain Python functions** (the README calls it "Local Python tools"). Entry 14 says how they reach LangChain and MCP.
5. **`langgraph/` wires agents together.** An agent's own loop, such as the Data Clerk's agentic-RAG graph, lives with the agent.

**Options not taken.**
- *Everything by kind, with the agent graph in `langgraph/data_clerk.py`* (the first version of this entry): the Data Clerk would be spread over four folders, and a folder called `agents/data_agent/` would sit beside the future `advisor/` and `critic/` as if it were a sixth agent.
- *Decorating `DataAgent` methods as LangChain tools in place:* ties the standalone class to LangChain and edits a vendored file.

**Update (2026-10-24, evening).** The first version of this entry kept the folder `agents/data_agent/`, put the agent graph in `langgraph/data_clerk.py` and called the DataAgent "the hands" of the Data Clerk. Three things changed it: the README says `agents/` holds the five agents and `langgraph/` wires agents into a graph (I had not read it when I wrote the first version); the DataAgent turned out to be a toolbox, not a second agent (entry 13); and the tool layer is plain functions in `tools/` (entry 14). The folder `agents/data_agent/` is now `agents/data_clerk/`. If you re-copy `data_agent.py` from the standalone project, copy it there.

**Gotchas.**
- **Run from the repo root** (`python -m prometheus.rag.ingest ...`, `pytest`). The folders `prometheus/langgraph`, `prometheus/langchain` and `prometheus/mcp` have the same names as the libraries they use. That is safe as long as `prometheus/` itself is never on `sys.path`. If it is (`cd prometheus && python ...`, or `PYTHONPATH=prometheus`), `import langgraph` would load our folder instead of the library. (`mcp` has had this property since week 1.)
- `prometheus/agents/` and `prometheus/agents/data_clerk/` have no `__init__.py`. They import as namespace packages (tested: `from prometheus.agents.data_clerk.data_agent import DataAgent`), so none was added.

## 13. One data agent: the Data Clerk. `DataAgent` is its toolbox, not a second agent

**Date:** 2026-10-24

**Question (Jacob).** Should the DataAgent code be merged into the Data Clerk, so there are not two data agents doing the same thing, or is it better to keep them separate?

**What each one is.**
- *Data Clerk:* the agent from the week 2 design. It reads a work order, decides which data action to run (search news, fetch a quote, calculate financials), checks the result and reports back. It is an LLM with tools and a loop. In the repo it exists only in the docs so far (`README.md`, `design/`), so there is no Data Clerk code to merge into.
- *`DataAgent`:* a plain Python class (`data_agent.py`, about 1,100 lines) with no LLM and no loop. It downloads and converts flat files, fetches, syncs and searches news, press releases and filings, and computes financial metrics.

They do not do the same thing: the Data Clerk decides and the DataAgent does. The overlap is only in the names. The class is called an agent, and it sat in `agents/`.

**Options considered.**
1. *Merge the code into one Data Clerk class.* Rejected, reasons below.
2. *Keep two peers,* a DataAgent and a Data Clerk side by side in `agents/`. Rejected: two things named like data agents, and no way to tell which one the Account Manager calls.
3. *One agent, the Data Clerk, with the DataAgent as its toolbox in the same folder.* **Chosen.**

**Why the code is not merged.**
- **Some methods must never reach an LLM.** `init_aws_cli` installs software, `fetch_historical_data` downloads over 100 GB, `convert_to_parquet` is a long batch job and `export_project` runs rsync to any destination. Keeping the toolbox separate means the agent sees only an explicit allow-list (entry 14).
- **Different error policies.** The toolbox raises on any error and stops. An agent has to turn a failed tool call into an observation and decide what to do next (entry 14).
- **The toolbox has to run without an LLM.** Its `main()` and test scripts are batch jobs, and it imports without LangChain.
- **Testability.** All network access and paid API keys (Finnhub, Massive) are inside the toolbox. If the Data Clerk is handed its toolbox, tests can hand it a fake one and run the whole agent with no network, no keys and no cost.
- **It is a vendored copy** from the standalone project. Edits would fork it.

**Decision.**
- The Data Clerk is the only data agent. Everything about it lives in `agents/data_clerk/`: the agent (`agent.py`, Part 7) and the toolbox (`data_agent.py`, unchanged).
- The class keeps the name `DataAgent`. Renaming it would fork the standalone project and break every note that mentions it. In new docs it is called "the toolbox".
- The folder `agents/data_agent/` was renamed `agents/data_clerk/`, so it sits beside the other roles named in the README (Advisor, Critic, Account Manager, Trader).
- The agent receives its toolbox instead of creating it, so tests can pass a fake one.

**Consequences.**
- *Pro:* one name for the data agent everywhere: docs, folder, graph node, prompts.
- *Pro:* what the LLM can touch is a short list we choose, not every public method.
- *Con:* a file called `data_agent.py` sits in the Data Clerk's folder. The docstring of `agent.py` will say that it is the toolbox.
- *Con:* re-copying `data_agent.py` from the standalone project must target `agents/data_clerk/`.

## 14. Data Clerk tools: plain functions, shown to LangChain now and to MCP next

**Date:** 2026-10-24

**Questions (Jacob).** Should the DataAgent code move into MCP as tools? Does LangChain need the tools to be set up a certain way?

**Facts checked today** (installed: `langchain` 1.4.7, `langgraph` 1.2.17, `fastmcp` 4.1.1, `mcp` 2.3.0):
1. **LangChain needs four things from a tool:** a unique name; a description (the docstring: a function without one is rejected); typed arguments, which become the JSON schema the model sees (`parse_docstring=True` also turns the `Args:` section into per-argument descriptions); and a return value the model can read (a dict becomes JSON text). An MCP tool declares the same three things (name, description, input schema), so one plain function can feed both.
2. **A tool's own errors are not handled by default.** In `ToolNode`, a model that forgets an argument, sends wrong types or invents a tool name gets an error message back and can retry. But if the tool itself raises (every DataAgent error does), the exception ends the whole graph run. With `handle_tool_errors=True` it becomes an error message for the model instead.
3. **`ToolNode` cannot be called by hand.** Calling `ToolNode.invoke(...)` directly fails with `Missing required config key` in LangGraph 1.x (older tutorials do it). Inside a compiled graph it works. So: test the plain functions directly, and test `ToolNode` behaviour through a small graph.
4. **`response_format="content_and_artifact"`** lets a tool return short text for the model and the full structured result for the application (tested). That is how retrieved chunk ids and scores can reach the work order without filling the prompt.
5. **LangChain's MCP bridge cannot be installed here.** `langchain-mcp-adapters` 0.3.2 requires `mcp<2`, and `fastmcp` 4.1.1 requires `mcp>=2`. Installing it would downgrade `mcp` from 2.3.0 to 1.30.0 under the working FastMCP server. A hand-made bridge of about 15 lines works instead (tested: the existing `fetch_quote` MCP tool ran through a LangChain `ToolNode`).
6. **`print()` inside an MCP tool is safe on this stack.** I expected the DataAgent's progress bars to corrupt a stdio MCP server. They do not: `mcp` 2.x keeps the protocol on its own channel and diverts stray stdout to stderr (tested, and visible in `mcp/server/stdio.py`). So stdout is not a reason to avoid MCP.

**Options considered.**
1. *Move the DataAgent code into `mcp/server.py`.* Rejected: ties the code to the server module, cannot be tested without MCP, and makes LangChain depend on the missing bridge.
2. *MCP-only tools, with the LangGraph agent consuming them through a bridge.* Matches the week 2 wording ("the Data Clerk calls its own MCP") most literally, and is possible with the hand-made bridge. Rejected for now: it adds a process boundary and a custom adapter before any agent works end to end.
3. *Plain functions in `tools/`, registered with LangChain now and with FastMCP next.* **Chosen.**

**Decision.**
- The tool functions live in `prometheus/tools/data_clerk.py` as plain Python with type hints and a docstring written for the model. One list, `DATA_CLERK_TOOLS`, is the allow-list: only what is on it is visible to the LLM. `init_aws_cli`, `fetch_historical_data`, `convert_to_parquet` and `export_project` are never on it.
- A tool function is a thin wrapper. It normalises its arguments (leading `$`, case), calls the DataAgent or `rag/`, and shapes the result for the model. For example, `performance_summary` returns the figures but not the 61-point bell curve (that is for the UI chart), and `retrieve_news` returns short text plus the chunk ids and scores as the artifact.
- Decorators are applied at the edge, not on the function. LangChain: `tool(fn, parse_docstring=True)` when the agent is built (Parts 6 and 7). FastMCP: `server.tool()(fn)` in `mcp/server.py` (Part 8). The function itself stays plain and testable.
- **Errors.** The toolbox keeps raising. The Data Clerk's `ToolNode` uses a custom `handle_tool_errors` function that returns a short, factual message (for example "fetch_news failed: Finnhub HTTP 429. Do not retry now.") instead of the default "Please fix your mistakes", so the loop reaches the verify and notify steps of the week 2 design.
- **MCP now or later.** The LangGraph Data Clerk calls the functions directly. FastMCP serves the same functions to other hosts (Cursor, other agents), which is the week 2 "its own MCP". If the Data Clerk should consume them through MCP later, the hand-made bridge, or a future `langchain-mcp-adapters` that supports `mcp` 2, does it with no change to the functions.
- **One owner for the index.** Chroma's embedded mode is single-process (entry 10). Whichever process serves retrieval also runs ingestion, or Chroma moves to server mode.

**Consequences.**
- *Pro:* the same function is tested once and works under both LangChain and MCP.
- *Pro:* no new dependency, and no downgrade of `mcp`.
- *Con:* until Part 8 the week 2 design is followed in spirit only: the Data Clerk does not go through MCP.
- *Con:* a tool's schema is generated from its signature in two places (LangChain and FastMCP). A signature change shows up in both, which is intended.

## 15. RAG style: the class-slide pipeline, with the Data Clerk LLM deciding whether to retrieve

**Date:** 2026-10-24

**Decision (Jacob).** Use the RAG pipeline from the class slide "Retrieval-augmented generation". The LLM is the Data Clerk agent, and it decides whether to retrieve.

**The slide, box by box, and where each piece lives.**

| Slide stage | Slide box | In Prometheus | Part |
|---|---|---|---|
| Before questions arrive | Documents | the DataAgent's text files, read by `rag/loaders.py` into LangChain `Document`s | 3 |
| | Chunk text | `RecursiveCharacterTextSplitter` in `rag/ingest.py` | 4 |
| | Encode chunks | `HuggingFaceEmbeddings` (the slide says "encode", LangChain says "embed") | 4 |
| | Store vectors + text | Chroma: vector, text and metadata under one id | 4 |
| For each question | Question | the user's request or the work order | 7 |
| | Encode query | the same embedding model, inside the search call | 5 |
| | Search vectors | Chroma nearest neighbours with metadata filters | 5 |
| | Fetch text by ID | Chroma returns the stored text for the hit ids (see below) | 5 |
| Generate answer | Question + retrieved text | the `retrieve_news` result: numbered sources | 6 |
| | LLM | the Data Clerk, through `models/chat.py` | 6, 7 |
| | Answer + citations | the Data Clerk's final message cites its sources, and the app checks them | 7 |

The slide's caption is the principle we rely on: retrieval changes the evidence in the prompt, and the model weights stay fixed. It is also what the week 3 submission asks us to show (the same work order with and without retrieval).

**Where the LLM decides.** In LangChain's terms (retrieval docs, read today) this is Agentic RAG: "an LLM-powered agent decides when and how to retrieve". The alternative, 2-Step RAG, always retrieves before it generates (high control, low flexibility, one LLM call). We chose the agentic form because the Data Clerk is not only a search box. Its actions in the week 2 design are search news, fetch a quote, search socials and calculate financials, and most requests need no stored text at all. It also has to choose between stored text (retrieve) and fresh data (fetch). So stage 2 of the slide becomes a tool the LLM may call, and stage 3 is the LLM's reply after it reads the tool result.

**Consequences for the build.**
1. **Retrieval is a tool** (`retrieve_news`, entry 14), next to the DataAgent's fetch tools. Stage 1, indexing, stays a batch step.
2. **Each hit carries what a citation needs:** id, text, headline, publisher, date, url and score. "Fetch text by ID" is the slide's way of saying the search returns ids and the text comes from a lookup. Chroma does both in one call, so `retriever.py` returns finished hits. The seam stays, so the text could later come from Postgres by id (the week 2 hybrid memory). `headline` joins the chunk metadata so a citation can show it even for a chunk that does not start the article.
3. **Citations are checked, not trusted.** The tool numbers its sources ([1], [2] ...) in the text the model sees, and the model cites by number, which is easier to copy than an id like `polygon:44a8...`. A deterministic check in `rag/citations.py` confirms that every cited number points at a chunk that was really retrieved, and turns the numbers back into ids and urls.
4. **No supporting text means no answer.** If nothing clears the relevance bar, the tool says so and the Data Clerk must say so too (the week 3 failure mode).
5. **The answer is a summary of evidence, not a trading decision.** The Advisor decides (week 3 submission). The retrieved chunk ids go onto the work order as the tool's artifact.
6. **Grading and query rewriting stay optional.** The first version is the simple loop: the LLM calls tools, reads the results and answers. The grader and rewriter from the LangChain "custom RAG agent" guide would make it a Hybrid form. We add them only if plain retrieval fails on real questions (the rule in `design/week3_notes.md`).

**Costs.** At least two LLM calls for a question that retrieves (decide, then answer), where 2-Step RAG needs one, and latency varies with the number of tool calls. The LLM can also skip retrieval when it should not. The system prompt tells it to retrieve for any question about a company's news, and a test in Part 8 checks that it does.

## 16. Loading the news files into Documents (Part 3)

**Date:** 2026-10-24

**Decision.** `prometheus/rag/loaders.py` reads a DataAgent news file with one plain function, `load_news(path) -> list[Document]`, and `prometheus/rag/config.py` says where the files are. All five proposals in the guide (section 3.3) held up.

- **One article is one Document,** with the article id as the Document id.
- **`page_content`** is the headline, the summary and the insight lines, separated by blank lines, exactly as the file has them, so the text can always be rebuilt from the file.
- **Metadata:** `article_id`, `headline`, `tickers` (a list), `published_utc`, `published_ts` (epoch seconds, a number, because Chroma range filters take numbers only), `publisher`, `api`, `url`. These are the filter and citation fields (entry 15). Left out: `keywords` (140 of 288 articles have them; Part 5 tests adding them to the text), `author`, `image_url`, `category`, `type` and `company`.
- **The file's own symbol is not stored.** It is always in `tickers` (288 of 288). If a record has no tickers it falls back to the file's symbol, and with neither it is an error.
- **Errors stop the load.** A record the DataAgent would not have written (no `<<<END>>>`, a new record inside an unfinished one, an unknown `<<<` marker, a header line without a colon, a missing `id`, `headline`, `published_utc` or `url`, a bad date, text outside a record) raises `ValueError` with the file and line. The DataAgent's own `_read_records` silently drops a record that an interrupted fetch cut short. The loader stops instead, so a missing article is noticed. Breaking the input on purpose (entry 19) found two more cases, now handled: a byte order mark at the start of a file used to stop the load, and a byte that is not UTF-8 gave an error without the file and line. Line 1 is now decoded with `utf-8-sig`, and each line is decoded on its own so that the error can name it.
- **Symbols are validated before they become a file name** (`news_file`): letters, digits, `.` and `-`, at most 10 characters, so `../../.env` is refused. Symbols will come from an LLM's tool call. The vendored DataAgent only strips and uppercases, so the tool wrapper (Part 6) must validate symbols too.
- **Settings:** `NEWS_OUTPUT_DIR` (the DataAgent's own name) from the environment, then `.env`, then `news/` at the repo root. A relative path is resolved against the repo root.

**Finding: Finnhub's `tickers` is only the symbol that was asked for.** For all 148 Finnhub articles `tickers` is exactly `[the file's symbol]`. For Polygon it is the article's real company list (122 of 140 name more than one). So the same Finnhub article read from AAPL.txt and from NVDA.txt has `["AAPL"]` and `["NVDA"]`. That is 3 of the 7 articles that sit in two files; the other 4 are Polygon articles and identical. One row per article by id alone would keep only the last list. **Part 4 rule: when an article is stored again, merge its `tickers` with the stored list (a union) and keep everything else.** A test checks the assumption the rule rests on: copies of an article differ only in `tickers`. The rule is implemented in `ingest.py` (entry 17).

**Options considered.**
1. A `BaseLoader` subclass. Rejected: both LangChain guides use plain functions.
2. Reuse `DataAgent._read_records`. Rejected: it is private, merges summary and insights into one text, drops cut-short records silently, and `rag/` must not import the DataAgent (entry 12).
3. Store the file's symbol as metadata. Rejected: see above.
4. Merge duplicates inside `load_news`. Rejected: the loader reads one file, and the merge needs the row already stored, so it belongs to ingestion.

**Also changed.**
- `pytest.ini` gets `pythonpath = .`. The existing tests found `prometheus` only because `PYTHONPATH` in this shell ends with a colon (set by the Pleora eBUS SDK), which Python reads as the current folder. With `PYTHONPATH` cleared they failed.
- `.env.example` now lists `POLYGON_API_KEY` (the DataAgent reads it; no code reads `MASSIVE_API_KEY`) and documents `NEWS_OUTPUT_DIR`. `HowToRun.txt` has a short RAG section.

**Tests.** `tests/test_rag_loaders.py`: 51 tests (39 when this entry was written, 12 added by entry 19), passing with and without `PYTHONPATH`. `tests/fixtures/news_sample.txt` holds three real records copied verbatim. Five deliberate breaks of the loader were each caught by the tests.

**Consequences.** *Pro:* the file-format contract is enforced and tested in one place, against real records and against every real file when present. *Con:* one bad record stops the whole file. Fix the file or re-fetch it.

## 17. Chunking, encoding and storing the news (Part 4)

**Date:** 2026-10-24

**Decision.** `prometheus/rag/store.py` owns the embedding model and the Chroma collection. `prometheus/rag/ingest.py` chunks, encodes and stores the news, and is safe to run again. Both are written to entries 10, 15 and 16.

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` through `HuggingFaceEmbeddings`, with `normalize_embeddings=True`. It runs on the **CPU by default** (`EMBEDDING_DEVICE=cuda` for the GPU). The scores are the same on both (Part 1), the CPU needs no CUDA start-up, and it leaves the GPU free for a local LLM. NVDA's 386 new chunks took 2.6 seconds on the CPU, lookups included. Settings: `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`, `VECTOR_STORE_PATH` (in `config.py` and `.env.example`).
- **Collection:** `news-<model name>`, for example `news-sentence-transformers-all-MiniLM-L6-v2`. The model is in the name so vectors from two models never meet. Another model means a new collection and a new ingest. Filings and press releases will get their own collections, because they chunk differently. The client is a `PersistentClient` with telemetry off and cosine distance fixed at creation (entry 10), set up in one place.
- **Chunking:** `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, add_start_index=True)`, in characters. Measured on the 288 articles against the embedder's 256-token limit: the tutorial's 1000 / 200 leaves 1 chunk over (263 tokens) and 3 with the headline put back, while 800 / 100 leaves none. The price is about 14% more chunks (557 against 489, duplicates counted twice).
- **Characters, not tokens, plus a check.** Characters are simple and match the tutorial and the week 3 plan. `ingest` counts the chunks the embedder would cut off (`over_limit`, 0 for all the news), so a corpus with more numbers per character, such as filings, cannot be truncated silently. Revisit with a token-based splitter if that check ever fires.
- **The headline goes back in front of any chunk that does not start with it.** Without it a chunk like `CRWV | negative | Stock declined 11.91% ...` does not say which article it belongs to (Lab 3.1 keeps the title on every chunk for the same reason). The headline is cut at 250 characters when it is put in front (the longest real one is 208), because a giant headline would fill the whole chunk and leave the article unfindable (entry 19).
- **Ids and metadata:** chunk id `<article id>#<n>`; the article's metadata (entry 16) plus `start_index`, `chunk` and `chunks`.
- **Safe to run again.** For each article, `ingest` compares the new chunks with those stored. An unchanged chunk (same id, text and metadata) is skipped and not encoded again. A new or different chunk is upserted. A stored chunk that the article no longer produces is deleted. The `tickers` become the union of the stored list and the new one (the rule from entry 16, stored list first). Checked on the real files: a second `--all` changes nothing, and NVDA on top of AAPL and META gives 3 updated chunks (the Finnhub duplicates), 16 unchanged (the Polygon duplicates) and a total of 538 = 557 - 19.
- **Rebuild:** `--rebuild` deletes the collection and ingests again. The index is derived data (entry 10). A full rebuild of the four real files takes about 9 seconds and ends with the same 538 chunks. `--rebuild` deletes only the current model's collection (the name carries the model): after `EMBEDDING_MODEL` is switched, the new model starts an empty collection and the old one stays in the folder until the folder is deleted (checked on a copy of the real index, which was not touched). Guide section 4.8 has the how-to.
- **What re-running cannot undo** (found while writing guide section 4.7; tested on a scratch copy of the fixture, never on the real files or the real index). `ingest` only looks at the articles in the file it reads. So an article taken out of a file stays in the index (the second `ingest` said `unchanged 2, deleted 0` and the store still held 3 articles). A ticker taken off an article stays too (`unchanged 4`, and the ticker was still on 2 chunks), because the `tickers` merge is a union. An article whose `id:` line is edited is stored twice (`added 1`, 4 articles). Edits to an article's text are picked up (`updated 1`). **Decision: removals are handled by `--rebuild --all`, not by pruning.** Pruning is not simple, because an article can sit in two symbol files (entry 16), so "no longer in this file" does not mean "gone", and a rebuild takes seconds. Two cautions, both tested: `--rebuild` without `--all` empties the index and ingests only the named symbols (4 chunks of 3 articles became 2 of 2), and `--rebuild --all` empties the index before the first file is read, so a broken file leaves a partial index until the file is fixed and the rebuild is run again. Check every file with the loader first (guide sections 4.7 and 4.8). A possible later hardening, not done because it adds code for a rare case: load every file before `--rebuild` deletes anything.
- **Tested without a model.** `ingest_documents(docs, store)` takes the store as an argument, so tests use a fake embedder. Three tests use the real model and skip themselves when it is not available.
- **One private attribute.** Counting tokens needs LangChain's inner model (`HuggingFaceEmbeddings._client`, a `SentenceTransformer`). Only `store.count_too_long` touches it, and a test fails if an upgrade removes it.
- **Left as they are on purpose:** raw HTML (entry 18), and `keywords` (Part 5 tests adding them to the text).

**Result on the real files.** AAPL 74 articles to 110 chunks, META 20 to 22, NVDA 181 to 405, TSLA 13 to 20. In total 538 chunks of 281 articles; chunks per article: 1 x 150, 2 x 44, 3 x 58, 4 x 21, 5 x 6, 6 x 2; text length median 530 and longest 963 characters; 0 of 538 over the limit.

**First taste.** "AI memory shortage" (the Case 2 phrase) returns an article whose `tickers` include `MU` and `SKHY`, although it was fetched under NVDA: the "which companies?" answer Case 2 asks for. The same article took two of the top three places (its chunks 1 and 3). Part 5 decides whether to group results by article or use MMR.

**Options considered.**
1. A token-based splitter built on the model's tokenizer. Rejected for now: more code, and characters measured fine on news. See the check above.
2. Delete and re-add everything on every run. Rejected: it encodes every chunk every time, and it would also drop the merged `tickers` of articles stored from another file.
3. One collection per symbol. Rejected (entry 16).
4. The model on the GPU by default. Rejected: same scores, an extra start-up, and a competing user of the GPU.

**Consequences.** *Pro:* re-running is cheap and safe, the stored text always carries what a citation needs, and the model cannot be mixed up. *Con:* the second run of a symbol still takes about 3 seconds, which is the time to load the model, not to encode. Chroma's embedded mode is single-process (entry 10), so ingest and search should run in one process at a time.

## 18. HTML in the news text: measured, left as written, with a tripwire

**Date:** 2026-10-24

**Question.** Text from the news APIs can contain HTML (`Nvidia&#39;s`). Should the loader decode or strip it before the text is chunked and encoded?

**What we measured** (`python labs/module3_rag/break_the_input.py`, on the real files and the real model).
- In the 288 real articles: HTML entities in 3 (two distinct articles, one of them in two files; all `&#39;`, all from Finnhub), HTML tags in 0, a non-breaking space in 1, and no zero-width characters, control characters or mojibake. The longest headline is 208 characters.
- Leaving the entities in barely changes the embedding. The text as written and decoded have a similarity of 0.997 to 0.998, and the score for the article's own headline moved by at most 0.005. The model sees `Nvidia&#39;s` as `n ##vid ##ia & # 39 ; s`, four extra tokens in a text of dozens.
- Put into one real article, the damage to how well it is found: entities -0.015, light tags (`p`, `b`, `a`, `br`) +0.022, `<script>` and `<style>` blocks -0.025, but **markup around every word -0.247**. A little HTML is noise, a lot of it is harm.
- Chroma stores all of it exactly as given (NUL, control characters, emoji and CJK, HTML, `<script>`, a line that starts with `<<<`), so storage does not change the text.

**Options considered.**
1. Leave the text as the DataAgent wrote it. **Chosen.**
2. Decode entities in the loader (`html.unescape`). Rejected for now.
3. Strip tags in the loader. Rejected for now.
4. Clean it at the source, in the DataAgent. Rejected: it is a vendored copy, and a fix there would not cover other writers.

**Decision.** Leave HTML as written, and watch for change with a tripwire.
- *It is rare and cheap.* 1% of the articles, with a measured effect inside the noise.
- *The text stays what the DataAgent wrote,* so the index can always be rebuilt from the file and compared with it (entry 16).
- *Decoding is not always safe.* It turns the escaped text `&lt;script&gt;` into a real tag. How text is shown is the job of whoever shows it: an LLM reads `&#39;` fine, and a web page must escape the text it displays, whatever the loader does.
- *Heavy markup does hurt (-0.247), so we must notice if the sources change.* The test `test_the_tripwire_for_html_in_the_real_text` fails if any real article contains an HTML tag, or if more than 10% contain an entity. It has been shown to fire: with one tag put into a real record it fails with "1 articles contain HTML tags ... see design choices entry 18".

**The fix, decided now so that it is not improvised later.** If the tripwire fires, in `loaders.py::_to_document` run one function over the headline, the summary and each insight line:
1. Decode only fully formed entities (`&#39;`, `&amp;`), one match at a time with `html.unescape`, so that a bare `&copy` in plain text is left alone.
2. Drop `<script>` and `<style>` blocks with their content, then remove tags that start with a letter (`</?[a-zA-Z][^<>]*>`). That keeps `<5%` and `P/E < 20`, but it would also eat something like `x<y and y>z`, which news text almost never contains.
3. Test it with a real record that contains each case.
4. Change the "exactly as the file has them" rule in entry 16, then run `ingest --all`. The reconcile step rewrites the chunks that changed and nothing else.

**Consequences.** *Pro:* nothing to maintain until the data changes, and we are told when it does. *Con:* about 1% of the chunks carry a raw `&#39;`, and any page that displays the text must escape it.

## 19. Breaking the input on purpose: what we tested, what it found, and how problems get fixed

**Date:** 2026-10-24

**Why.** The news files come from outside (APIs and web text), and their text will later be read by an LLM that can place trades. The pipeline has to survive text that is damaged, odd or hostile, not only the clean records we happened to fetch. So each layer was tested by breaking its input on purpose.

**Four kinds of breaking.**
1. *The record structure.* Eleven damaged inputs in `tests/test_rag_loaders.py`: no `<<<END>>>`, a new record inside an unfinished one, an unknown marker, a header line without a colon, a missing `url`, a missing `headline`, a bad date, no tickers and no symbol, stray text before a record, a marker line after a record, and a marker with no escape inside the text. Each must stop with an error that names the file and the line, and nothing is half-loaded.
2. *The text.* One clean article, damaged 15 ways and pushed through the real loader, chunker and embedding model: `python labs/module3_rag/break_the_input.py`. The table below has the results, and the behaviors that matter are now permanent tests.
3. *The real data.* Three real records copied word for word into `tests/fixtures/news_sample.txt`, and checks on every real news file: it loads, the file's symbol is in each article's `tickers`, a duplicate article differs only in `tickers`, every real chunk fits the embedder, and the HTML tripwire (entry 18).
4. *The code.* We broke the code on purpose to prove that the tests can fail (mutation checks): 5 breaks in the loader, 9 in `ingest.py` and `store.py`, and 5 more for the BOM, the line label, the line endings, the headline cap and the warning below. Every one was caught by at least one test, and each file was restored and compared byte for byte.

**The text experiment.** The score is how well the damaged article is still found by "foldable iPhone goes on sale" (0.658 for the clean one).

| What we broke | What happened | Verdict |
|---|---|---|
| HTML entities | loads; score -0.015 | left as written (entry 18) |
| HTML tags (`p`, `b`, `a`, `br`) | loads; score +0.022 | left as written (entry 18) |
| `<script>` and `<style>` blocks | loads; score -0.025 | left as written; an HTML page must escape the text |
| markup around every word | loads; score -0.247 | harmful, so the tripwire watches for it (entry 18) |
| non-breaking and zero-width characters | no change: the tokenizer reads a non-breaking space as a space and drops a zero-width one | nothing to do |
| injected instructions ("Ignore all previous instructions ...") | stored as plain text; score -0.024 | nothing here reads text as instructions; the defense is where text meets a model (Part 7) |
| 2,500 characters without a space | loads; 6 chunks; none over the limit; score unchanged | harmless: the tokenizer turns any word longer than 100 characters into one unknown token |
| 40 hashes of 64 characters (dense text) | loads; 4 of 5 chunks over the limit; score -0.103 | now visible through `over_limit` and a WARNING; a token-based splitter if it ever matters (entry 17) |
| a 38,000 character summary | 53 chunks, none over the limit | fine |
| a 3,000 character headline | **found:** every chunk 3,798 characters, all over the limit, the article unfindable (score -0.034, a drop of 0.691) | **fixed:** the headline put in front of a chunk is cut at 250 characters. Now the longest chunk is 1,048, none is over, and the score is -0.009 |
| an unescaped `<<<END>>>` line in the text | stops: `AAPL.txt:13: text outside a record: 'more text'` | correct as it is: loud, nothing half-loaded |
| the file starts with a byte order mark (BOM) | **found:** stopped with a message about an invisible character, on a good file | **fixed:** line 1 is decoded with `utf-8-sig` |
| Windows line endings (CRLF) | loads | fine |
| one invalid UTF-8 byte | **found:** an error with no file and no line | **fixed:** `AAPL.txt:9: not valid UTF-8 (invalid start byte, at byte 69 of the line)` |
| a NUL character in the text | loads; Chroma stores it exactly | fine |

Three bugs were found (the headline, the BOM and the unlabeled byte error), and one silent but harmful case became loud (the WARNING when chunks are over the limit). The fixes change nothing for good data: after them, `ingest --all` on the real files reported every chunk unchanged (added 0, updated 0, deleted 0) and the collection still held 538 chunks.

**The high-level approach to fixing, as rules.**
1. **Structure problems are loud.** Stop, name the file and the line, never half-load, never repair by guessing. The fix is in the file, or re-fetch it. This is the DataAgent's own policy (entry 16).
2. **Text problems are measured first.** How many articles, and what does it do to how well the article is found? Then decide on the evidence: leave it, normalize it, or reject it. Cosmetic problems that cost less than the noise are left (entry 18). Problems that make an article unfindable or stop the load are fixed.
3. **Fix a problem once, at the lowest layer that sees it:** reading the file in the loader, cutting text in `ingest`, storing in `store`. Not in several places.
4. **Every fix starts with a failing test.** Use a real record when there is one, otherwise the smallest made-up one. Afterwards undo the fix and check that the test fails again (the mutation check).
5. **Make silent harm visible even when it is not fixed:** the `over_limit` count, the WARNING and the HTML tripwire.
6. **Re-running is cheap and safe.** Ingestion reconciles with what is stored (entry 17). After a fix that changes text, run `ingest --all`: changed chunks are rewritten and the stale chunks of the same article are deleted. The regression check is that everything the fix should not touch comes back `unchanged`. A fix that must remove something (an article, a ticker) needs `--rebuild --all`, because re-running cannot remove it (entry 17; this sentence said "stale ones deleted" without that limit until guide section 4.7 tested it).
7. **Text from the web is data, never instructions.** The pipeline stores it and returns it as text and never interprets it. The defenses sit where text meets a model or a person. In Part 7 the retrieved text is delimited in the prompt and the model is told to treat it as data, the LLM can only call the tools on the allow-list (entry 14), symbols are validated (entry 16) and citations are checked (entry 15). Any HTML page that shows the text must escape it.

**Not tested yet (an honest list).** An LLM reading injected text (Part 7 adds a test with injected instructions inside a retrieved chunk). Text shown in an HTML page. The market-wide JSON lines files (there is no loader yet). Another embedding model's tokenizer (the 100-character rule is BERT's). Files of hundreds of megabytes. Two processes writing the Chroma folder at once.
