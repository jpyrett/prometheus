# Session notes: Prometheus handoff (read this first in a new chat)

**Updated:** 2026-11-04 (section 0: Module 4, the Tree of Thought RaiseFunds, is the latest work); the rest of the file is 2026-10-29 (end of the Week 4 work, the data sources: press releases with a real run, then filings built and tested; on 2026-10-30 the 10-Q text of AAPL (3) and META (4) was fetched into `data/filings` and the evidence tool run on it; the index and the desk run are **not yet done**)
**Project:** Prometheus, a multi-agent stock trading desk (CMU Agentic AI capstone). Jacob is the developer and reviews every change.
**Repo:** `/run/media/jpyrett/980pro/CMU-AgenticAI/capstone` (branch `dev`; last commit `bb8f06b`, 2026-10-25; **everything since is uncommitted**, see section 9).

Open this file in the next chat (`@design/session_notes.md`). `AGENTS.md` is read automatically.

---

## 0. LATEST (2026-11-04): Module 4, Tree of Thought: RaiseFunds (read this before anything else about the Account Manager)

**What it is.** The Account Manager decides what to sell to pay for a purchase, by a Tree of Thought (`prometheus/reasoning/raise_funds.py`, a LangGraph; the LLM agent is `prometheus/agents/account_manager/account_manager.py`). The LLM proposes sell steps; code checks every step (only holdings in profit, one share of every holding kept, never the stock being bought, T+1, lean), calculates every score, prunes (beam) and picks the answer. Prices come from the Data Clerk (`get_quote`, Finnhub `/quote`). Portfolios are JSON in `configs/portfolios/` (`cash_avail` is free settled cash).

**Read, in this order:** `design/week4_submission.md` (the Module 4 answers, concise; this is what is handed in), `design/week4_tot_guide.md` (sections 9m to 9r are the latest; 9a to 9l are history on slow routing), `docs/design_choices.md` entries 42 to 50 (49 depth, 50 speed and defaults). Do not read the whole design log.

**The answers Jacob asked for (2026-11-04), all measured live with gpt-oss-120b:**
- *BFS or DFS?* **Beam search, width 3** (levels in parallel). Depth first (beam 1) found 86.6% of the brute-force best plan on the hard purchase (3 LLY), beam 3 found 99.4%, beam 6 and beam 10/branching 4 the same as 3 for 1.7 to 2.8 times the calls. Same session time. Easy purchases (9 COST): every width finds the same plan.
- *Why was it slow?* The model's speed. The same gpt-oss-120b runs at ~55 tokens/s on OpenRouter's own choice of provider, 650 on Cerebras, 370 on Groq. The Account Manager now sets `provider.order = [cerebras, groq]` (fallback on): 9 COST 33 to 47 s -> 5 s; hard session 67 to 72 s -> 8 s; the four-purchase "claude" session 115 to 165 s -> 10 to 14 s.
- *Reasoning effort?* **Low.** Same plan at medium (8 s) and high (46 s) on fast providers; on slow routing high ran over 5 minutes. Not needed.
- *Wider or deeper?* Deeper: yes, automatically (entry 49: depth = fewest stocks that could pay + 2, at least 3). Wider: no (beam 3 is enough).
- *Whole plans as thoughts* (`--whole-plans`): tried, slower (121 s), lower score; option, off.
- *The auto-depth question* is answered yes and built (entry 49, guide 9q).

**Defaults now** (`AccountManager()` and the CLI): strict rules, green-first, explore, lean, parallel, "up to N", effort low and Cerebras/Groq for gpt-oss, 90 s limit per purchase (`--deadline 0` none). `AccountManager.classic()` is the old behaviour; the tests of entries 42 to 47 use it. Off switches are in `HowToRun.txt` 8h.

**State.** Tests: **1681 passed, 38 skipped** (the live ones; `PROMETHEUS_LIVE_TESTS=1` runs them: 30 passed in 131 s on 2026-11-04, paid, a few cents). Three Data Clerk tests fail and are NOT from this work (`test_data_clerk_tools.py` x2 on the press release and filing date windows, `test_work_order.py::test_a_term_order_is_swept...`); they fail on today's date or data, not looked at. `tests/fastmcp_tools.py::test_forced_failure_known_demo` fails on purpose. Nothing is committed (branch `dev`).

**Open (for Jacob):**
1. Review `design/week4_submission.md`; it is written for him to edit. The rubric text in each section is his.
2. The best LLY plan (87.1) needs PG 3 and GOOGL 2; the tree finds 86.6 (99.4%). A better generator prompt, not a wider beam, would close it.
3. One run per setting is a sample. A second hard scenario would make the beam comparison stronger.
4. Cerebras and Groq are the preferred hosts; if both are down the speed falls back to 33 to 70 s a purchase and the 90 s limit binds. The provider sees the prompt (prices, holdings, quantities; no keys).
5. Whole plans were not re-measured on the fast providers.
6. The Account Manager only proposes a plan; nothing is sold or bought (the Trader does that). Sale proceeds T+1, fees and tax rates are not modelled.

**Commands that cost money (ask first, AGENTS.md):** `scripts/raise_funds_diag.py`, `scripts/raise_funds_beam_experiment.py`, the Account Manager CLI, the two live tests. Free: the tests, `scripts/raise_funds_offline_demo.py`, `scripts/raise_funds_oracle.py` (brute-force best plans, 2 s).

**Quick paste for the next chat on this topic:**
> Continue Capstone Prometheus, Module 4 (Tree of Thought, RaiseFunds). Read `AGENTS.md`, then `@design/session_notes.md` section 0, then `@design/week4_submission.md` and `@design/week4_tot_guide.md` (9m to 9r). The tree is fast (1 to 6 s a purchase) with beam search width 3 on Cerebras/Groq; defaults are the measured winners; do not run paid commands without asking me. <say what is next: edit the submission / a second hard scenario / a better generator prompt / wire RaiseFunds into the desk by default>.

---

## 1. Read order for a new chat

1. `AGENTS.md`: where things are, what never to run unasked, the rules for changes (short).
2. This file.
3. `design/week4.md`: the two features in full (decisions with alternatives, every step, open items). Part 1 (sections 1 to 10) is press releases with the real run; **part 2 (sections 11 to 21) is filings**, and its section 21 is the recipe of section 5 below as corrected by its second use. **It is the model for how to document the next feature** (`design/week5.md`).
4. `docs/design_choices.md`: 35 numbered entries, 210 KB. Do **not** read it whole. Read the entry you need: 24 (the Data Clerk agent), 25 (MCP server and work order), 26 (the Advisor), 29 (the desk and script), 30 (guardrails), 31 (company name lookup), 34 (press releases), 35 (filings).
5. `HowToRun.txt`: every command, with what is free and what costs money (section 8f is press releases, 8g is filings).
6. `README.md` (layout), `design/week3_e2e.md` (diagrams; Week 4 additions are yellow), `labs/module3_rag/how_to_do_langchain_rag.md` (the guide to Parts 3 to 9; it has no Week 4 part, `week4.md` is that record).

---

## 2. What exists (state at handoff)

```
START -> account_manager -> data_clerk -> advisor -> END         prometheus/langgraph/desk.py (one work order is the state)
```

| Piece | Where | State |
|---|---|---|
| Account Manager | node in `desk.py` (`order_from_request`, `verify_request`) | **code, not an LLM**; builds the order from symbol, term, source (`news`, `press_releases`, `filings`), company, days |
| Data Clerk | `agents/data_clerk/data_clerk_agent.py` (`DataClerk.request_data(order)`) | LLM + 11 tools + LangGraph loop + budget + citation check; for press releases and filings it first resolves the symbol or company **in code** |
| Filing evidence | `agents/data_clerk/filing_signals.py` + the tool `fetch_filings_recommendation(symbol, since_days)` in `tools/data_clerk.py` | **built (entry 38), scores nothing**: from the STORED filings, the latest balance sheet for both periods, capital actions, income headlines and exact quotes on dilution, executive compensation (negative-type) and buybacks, dividends (positive-type), with gaps; the model reads a digest, the whole result is the artifact. **Attached to the work order since entry 39** (`data.filing_evidence`, per symbol); **read by the Advisor since entry 40** (a signal may cite an evidence item "E3"; code checks the item, the quote and the direction its type fixes; the score is still calculated from the counted signals), and read by CODE for every order with one symbol (the sweep) if the model did not call the tool. The real stored files got their 10-Q text on 2026-10-30 (AAPL 3, META 4), so on them it gives the full balance sheet; a company without fetched 10-Q text gives 8-K passages and a "no balance sheet" gap (`week4.md` section 43) |
| Toolbox | `agents/data_clerk/data_toolbox.py` (`DataToolbox`) | plain Python, no LLM: fetch news, press releases, filings; financials; price summary |
| Tools (the allow-list) | `tools/data_clerk.py` (`DATA_CLERK_TOOLS`) | `retrieve_news`, `retrieve_press_releases`, `retrieve_filings`, `fetch_news`, `fetch_press_releases`, `fetch_filings`, `keyword_search`, `discover_tickers`, `get_financials`, `get_price_summary` |
| RAG | `rag/` (`loaders`, `store`, `ingest`, `retriever`, `citations`, `config`) | three Chroma collections in one folder: `news-...`, `press_releases-...` and `filings-...` (`kind` argument); MiniLM on CPU. Filings: one Document per section, one short Document for a filing with no text (`week4.md` part 2) |
| Symbol lookup | `tools/symbols.py` | `resolve_company` (the desk's symbol field, news) and `resolve_target` (press releases **and filings**: symbol first, then the company name matched on its words in two tiers, never a typo) |
| Work order | `common/work_order.py` | `source` is `news`, `press_releases` or `filings` (`LOOKUP_SOURCES`); `action_type` `search_news`, or `search_filing` for filings only (`ACTION_SOURCE`, checked both ways); fields `subject`, `company`, `resolved`; `verify_request`, `verify_action` |
| Advisor v0 | `agents/advisor/advisor.py` | model proposes signals (exact quotes), **code** verifies them and calculates the score |
| MCP server | `mcp/data_clerk_server.py` | the same tool methods; 8 tools by default, fetch tools only with `--allow-fetch` |
| Main script | `./test_prometheus.sh` | prompts Symbol, Term, Source, (Company name: press releases and filings only), days; asks before any paid fetch |

**Not built:** the Critic, the Trader, an LLM Account Manager, the work-order actions `fetch_quote` / `calculate_financials` / `search_socials` (`verify_request` still refuses them), Redis for quotes, `social` as a searchable source, memory, a graph checkpointer. **Built but never run for real:** filings (the real file is not indexed, no real desk run: `week4.md` section 19).

**Numbers:** tests **1380 passed, 8 skipped** offline (1353 before the sweep and the Advisor reading the evidence, 1342 before the results on the order, 1258 before the evidence tool, 1248 before `search_filing`, 1188 before the 10-Q text, 1003 before filings, 728 before Week 4; the 8 need the real embedding model: `HF_HUB_OFFLINE=1` with the model cached runs them, **1388 passed**). News collection **1,938 chunks**; press release collection **33 chunks** (AAPL only). `data/press_releases/AAPL_PRs.txt` = 33 real releases (byte-identical to the older `AAPL.txt`, which is untouched). Design entries: 40 (40 = every order carries all three sources by a code sweep after the model, and the Advisor reads `data.results` and `data.filing_evidence`; 39 = the 10-Q text from EDGAR, 37 = the action type `search_filing`, 38 = the filing evidence tool `fetch_filings_recommendation`, 39 = the results of each source on the work order). **Filings index:** AAPL was ingested on 2026-10-30 (58 Documents, 610 chunks) and a real desk run was made (`week4.md` section 55, `data/processed/desk_AAPL_filings_nofetch.json`; the run with the paid fetch, `week4.md` section 56, overwrote `desk_AAPL_filings.json`); META was ingested the same day (143 Documents, 3,326 chunks; the collection holds 3,936; `week4.md` section 57); no META desk run yet.

---

## 3. Run things

```bash
cd /run/media/jpyrett/980pro/CMU-AgenticAI/capstone
PY=~/miniconda3/envs/cmu/bin/python         # the cmu conda env; the system python does not exist
$PY -m pytest -q --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo     # that one test fails on purpose
find prometheus tests labs -name __pycache__ -type d -prune -exec rm -rf {} +          # after a test run: the MCP stdio test leaves these
$PY -m prometheus.tools.data_clerk                     # the 11 tools as the model sees them (free)
$PY -m prometheus.langgraph.desk --show-graph          # the desk graph (free)
$PY -m prometheus.rag.loaders AAPL --press-releases    # what a press release file becomes (free)
$PY -m prometheus.rag.loaders AAPL --filings           # what a filings file becomes (free): 28 Documents for AAPL
PROMETHEUS_DRY_RUN=1 ./test_prometheus.sh --skip-inspect --source press_releases --company "Apple Inc" --yes   # prints the command, runs nothing
```

Costs money or writes files (**ask Jacob first**): any real `./test_prometheus.sh` or `desk` run with a symbol, `--allow-fetch`, the lab scripts listed in `AGENTS.md`, the Finnhub symbol lookup (`tools.symbols --exact`). A full desk run is about 30 to 90 seconds (two models). Roles' models come from `.env` (names in `.env.example`); never read or print `.env`.

---

## 4. Working rules (Jacob's, `AGENTS.md`, and what this session taught)

**Jacob's:** he wants to be told what is being done while it is done, and wants the work to be a learning exercise on LangChain/LangGraph. Update the week's `design/weekN.md` **as you go**. Do not make up test data: copy real records. Keep it simple; do not assume, ask when information is missing; any error stops the operation. Do not commit unless asked. He reviews every change.

**From `AGENTS.md`:** keep an LLM out of anything a program can check (scores, citations, budgets, and now the symbol/company lookup are code). A design decision gets a new numbered entry at the end of `docs/design_choices.md` (problem, decision, alternatives, consequences); never rewrite an old entry to look as if it had always said the new thing. Keep a rename free of behaviour changes. Leave no `__pycache__`. Never delete files or change `requirements.txt` versions without asking.

**Lessons from Week 4 (each cost something):**
- **A real run finds what fakes cannot.** Tests passed on made-up names and the real directory proved them wrong three times (names carry full stops, the search takes 20 characters, the model leaves the company out of its query). After the tests, **ask, then do one real run, and read the output against the stored text.**
- **Run the suite once with the network on.** Eight tests are skipped offline; one of them found a bug in code I had just written.
- **A prompt-level fix is measured, not guaranteed.** Measure before and after (6 runs each was done; say that n is small) and report a result that did not improve.
- **Put a number on "the tests are meaningful":** break the code one way at a time and confirm a test fails (22 of 22 for the code, 12 of 12 for the script; an `except` of a subclass is an equivalent mutant).
- **Write documents with the editor tool, never through a shell heredoc.** A heredoc that ended early made the shell read `->` as a redirection and emptied `data/press_releases/AAPL_PRs.txt` (restored by re-fetching, hash verified). Also: scripted test turns need their own message objects (LangGraph's `add_messages` replaces a repeated one); a new parameter must reach the test fakes (`lambda kind="news": ...`).
- **The auto-reviewer blocks** unseen helper scripts in `/tmp` and opening a browser. Use inline scripts (visible in the command) and the repo's own tools.
- A fluent answer is not a verified one: check numbers in an answer against the stored text; the Data Clerk's known stretch of a source (guide 8.10) still happens.

---

## 5. Recipe: add a data source (the press release pattern, in order)

Do these in order and test each step; every one has a real example in `week4.md` section 5. The second use of the recipe (filings) is `week4.md` part 2 (sections 11 to 21), and its section 21 lists what the recipe should have said (one shared parser, the data decides the unit of a Document, change every "not wired" pin in the same step, a fake embedder with repeated vectors makes the search flaky).

1. **Read what exists first.** The toolbox may already fetch it (press releases and filings both did), and a real file may already be in `data/` to build the fixture from.
2. **Decide the corpus, in a design note:** its own Chroma collection (not the news one; a `source` field would force a re-ingest and change news results), the same `Document` metadata keys as news (`article_id, headline, tickers, published_utc, published_ts, publisher, api, url`) so chunking, search, `Hit`, `SourceBook` and the Advisor work unchanged, and an id prefix (`pr:`) so ids never collide in one `SourceBook`.
3. **Config and toolbox:** a `*_dir()` setting in `rag/config.py` (a function, not a constant), the toolbox writes `<SYMBOL>_<SUFFIX>.txt` into it, appended, de-duplicated by id. The record format is the contract with the loader.
4. **Loader** in `rag/loaders.py` (a separate parser, so one format cannot break the other), with errors that name file and line; a CLI flag to look at it.
5. **Store `kind`** (`rag/store.py`), **ingest** (`ingest_<kind>`, the same `ingest_documents`), **search** (a thin wrapper over `search_news` with the other store), CLI flags.
6. **Tool** in `tools/data_clerk.py`: a method whose docstring is the model's instructions (LangChain builds the schema from name, type hints, docstring); one shared `SourceBook`; the fetch tool also indexes; add it to `DATA_CLERK_TOOLS`, `RETRIEVE_TOOLS`, the MCP `ANNOTATIONS`.
7. **Agent:** budget entry, `Budget.observe` (text from the web counts as read: a fetch is refused afterwards), one bullet in the system prompt, the question hints in `request_data`.
8. **Work order:** `WIRED_SOURCES`, `MAX_DAYS`, `SEARCH_TOOL`, `verify_request` and `verify_action` per source.
9. **Desk and script:** `order_from_request`, `--source`, days default and limit; the script's prompts **without changing the news order or command** (its tests pin them).
10. **Tests:** a fixture of **real records copied byte for byte by a script**, never typed; loader (broken input names the line), collection isolation, ingest twice (nothing changes), search, tool, agent (scripted model; include the **poisoned text cannot unlock a fetch** test), work order, desk, MCP, script. Update the counts in the docs.
11. **Mutation check, then a real run** (ask first), and read the answer against the stored text. Record every surprise.
12. **Docs:** `design/weekN.md` as you go, a new numbered design entry, `HowToRun.txt` (counts and a section), `README.md`, `design/week3_e2e.md` diagrams (new boxes in yellow), this file.

---

## 6. Candidate next features

| Feature | Why now | First step |
|---|---|---|
| **Run filings for real** (code and tests are done, 10-Q text from EDGAR included) | the part of the recipe not done (section 5, step 11): the real index, search and desk run, read against the stored text (the 10-Q text itself was fetched into `data/filings` for AAPL and META on 2026-10-30, `week4.md` section 43); it will show whether a 10-K crowds the 8-Ks out of the search | `week4.md` sections 19 and 29; **ask Jacob first** (it writes `data/filings` and `data/processed/vectorstore`, calls the models, and the live Massive fetch is paid) |
| ~~Filings as a source~~ | **built** (`week4.md` part 2, entry 35): by symbol and/or company, the Account Manager passes it, one Document per section, no-text filings kept | |
| ~~Let the Advisor use the results on the order~~ | **built** (`week4.md` part 7, entry 40): it reads `data.results` and `data.filing_evidence`; every order carries all three sources through a code sweep after the model. **Not measured with the real model**: a before/after run of the Advisor (paid, ask first) and a look at how often the sweep finds press releases and filings under the 0.40 bar | `./test_prometheus.sh --source filings` on an indexed company, then compare with and without `data.filing_evidence` (`Advisor.advise` reads whatever the order carries) |
| **The Critic** | (Jacob chose this, 2026-10-30, for the evidence signals whose direction code forces by type: `week4.md` section 55) the Advisor counts a headline as a bullish signal and a company's own quotes about itself as evidence (all three signals in the real press release run were headlines); the check verifies a quote is real and names the company, **not its direction** | decide the check: direction from the quote, or a second model, with the rule that code verifies what code can |
| **Other work-order actions** (`calculate_financials`, `fetch_quote`) | design entry 33 "Next": `get_financials` and `get_price_summary` exist as tools | a `request_data` path that calls them for an order |
| **Social media** (`search_socials`) | `data/social_media/` exists (not examined) | look at what is in it |
| Smaller: see section 7 | | |

---

## 7. Decisions waiting for Jacob, and known gaps

**Waiting for Jacob (each is one line of code, listed in `week4.md` section 10 and, for filings, section 20):**
1. **The company-name match.** *Settled by Jacob (2026-10-29):* a typo is not matched (`Proctor & Gamble`, `Pepsi` for `Pepsico`), and the search may be more flexible if nothing breaks. *Done:* tier 1 = the same words (case, spacing, punctuation ignored); tier 2, only if tier 1 finds nothing = the same words without the generic company words (`Inc, Corp, Corporation, Co, Company, Ltd, Limited, plc, Incorporated, The, and`), so `Pepsico`, `Microsoft Corp`, `Apple` are found; unique or refused, never a guess (`week4.md` 9.10). *Still my reading:* punctuation is not a difference. *Not checked against the live Finnhub since tier 2* (a paid call; ask first): try `Pepsico`, `Microsoft Corp`, `Apple`, `Proctor & Gamble`.
2. **Asking by company name alone at the script's prompts** needs `--company` today (a blank symbol and term are asked again before the source is known). Asking the source first would fix it and shift every news answer by one line.
3. **Wire items in news:** 20 stored news articles are other organisations' releases from `GlobeNewswire` (Polygon). A `publisher` filter would keep them out of news search; not built (changes news results).
4. **The 0.40 relevance bar** was tuned on news; for press release teasers `new product launch` scored 0.39 and found nothing.
5. Delete the old `data/press_releases/AAPL.txt` (byte-identical to `AAPL_PRs.txt`; nothing reads it).
6. **Commit** (nothing since 2026-10-25).
7. **Filings, your call (`week4.md` section 20):** the real run (above); whether to add a `forms` filter (88 of META's 110 filings are Form 144) and a way to stop a 10-K crowding the 8-Ks out of the closest chunks; ~~whether to add `search_filing` to `ACTION_TYPES`~~ (done, entry 37 and `week4.md` part 4: a filings order's action is `search_filing`, checked against the source both ways; news and press releases keep `search_news`); ~~fetching the 10-Q's text from EDGAR~~ (done, entry 36: inside `fetch_filings`; Jacob's contact is the `SEC_USER_AGENT` value, **he has to add it to `.env`, an agent must not edit that file**; only one real 10-Q layout, Apple's, has been seen).

**Known gaps (not fixed):** the Advisor's direction-by-type rule for evidence signals can override the model's own judgement (stance bearish, model score 30, calculated score 60 on the real AAPL run: `TODOS.txt`, `week4.md` section 55; left for the Critic); typo-tolerant company-name matching (a TODO in `find_company`, only if wanted: a typo is not matched today, by Jacob's definition); the Advisor's direction is unchecked (see the Critic); the citation check ignores a range such as `[8-10]`; a live order's "fetch first" is an instruction to the model, not verified by code; the desk names a saved order by symbol, term and source only, so a later run overwrites an earlier one; only AAPL has press releases indexed; only `Common Stock` rows count in the symbol directory (an ETF, REIT or ADR is "not found"); Finnhub's press release text is a ~360 character teaser, so retrieval is on headline plus teaser; `design/week3_e2e_diagrams.png` shows the Week 3 diagrams until rendered again (no Mermaid renderer or Node.js here; the diagrams were checked with a structural linter, not rendered).

---

## 8. Facts about the data sources (learned the hard way)

- **Finnhub `/search`:** a query over **20 characters** is HTTP 422 `q too long`; it is **punctuation-sensitive** (`Meta Platforms Inc` finds nothing, `Meta Platforms, Inc.` finds META); real names are `Apple Inc.`, `Microsoft Corporation`, `The Procter & Gamble Company`. `tools/symbols.py` builds the query from the leading words that fit and matches exactly on what comes back.
- **Finnhub `/press-releases`:** returned the same 33 AAPL releases for 365 days on two different days (ids are stable: sha1 of symbol, time and headline, 16 hex characters); the text is a teaser; URLs are nasdaq.com; no publisher field.
- **News vs press releases are separate by design** (entry 4): `fetch_news` never calls the press release endpoint; none of the 33 Apple releases is in `news/`.
- **EDGAR (one real download, 2026-10-29):** the url stored in a filing record is the COMPLETE submission (AAPL's 10-Q: 5.2 MB, 60 documents; the 10-Q is document 1, an inline-XBRL page, 766,000 characters of which 57,843 are text). A User-Agent with a contact is required; 10 requests a second at most. Fixture: `tests/fixtures/edgar_aapl_10q_0000320193-26-000006.txt.gz` (a byte-exact prefix of the real file).
- **Massive (Polygon) filings** (counted by script on the real files): only 10-K (Business, Risk Factors) and 8-K (items) have text; 10-Q, DEF 14A, Schedule 13G, Form 144 come with form, date and link only (AAPL 18 of 27, META 97 of 110). `filed` is a date, no time. A 10-K is 84,000 (AAPL) to 220,000 (META) characters, 28 to 348 chunks a section. `text_status: ok` exactly when there is text, in both files.
- Polygon `next_url` paging is broken (use time-window paging); NewsAPI is not available (no paid plan); `docs/DataToolbox.md` has the file formats.

---

## 9. Repo state and stale references

- Branch `dev`, last commit `bb8f06b` (2026-10-25). Uncommitted: about 36 modified, 15 renamed (the `labs/module3_rag/other/` move), 1 deleted (`labs/module3_rag/hpw_to_do_langchain_rag.md`), and untracked: `AGENTS.md`, `design/week4.md`, `labs/module3_rag/how_to_do_langchain_rag.md`, `data/press_releases/AAPL_PRs.txt`, `tests/fixtures/press_releases_sample.txt`, `tests/test_press_releases.py`, and (filings) `tests/fixtures/filings_sample.txt`, `tests/test_filings.py`, `tests/fixtures/edgar_aapl_10q_0000320193-26-000006.txt.gz`, `tests/test_edgar_text.py`, `prometheus/agents/data_clerk/edgar_text.py`. The filings work also modified `prometheus/` (config, loaders, store, ingest, retriever, tools, agent, prompt, work order, desk, MCP server, toolbox), `test_prometheus.sh`, six existing test files, `HowToRun.txt`, `README.md`, `docs/design_choices.md` and `design/week3_e2e.md`. **Do not commit unless Jacob asks.**
- **Stale:** `AGENTS.md` (under "Rules for changes") still names `labs/module3_rag/hpw_to_do_langchain_rag.md`; the file is now `how_to_do_langchain_rag.md` (Jacob's rename). Several other docs may too. `docs/SESSION_NOTES.md` is the **old DataToolbox notes from the standalone `bmd` project** (a different file, from Oct 24): its environment line (`bmd` conda env) is not this repo's.
- New files from the real run under `data/processed/` (saved evidence): `desk_AAPL_press_releases.json` (the script's real run), `desk_ZZZZ_press_releases.json`, `desk_press_releases.json`, `desk_MSFT_press_releases.json`, `desk_AAPL_executive_changes_press_releases.json`, `desk_AAPL_dividend_press_releases.json`, and a new collection in `data/processed/vectorstore/`. Never touch `data/processed/` or `data/historical/` unasked.

---

## 10. Earlier work (Weeks 1 and 2, still true): the quotes MCP server

`prometheus/mcp/server.py` is the course's FastMCP quotes server (tools `fetch_quote`, `verify_fetch_quote`, `enter_symbol` (HITL elicitation), `noisy_fetch_quote`, `report_progress`, `ping_catalog`; resources `config://server`, `market://{symbol}/quote` (a stub, Redis later); prompt `get_market_data`). `prometheus/mcp/client.py` is the host `connect_client()`. `main.py` is the old LocalLLM + MCP smoke (`./test_prometheus.sh --legacy`).

**Hard rules to keep:**
1. Never `cache=True` on the Client for the quote path (breaks live quotes); the server uses `cache_ttl=None`.
2. Never import `client.py` from `server.py`.
3. Prompt text must not name MCP URIs or tools.
4. Prefer tool arguments, resource URIs and config over MCP Roots; prefer the host LLM (LangGraph) over MCP Sampling (both deprecated 2026-07-28).
5. Port from the labs by hand, keeping Prometheus naming and the trading domain.

Still open from then: Redis-backed quotes; richer HITL (confirm size and side).

---

## 11. Quick paste for a new chat

> Continue Capstone Prometheus. Read `AGENTS.md`, then `@design/session_notes.md`, then `@design/week4.md` (part 1 press releases, part 2 filings: the model for how to build and document a data source; section 5 of the notes is the recipe, `week4.md` section 21 is its corrections).
> State: the desk (`account_manager -> data_clerk -> advisor`) works for `news` and `press_releases` (one real AAPL run recorded) and is built for `filings` (1380 tests pass offline, mutation checks all caught, a filings order's action is `search_filing`, the 10-Q text comes from EDGAR inside `fetch_filings`; but the real filings ingest, search and desk run are NOT done: `week4.md` section 19; the 10-Q text of AAPL and META was fetched on 2026-10-30, `week4.md` section 43; `SEC_USER_AGENT` is not in `.env` yet: add it). Never run anything that calls a paid API or writes `data/processed/` without asking me. Write docs with the editor tool, update `design/week5.md` as you go, add a numbered entry to `docs/design_choices.md`, do not commit.
> Next: **<say which: the real filings run (section 19) / the Critic / other work-order actions / social media>**.
