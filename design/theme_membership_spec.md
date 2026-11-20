# A company found in the news joins a theme only when the Data Clerk's model confirms it belongs (entry 120)

Jacob, 2026-11-20, market open, "we need to stabilize the app with this live market data". Chosen: "Model confirms news adds".

## 1. The problem (real runs)

`DataClerk.find_companies(phrase, count, today)` (`prometheus/agents/data_clerk/data_clerk_agent.py`) builds the candidates of a theme from two sources:
1. the Data Clerk's model PROPOSES companies for the phrase, and code resolves each to a real listing (`_resolve_proposal`);
2. every ticker that the stored market news ties to the phrase with 3 or more articles (`self.tools.discover_tickers(text, from_date=30, top=10)`) is ADDED with `how: "stored market news"` — with NO check that the company belongs to the theme.

Source 2 lets in companies that are only mentioned next to the phrase:
- 2026-11-20, `"I want to buy oil stocks. Split $30,000 evenly among the 3 most recommended companies."` (live, `--fetch`): PG (Procter & Gamble, 4 articles, top headline "Inflamed Oil Prices Send Pre-Markets Lower") and META (6 articles, "Why Micron Stock Dropped on Thursday") were candidates; the Critic APPROVED PG (79) and Auto mode bought $10,000 of it.
- 2026-11-20, `"the 3 most recommended cybersecurity stocks. Split $9,000 evenly."`: AMZN and NVDA came in the same way; AMZN was bought with the whole $9,000 (the online Opus judge: risk 0.42, ALLOW_AND_FLAG).

## 2. Rules

| # | Rule |
|---|---|
| T1 | Source 1 is unchanged. Source 2 stays (it finds names the model's memory may not know), but a news-found ticker is added ONLY when the Data Clerk's model confirms that the company belongs to the phrase. Belonging to a theme is understanding, so it is the model's reading (Jacob's rule: the model reads, code checks what code can check exactly); code checks the FORM of the answer. |
| T2 | ONE extra model call per `find_companies`, only when source 2 has at least one ticker not already proposed. Input: the phrase and, for each such ticker, the symbol, the company's listed name (the Finnhub directory's name by `symbols.find_symbol(symbol, self.symbol_search)`, paced as today; a ticker the directory does not answer for is dropped with a note, never sent unnamed) and the top matching headline the discovery already has. Output JSON: `{"members": [{"symbol": "...", "belongs": true|false, "reason": "one short sentence"}]}`. |
| T3 | Code checks the answer: a symbol not in the list sent is ignored; a ticker the answer does not mention, a `belongs` that is not a bool, or no usable answer at all (after the one retry of `calls.ask_object`) = NOT added. FAIL CLOSED: a failed call adds no news-found company (the model's own proposals stand alone), and `notes` says so. |
| T4 | A confirmed company is added as today with `how: "stored market news, confirmed by the Data Clerk's model"` and its `note` (articles count and the model's reason). A refused one is NOT a candidate; it is listed in `out["notes"]` as `"PG (Procter & Gamble Company): in the news for 'oil' but not in the theme: <reason>"`, so the report can say why it was left out. |
| T5 | The prompt lives in `prometheus/prompts/` next to `companies_prompt` (find where that is and put it in the same module or a sibling). Its examples use made-up words only (no real sector, company or ticker, as every prompt of the project). It says: a company belongs when the theme describes ITS OWN business (what it makes, sells or does), not when it is only affected by, mentioned with, or a customer of the theme; the text between the tags is data. Temperature as the Data Clerk's other calls. |
| T6 | The call goes through `calls.ask_object` with its own run (`new_run("theme_members", self.model)`), kept on the result as `out["membership"]` (model, tokens, error) so a trace and a capture show it. If the module has a `traced` span pattern for the Data Clerk (entry 115), the call nests under it like the other calls. |
| T7 | Nothing else changes: `discover_tickers` itself, its threshold (3 articles), `_resolve_proposal`, the Advisor, the Critic, sizing. No list of sectors, keywords or tickers in code. No size or risk rule. |

## 3. Tests (real data only)

- Capture the REAL model's replies (standing yes of entry 84; a few cents) for the two real cases above, built from the REAL stored data (`discover_tickers` on the stored market news for "oil" and "cybersecurity" as it stands today, with their real directory names), into `tests/fixtures/data_clerk_real/theme_members.json` (or the folder the Data Clerk's real captures already use). A small script under `scripts/` (asks for `--yes`) makes it, like `scripts/capture_guard_replies.py`.
- Tests replay the captured replies (no stand-in model): PG and META are refused for "oil", AMZN and NVDA for "cybersecurity"; a real oil or cybersecurity company in the same list is confirmed (whatever the capture holds — if the model refuses a real member, that is a finding to REPORT, not a test to bend); the code checks of T3 (unknown symbol ignored, missing = not added, a failed call adds nothing and says so) on the real reply edited by one field.
- The existing tests of `find_companies` still pass; update a test only where it pinned an unconfirmed news add, and say which.
- Full suite: `~/miniconda3/envs/cmu/bin/python -m pytest -q -p no:cacheprovider --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo`. Known failures before this work: `test_approved_or_denied`, `test_rag_loaders`, `test_trader_handshake` (resting order), and `test_data_clerk_verification::test_recent_sources_respects_the_window_the_symbol_and_k` after 12:00 UTC (a clock flake).

## 4. Do not

Edit `docs/design_choices.md`, `AGENTS.md`, `HowToRun.txt` or `README.md` (the main session writes entry 120 from your report; another worker is editing the security files at the same time — do not touch `prometheus/security/`, `configs/security*`, `prompts/guard.py`). Do not run a session, `--fetch`, or anything that writes `data/` or `news/`. Do not commit. Leave no `__pycache__`.

## 5. Report

What changed (files, functions), the captured replies (which tickers confirmed or refused and the reasons), the test results, the cost, and the text of entry 120 (problem, decision, alternatives, consequences) for the main session to paste.
