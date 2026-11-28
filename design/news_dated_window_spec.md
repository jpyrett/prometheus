# The news of a past day: a dated window for a news request (design choices entry 151)

Jacob, 2026-11-28: *"We have a bug. We cannot fetch a single day news in the past ... it is currently finding the last most recent day instead of the specific day. Fix this."*

## 0. The bug

A news request (entry 149) has ONE kind of window: `news_days`, a rolling span counted back from NOW. "What was the news on TSLA on November 3?" has no field to land in:

- `news_window` (intent.py) reads only "last/past N days", "this week/month", "today", "yesterday". A date is not read.
- The model's form has only `news_days`, and the prompt does not tell it today's date, so it cannot turn "November 3" or "last Tuesday" into anything but a number of days, which is counted back from today.
- The work order carries `since_days` only, `refresh_news` fetches `from_date=days` back from now, and `retrieve_news` / `search_news` count `since_days` back from now. The result is the latest news, never the day asked for.

The lower layers already support a dated window: `search_news(..., now=, until=)` and `fetch_news(..., start=, end=)` (`fetch_window`). The gap is the reading of the request, the work order, and the news run that joins them.

## 1. Rules

| # | Rule |
|---|---|
| D1 | **A dated window.** A news intent may hold `news_from` and `news_to`, two calendar dates (US Eastern days, `to` included). One day is `from == to`. Both or neither. When they are set, `news_days` is None and is not used anywhere; when they are not, everything is exactly as entry 149 (rolling `news_days`). The intent's validator refuses: one without the other, `from > to`, `to` after today's Eastern date (the simulated date while a day is bound), a span longer than `news.max_days` (90), a `from` older than `news.max_age_days` (new config key, 365) before today, and either field on an action that is not `news`. Both are left out of the dump when None (like `news_days`), so the Proving Ground's research-cache keys do not change. |
| D2 | **Who reads it.** Like N3: a PROGRAM reads an explicit date from the words and wins over the model; otherwise the model's `news_from`/`news_to`; otherwise the rolling rules of N3. `news_dates(typed, today)` reads a closed list: ISO `2026-11-03`; `11/3`, `11/3/2026`, `11/3/26` (US order, month first); `Nov 3`, `November 3rd`, `Nov. 3, 2026`, `3 November 2026` (month names and their 3-letter forms, any case, optional ordinal suffix and comma). A range is two dates joined by `to`, `through`, `thru`, `until`, `-`, `–`, or `between X and Y` / `from X to Y`; a second date with only a day number takes the first one's month ("Nov 3 to 5"). A date with no year is the most recent such date on or before today (November 30 typed on 2026-11-28 is 2025-11-30). A date that does not exist (Feb 30, 13/1) is not read (the model's value or the clarify path stands). An explicit date with a rolling span in the same words ("Nov 3 and the last week"): the date wins and `corrected` says so in one line ("showing the news of 2026-11-03; the 'last week' in the same request is not used"). Relative day phrases the program does not read ("last Tuesday", "the day before Thanksgiving", "Black Friday") are the MODEL's: it reads the request (Jacob's rule) and code checks the result against D1. |
| D3 | **The model's form.** The prompt (`prompts/account_manager_request.py`) gets `news_from` and `news_to` in the JSON shape, the RETRY line and the `known` keys of `parse_intent`, and one bullet: only with action "news", when the person asks for the news of a particular past day or span of days, write the first and last day as YYYY-MM-DD (the same day twice for one day) and leave `news_days` null; null otherwise. TODAY'S DATE (Eastern; the bound day while the Proving Ground has one) is given in the USER message ("Today is Saturday 2026-11-28."), never in the system prompt, so the prompt text stays cache-stable. Made-up tickers in the examples, like the other bullets. |
| D4 | **The work order.** `WorkOrder` gets `news_from: str | None` and `news_to: str | None` (ISO dates). `new_work_order` takes them; `verify_request` checks D1's rules on them (both or neither, valid ISO dates, from <= to, span <= the cap) and that `since_days` is None when they are set. `time_period` is "2026-11-03" for one day, "2026-11-03 to 2026-11-05" for a span. The query reads "Summarize the news of TSLA on 2026-11-03" / "from 2026-11-03 to 2026-11-05". |
| D5 | **The search is held to the window IN CODE.** The model cannot be trusted to pass dates, so the window is applied below it: `DataClerkTools` gets a per-run window (`tools.within(start, end)`, a copy, like `DataClerk.fetching`) that `_retrieve` applies to EVERY search, the model's and the sweep's: `since=start_of_day(news_from)`, `until=end_of_day(news_to)`, `now=end_of_day(news_to)` (both ET, `trading_clock`; add `start_of_day` beside `end_of_day`). A `since_days` the model passes then counts back from the window's end and the narrower bound wins, so it can only narrow, never widen or move the window. `search_news` gets an explicit lower bound `since: datetime | None` (aware; combined with `since_days` by taking the later), so the day boundary is exact (no 24-hour arithmetic over a DST change). A bound Proving Ground day still caps `until`/`now` as today. The "no stored article matches the filters" message names the window ("from 2026-11-03 to 2026-11-03") instead of "the last N days". |
| D6 | **The question.** `_news_question` with a dated window says "use news from 2026-11-03 (that day only, US Eastern time)" / "from 2026-11-03 to 2026-11-05", and "the search is already limited to that window: do not set since_days". |
| D7 | **The fetch (N7 kept).** `refresh_news(symbols, days, start=None, end=None)`: with a window it calls `fetch_news(symbol, start=news_from, end=news_to + 1 day)` (the providers' days are UTC days and an evening article of an Eastern day is the next UTC day; `fetch_window` already clips the end to now). The search, not the fetch, decides what is shown (D5), so the extra hours are never shown. The guards, the refusal while a day is bound and the "showing the stored news" fallback are unchanged. |
| D8 | **What is shown.** `news_view` adds `window: {from, to} | None` (and `days` stays None for a dated run). The headline (`news_headline`), `describe()`, the session's `news_lines` and the page's NewsPanel header say "on 2026-11-03" (one day) or "from 2026-11-03 to 2026-11-05", in place of "for the last N days". `describe()` adds the weekday for one day: "show the news of Tesla on Tuesday 2026-11-03 (nothing will be bought)". A day with no stored or fetched article says so plainly (the existing no-supporting-text path, with the window in its words). |
| D9 | **Nothing else changes.** Rolling windows, every other action, route and printed line are as they were; every existing test passes with no expected value changed. Weekends and market holidays are valid days (news is published on them): the market calendar is NOT consulted. No size, risk or liquidity rule (entries 91, 93, 94). |

## 2. What changes, module by module

- `configs/account_manager_v1.json` and its loader: `news.max_age_days` 365.
- `prometheus/common/trading_clock.py`: `start_of_day(day)` beside `end_of_day`.
- `prometheus/agents/account_manager/intent.py`: the fields and validator (D1), `news_dates` (D2) beside `news_window`, `_news_intent` choosing between them, `describe`/`_describe`/`what`/`as_request`, the `known` keys. `parse_intent`/`understand_request` pass today's date into the user message (D3).
- `prometheus/prompts/account_manager_request.py`: D3.
- `prometheus/common/work_order.py`: D4.
- `prometheus/rag/retriever.py`: `since` (D5). `prometheus/tools/data_clerk.py`: `within`, `_retrieve` applying the window, the message (D5).
- `prometheus/agents/data_clerk/data_clerk_agent.py`: `_news_question` (D6), `refresh_news` (D7), `request_news` asking with `self.tools.within(...)` when the order has a window.
- `prometheus/langgraph/desk.py`: `news_plan` and `news_data` build the order, query and fetch from the window when the intent has one (`plan["news_from"]`, `plan["news_to"]`).
- `prometheus/application/web/views.py`, `prometheus/application/session.py`, `ui/src/api/types.ts`, `ui/src/components/NewsPanel.tsx`: D8.
- `docs/design_choices.md` entry 151; `HowToRun.txt` one example ("What was the news on TSLA on 2026-11-20?"); `AGENTS.md` one sentence.

## 3. Tests (real data where it exists; no stand-in agents in new tests)

- `news_dates`: each form of D2 once, a range of each joiner once, a yearless date before and after today, a date that does not exist, a date plus a rolling span (date wins, note). No boundary repeats (entry 90).
- The validator of D1, each refusal once.
- `search_news` with `since`/`until` on a small real store (the existing retriever test store): an article at 23:30 ET of the day is in, one at 00:30 ET of the next day is out, one of the day before is out; a model `since_days` cannot widen the window.
- `_retrieve` with `within`: the model's call and the sweep's are both held to the window.
- `refresh_news` with a window calls `fetch_news` with `start`/`end` (a recording toolbox, no network).
- The desk route on the stored news: "What was the news on TSLA on 2026-11-20?" (stored news only, `--refuse-news-fetch`) gives an order with `news_from == news_to == "2026-11-20"`, `since_days` None, and every source's `published_utc` inside the Eastern day.
- The UI: NewsPanel's header for a dated view; the vitest suite passes.
- REAL model forms: capture the model's replies for "news on TSLA on November 20", "what happened with Apple last Tuesday?", "NVDA news from Nov 3 to Nov 5", "TSLA news the day before Thanksgiving", and one rolling control ("TSLA news last week"), with today fixed in the user message, into `tests/fixtures/session_real/news_dated_forms.json` (the capture script the earlier form fixtures used; a paid model call under the standing yes for desk tests). Checked: action, stocks, window, and that the program's date wins where one is written.
- The whole Python suite (the one deselected test as usual) and the UI tests pass with no existing expected value changed.

## 4. Order of work

config + `start_of_day` -> `news_dates` + intent + prompt (D1 to D3) -> work order (D4) -> retriever `since` + tools `within` (D5) -> Data Clerk question and fetch (D6, D7) -> desk -> views, session, UI (D8) -> tests -> form capture -> docs entry 151.

## 5. As built (2026-11-28)

Built and working: Jacob's request on the live page, "what is the news for KO on July 20, 2026", is read by the program as 2026-07-20 to 2026-07-20 and answered with that day's news. Python suite before the change 2 failed, 2781 passed; after 2 failed, 2833 passed (+52 in `tests/test_news_dated_window.py`; the same two failures read live stored data and fail identically on the untouched tree: `test_as_of_clock.py::test_the_toolbox_searches_the_stored_text_as_of_the_bound_day_and_never_refreshes`, `test_rag_ingest.py::test_real_chunks_of_every_filing_fit_the_embedder_limit_once_cut_by_tokens_and_lose_no_text`). UI 206 tests pass (3 new, `ui/src/components/newsDated.test.tsx`). No existing expected value changed.

Where the build differs from the rules above:

- **A weekday check of the model's window (not in D2).** The real model got "what happened with Apple last Tuesday?" (today Saturday 2026-11-28) wrong twice: 2026-11-21 (a Saturday) in the first capture, 2026-11-22 (a Sunday) in the second. `weekday_problem` (intent.py) refuses a model window when a weekday the words name falls on no day of it, and the model is asked again with the reason; its second reply was 2026-11-24, right. It never touches a date the program read from the words. Relative days stay the model's (Jacob's rule), but its day arithmetic is not trusted unchecked.
- **`calendar_today()`** (trading_clock): the Eastern calendar date (the bound day while one is bound), the "today" of D1 and D3; the simulated trading day is not a person's today.
- **`DataClerkTools.fetch_news_window(symbol, start, end)`** is a separate method, NOT a tool of the model: adding `start`/`end` to the model's `fetch_news` would change the tool schema the model sees on every request. `refresh_news` calls it with `end = news_to + 1 day` (D7).
- **A program-read window that cannot be shown** (in the future, over 90 days long, older than 365 days, backwards) is `NeedsClarification` at once; a model-written one goes through the validator and the model's retry.
- **`verify_request` also refuses a dated window on a source that is not the news.**
- **The model's dates beat rolling words** (a literal reading of D2): "yesterday", when the model dates it, is one day. Dates the model puts on an action that is not `news` are dropped silently (as `news_days`), and refused on a quote.
- **The real form capture**: `scripts/capture_news_dated_forms.py` -> `tests/fixtures/session_real/news_dated_forms.json`, six requests with today fixed at Saturday 2026-11-28 (the five of section 3 and "What was the news on TSLA on 2026-11-20?" for the desk test). "November 20", the ISO form, "Nov 3 to Nov 5", "the day before Thanksgiving" (2026-11-25) and "last week" (`news_days` 7) were read right the first time.
- **Not verified**: the dated fetch against the real providers was not run by the tests (no network); only the arguments handed to the toolbox are tested. The live page run fetched it.
- **Lesson**: the page on port 8000 must be restarted right after the Python change. Jacob's first try hit the old server, which refused the new `news.max_age_days` key.

## 6. Open items for Jacob

1. `news.max_age_days` 365 (how far back a past day may be asked for). The stored news goes back about 6 months (the Proving Ground backfill) and the fetch providers about a year.
2. "This week" stays rolling (7 days back from today), not Monday-to-today. Say if you want calendar weeks.
