# DataToolbox Guide

`src/capstone/data_toolbox.py` is a standalone, CPU-only class that (1) installs the AWS CLI,
(2) downloads historical flat files, (3) converts them to Parquet and (4) fetches news into text files for a future RAG and (5) discovers companies from trend keywords. It does not use the bmd Redis/RAG code.

**Error policy:** any error stops the current operation and is raised/returned. There are no exceptions.

**Secrets:** never put keys in code, logs, notes or chat. Credentials live only in
`src/capstone/.aws/` and `src/capstone/.env` (both git-ignored). Do not read those files; the code
reads them.

## Environment

| Item | Value |
|---|---|
| Python | `/home/jpyrett/miniforge3/envs/bmd/bin/python` |
| Libraries | pyarrow, httpx, python-dotenv |
| OS | Ubuntu 24.04 x86_64 (host `bmdbox`); 24.04 and 26.04 are accepted |
| Run | `cd /home/jpyrett/bmd/src/capstone && ~/miniforge3/envs/bmd/bin/python data_toolbox.py` |

## Layout

```
src/capstone/
  data_toolbox.py
  docs/DataToolbox.md       this guide
  docs/SESSION_NOTES.md   how the recommendation system works (Week 5 session notes; the old DataToolbox notes are in git history)
  .aws/{credentials,config}
  .env                    API keys (FINNHUB_API_KEY, POLYGON_API_KEY, ...)
  news/{SYMBOL}.txt       per-symbol news (fetch_news)
  news/market/            market-wide news store (sync_market_news / discover)
  data/press_releases/{SYMBOL}.txt   press releases (fetch_press_releases)
  data/filings/{SYMBOL}.txt          SEC filings (fetch_filings)
  data/
    historical/{type}/{year}/{month}/YYYY-MM-DD.csv.gz     downloads
    historical/converted/{type}/YYYY-MM-DD.parquet         Parquet output
```

## 1. `init_aws_cli()`

- Linux Ubuntu only. Downloads the official AWS CLI v2 zip and installs it to `~/.local/aws-cli`,
  linked at `~/.local/bin/aws` (no sudo, `--update`). Skips if the same version is installed.
- Sets `preferred_transfer_client = classic`. The default `crt` client gets **403 Forbidden** from
  `files.massive.com`.
- Adds a trailing newline to `.aws/config` if missing (`aws configure set` otherwise appends to the
  last line and corrupts the file).
- Concurrency and chunk size are **not** overridden. They come from `.aws/config`:
  `max_concurrent_requests = 20`, `max_queue_size = 10000`, `multipart_threshold = 64MB`,
  `multipart_chunksize = 16MB`.
- Credentials/config are passed with `AWS_SHARED_CREDENTIALS_FILE` / `AWS_CONFIG_FILE`
  (default profile). Checksums are disabled with `AWS_REQUEST_CHECKSUM_CALCULATION` and
  `AWS_RESPONSE_CHECKSUM_VALIDATION` set to `when_required`.

## 2. `fetch_historical_data()`

Syncs the **current month** for four types, one `aws s3 sync` per type:

- Source: `s3://flatfiles/us_stocks_sip/{type}/{year}/{month}`, endpoint `https://files.massive.com`
- Types: `day_aggs_v1`, `minute_aggs_v1`, `trades_v1`, `quotes_v1`
- `sync` skips completed files, so restarts are safe.

Manual equivalent:

```bash
aws s3 sync s3://flatfiles/us_stocks_sip/trades_v1/2026/10 . --endpoint-url https://files.massive.com
```

### Download speed (measured Oct 24, 2026)

- Raw internet is fast (about 38 MiB/s to a non-Massive host). The 1 Gbps wired NIC, CPU, RAM and
  NVMe were all idle, so the local machine is not the bottleneck.
- One 3.2 GiB trades file alone: about 29 MiB/s. The full 20-stream sync: about 12-16 MiB/s.
- Parallel runs share one cap, so extra streams do not help. **Never run several syncs or manual
  `aws` commands at the same time as DataToolbox** (earlier runs were slowed to about 8 MiB/s by
  three stray manual syncs).
- `trades_v1` for a month is about 46.9 GiB; `quotes_v1` is larger.

## 3. `convert_to_parquet()`

- CSV.GZ -> Parquet, CPU only (pyarrow streaming CSV reader, `ParquetWriter`, zstd).
- Output: `data/historical/converted/{type}/YYYY-MM-DD.parquet`, with a stdout progress bar
  `[i/N] type/YYYY-MM-DD.csv.gz [####] 100% N rows`.
- Columns are read as strings, then cast per batch with `_cast_batch` (vectorized, no row
  validation). If a direct cast fails (for example `1.000000` in an integer column) it goes through
  float64 and rounds.
- Schemas (`SCHEMAS` in the file) cover all four types.
- **TODO:** row-level validation/repair of bad values (for example `11.3`). The earlier regex
  validation over about 120M rows maxed the CPU and also dropped every row, so it was removed.
  Structurally broken CSV rows are still skipped by the CSV reader.
- Status: the current `_cast_batch` has **not yet been validated** on a full conversion.

## 4. `fetch_news(company, symbol, keywords, from_date)`

```python
DataToolbox().fetch_news(company="Apple", symbol="AAPL", keywords=["iPhone", "earnings"], from_date=30)
```

| Argument | Meaning |
|---|---|
| `symbol` | Required (Finnhub company news needs a symbol). |
| `company` | Optional company name filter. |
| `keywords` | Optional list of keywords. |
| `from_date` | Positive int = days back from now. |

Returns `{symbol, path, fetched, written, duplicates}`. Sources per symbol: Finnhub
`/company-news` and Polygon `/v2/reference/news?ticker=`. Press releases are **not** fetched here;
see `fetch_press_releases` (section 6).

- Filter text = headline + summary + API keywords + insight reasoning (case-insensitive, whole
  word). Company AND any keyword; if only one is given, only that one; neither keeps everything.
- **No scraping.** Article bodies are not downloaded (paywalls, robots.txt and poor yield). The
  RAG text is what the APIs return: summary, keywords, tickers, image URL, and Polygon `insights`
  (per-ticker sentiment with reasoning). See `docs/design_choices.md`.
- Output `{NEWS_OUTPUT_DIR}/{SYMBOL}.txt` (default `./news`, relative to `data_toolbox.py`),
  appended as articles arrive; a rerun skips URLs already present (host + path). Split on lines
  that are exactly `<<<ARTICLE>>>`:

```
<<<ARTICLE>>>
id / symbol / company / api / type / published_utc / publisher / author / url / image_url /
category / tickers / keywords / headline      (one `name: value` line each, single-line)
<<<SUMMARY>>>
summary text
<<<INSIGHTS>>>
TICKER | sentiment | reasoning      (one line per Polygon insight)
<<<END>>>
```

Text lines starting with `<<<` are prefixed with a space so markers stay unambiguous.

### Rate limits and errors

| API | Pacing | Notes |
|---|---|---|
| Finnhub | 0.25 s between calls | key in the `X-Finnhub-Token` header; 7-day windows |
| Polygon/Massive | 0.1 s between calls | `limit=1000`, time-window paging (below) |

- HTTP 429/5xx are retried with backoff (honours `Retry-After`), 5 attempts, then raised. Error
  messages never include the URL, so keys are not leaked. NewsAPI is not used.
- **Polygon paging:** `next_url` does not work (without `limit` it returns 10 per page, with it the
  pages repeat). `_iter_polygon` moves `published_utc.gte` forward after each full page and
  de-duplicates by id. It raises if 1000+ articles share one timestamp.

## 5. `sync_market_news(from_date=7)` and `discover(phrase, from_date=7, top=10, refresh=True)`

Finds which companies a trend is about (for example "GLP-1" returns NVO, LLY, VKTX), without
knowing tickers up front. Neither API has free-text search, so market-wide news is stored locally
and searched there.

- `sync_market_news`: pulls Polygon market-wide news (no ticker, every page) plus the Finnhub
  `general` and `merger` feeds (latest items only) into `{NEWS_OUTPUT_DIR}/market/YYYY-MM-DD.jsonl`
  (one JSON article per line, by UTC publish day). `market/_coverage.json` records the stored range,
  so a rerun fetches only the new time (1 h overlap) and de-duplicates by URL. A progress bar
  shows percent, article counts, elapsed time and ETA.
- `discover`: syncs first (`refresh=False` skips it), then searches the store: whole-word,
  case-insensitive match on headline, summary, keywords and insight reasoning. Tickers are counted
  from the article's insights if present (what it is really about), else from its `tickers` list.
  Returns `{phrase, from_date, scanned, matched, tickers: [{symbol, articles, positive, neutral,
  negative, providers, headlines}]}`, best first.
- Measured: about 3,400 market articles for 7 days (roughly 500 per day). "GLP-1" over 7 days
  matched 19 articles: NVO 12, LLY 11, VKTX 5.
- Limit: matching is literal. "weight loss pills" will not match articles that only say "GLP-1".
- Test program: `test_news.py` (`NewsTest(cases, phrases, from_days)`), so DataToolbox runs are not
  interrupted. Not wired into `main()`.

## 6. `fetch_press_releases(symbol, from_date=365)`

```python
DataToolbox().fetch_press_releases("AAPL", from_date=365)
```

- Fetches official company press releases (Finnhub `/press-releases`, one call for the range) and
  appends them, oldest first, to `data/press_releases/{SYMBOL}.txt`.
- **Fully separate from the news code** (own request, parser, file and record format; shares only
  the HTTP helper and key loading), so an agent tool can call it on its own. `fetch_news` does not
  fetch press releases.
- Record format (split on `<<<PRESS_RELEASE>>>`):

```
<<<PRESS_RELEASE>>>
id / symbol / published_utc / url / headline     (one `name: value` line each)
<<<TEXT>>>
full release text
<<<END>>>
```

- Reruns skip ids already in the file. Errors stop the operation and raise.
- Returns `{symbol, path, fetched, written, duplicates}`. Measured: AAPL, 365 days -> 33 releases.
- Adapted from the BigMessyData `FinnhubSource.fetch_press_releases` (same endpoint), without the
  async client, `NewsArticle` class or swallowed errors.
- Test: `test_news.py` (`PRESS_RELEASE_SYMBOLS`, `PRESS_RELEASE_DAYS`). Not wired into `main()`.

## 7. `fetch_filings(symbol, from_date=365, forms=FILING_FORMS)`

- Appends SEC EDGAR filings for a ticker, oldest first, to `data/filings/{SYMBOL}.txt`. Source: the
  Massive (Polygon) filings endpoints, with the existing `POLYGON_API_KEY`; no extra account. They are
  free during the early-access beta (the plan may change). Separate from news and press releases.
- Index: `/stocks/filings/vX/index` (all forms, SEC URL). Text: `/stocks/filings/10-K/vX/sections` (Business and
  Risk Factors) and `/stocks/filings/8-K/vX/text` (Items). 10-Q has no text endpoint, so it is listed with
  `text_status: not_available`.
- Record format (split on `<<<FILING>>>`):

```
<<<FILING>>>
accession / symbol / form / filed / period_end / url / text_status    (one `name: value` line each)
<<<TEXT>>>
== section name ==
section text
<<<END>>>
```

- De-duplicated by accession number; errors stop the operation. If a form returns 100+ rows the call raises
  and asks for a smaller `from_date`.
- Returns `{symbol, path, fetched, written, duplicates}`. Measured: AAPL, 365 days -> 12 filings.
- Test: `test_news.py` (`FILING_SYMBOLS`, `FILING_DAYS`). Not wired into `main()`.
- Default forms: 10-K, 10-Q, 8-K, PRE 14C, DEF 14C, DEF 14A, Schedule 13D/13G (and /A) and 144 (and /A); Form 4 is
  excluded (pass `forms=`). Only 10-K and 8-K have text; other forms are SEC links.
- SEC EDGAR direct is not used (decision in `docs/design_choices.md` entry 5). FINRA is not covered.

## 8. `search(symbol, terms, from_date=30, refresh=True)`

- Combined search for one ticker: runs `fetch_news(symbol, keywords=terms)` and `fetch_filings(symbol)` (when
  `refresh`), then returns the stored news and filings that contain any term (whole word).
- Returns `{symbol, terms, from_date, news: [{published, publisher, headline, url}], filings: [{filed, form,
  url, text_status}], filings_in_window}`. Filings without text match on form and URL only.
- Example: `search("META", ["Zuckerberg", "wealth tax", "billionaire tax"], 7)` -> 19 news items and 1 filing.
- Details and the Finnhub URL de-dup fix: `docs/design_choices.md` entry 6.
- Also matches CNBC RSS items (section 9) by headline and description text.

## 9. `sync_cnbc_news()`

- Reads seven public CNBC RSS feeds (Top News, World, Business, Earnings, Technology, Finance, Investing) and
  writes each item in the existing news schema (`api` = `cnbc-rss`, `publisher` = `CNBC`) to `news/market/`,
  de-duplicated by URL. Returns `{path, fetched, written, duplicates}`.
- RSS has headline, description, link and time only (no tickers, author, image) and only the latest 30 items per
  feed, so run it regularly. `discover` ignores these items (no tickers); `search` finds them by text.
- Terms restrict repackaging CNBC content. Decision: `docs/design_choices.md` entry 7.

## Known issues and TODOs

- Row-level validation/repair in `_cast_batch` (see section 3).
- Validate the Parquet conversion on a real full run (`size` must come out int64 from `1.000000`).
- `quotes_v1` download not yet validated end to end.
- Decide the symbol list and call `fetch_news` from `main()`.
- Phrase matching is literal; query expansion or embeddings could be added later.
- If CPU is a concern during conversion, a lighter Parquet compression could be tested.
