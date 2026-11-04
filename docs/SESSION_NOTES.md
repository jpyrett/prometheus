# DataToolbox Working Session Notes

Notes for future Claude/Copilot chats. Full details are in [DataToolbox.md](DataToolbox.md).
Update this file at the end of each working session.

## Rules from the user (follow these)

- Keep it as simple as possible. Do not over-engineer.
- Do not assume anything; ask when information is missing.
- Any error: stop the current operation and return the error.
- **Do not read** `src/capstone/.aws/` or `src/capstone/.env`. Only the code reads them.
- Never print or repeat secrets. `notes/notes.txt` contains plaintext credentials; do not quote it.
- Do not use the bmd Redis/RAG code; the user is re-implementing that.
- Do not poll running terminals unless asked ("stop checking, I will message you").
- Do not run other `aws` commands while DataToolbox is downloading; they share one bandwidth cap.
- Do not create extra markdown docs unless asked.

## Environment

- Python: `/home/jpyrett/miniforge3/envs/bmd/bin/python`; host `bmdbox`, Ubuntu 24.04 x86_64.
- AWS CLI v2 2.38.5 in `~/.local/aws-cli`, linked at `~/.local/bin/aws`.
- Run: `cd /home/jpyrett/bmd/src/capstone && ~/miniforge3/envs/bmd/bin/python data_agent.py`
- Stop everything: `pkill -f data_agent.py; pkill -f '.local/aws-cli'`
- `.gitignore` covers `src/capstone/.aws/credentials`, `src/capstone/.env`, `src/capstone/data`.

## Status (Oct 24, 2026)

| Feature | Status |
|---|---|
| `init_aws_cli()` | Done, validated |
| `fetch_historical_data()` | Done. Full run in progress: day and minute aggs done; `trades_v1` at about 21 GiB of 46.9 GiB; `quotes_v1` still to do |
| `convert_to_parquet()` | Written, **not validated** with the current validation-free `_cast_batch` |
| `fetch_news()` | Done; API fields only (no scraping), Polygon time-window paging |
| `sync_market_news()` / `discover()` | Done, tested: 3,411 articles/7 days; "GLP-1" -> NVO, LLY, VKTX |
| `main()` | Calls init, fetch, convert only. News is not wired in (no symbol list yet) |

## Decisions and lessons

- `preferred_transfer_client = classic` is required (`crt` returns 403).
- Month-level `aws s3 sync` only; no per-file requests, no checksumming. `sync` skips finished files.
- The user's `.aws/config` values (`max_concurrent_requests = 20`, etc.) are respected; code does not override them.
- Row validation was removed (regex over about 120M rows maxed the CPU and dropped all rows). It is a TODO.
- Download throughput is capped server side: about 29 MiB/s for one file, 12-16 MiB/s for 20 streams. Local CPU, RAM, disk and NIC are fine. Stray parallel syncs made earlier runs slower.
- `aws` redraws its progress line with carriage returns; use `tr '\r' '\n'` when reading logs.
- Large conversion tests in `pylanceRunCodeSnippet` can be cancelled by the user; prefer the terminal.
- `fetch_news`: Finnhub (company news) and Polygon. Press releases are a separate function/file (`fetch_press_releases`, `data/press_releases/`), deliberately independent of news (an agent tool calls it). Filter on headline + summary + keywords + insight reasoning. No scraping (removed). Per-symbol append-only file, `<<<ARTICLE>>>` splittable, with `<<<SUMMARY>>>` and `<<<INSIGHTS>>>` sections.
- Polygon `next_url` paging is broken (10 per page, then repeats). Use `published_utc.gte` time-window paging with id de-dup. Progress stuck while counts grow = repeated pages.
- `discover`: no free-text search exists in either API, so market-wide news is stored in `news/market/*.jsonl` and searched locally. Design reasons in `design_choices.md`.
- Polygon volume is about 500 articles/day, not thousands.
- NewsAPI is not available (no paid plan).

- `fetch_filings`: Massive filings endpoints (index, 10-K sections, 8-K text), separate from news; 10-Q has no text; wider form list (14C, 14A, 13D/G, 144); EDGAR direct dropped (Massive is enough). Finnhub /stock/filings is metadata only.
- `sync_cnbc_news`: CNBC public RSS (7 feeds) converted to the news schema into `news/market`; headline and link only, no tickers; `search` matches it by text. Alpaca is Benzinga only; licensed feed skipped. Test client has a CNBC step; full run after a rebuild of `news/` exited 0 (market store 3,561 articles, 152 CNBC; META search 20 news + 1 filing, 1 CNBC).
- `quotes_v1` download (2026-10-24 14:00): sync of `data/historical/quotes_v1/2026/10` is running. 17 files exist remotely (about 147 GB); 7 are complete, 2026-10-15 is partial (9.6 of 10.3 GB). 2026-10-08, 10-13 and 10-14 are not on disk yet and no parquet exists for October. Nothing converted or validated yet.

## Next steps

1. When the download finishes, confirm `convert_to_parquet()` runs: row counts above 0, `size` as int64, schemas correct.
2. Validate the `quotes_v1` download and conversion (confirm 2026-10-08, 10-13 and 10-14 arrive).
3. Define the symbol list and wire `fetch_news` into `main()`.
4. Add row-level validation/repair in `_cast_batch` (TODO).
5. The user will build the RAG on `{NEWS_OUTPUT_DIR}/{SYMBOL}.txt` and `news/market/`; keep formats stable.
6. Optional: query expansion or embeddings for `discover`; `.env.example`.
