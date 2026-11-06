"""DataToolbox: download historical flat files (AWS CLI) and convert them to Parquet.

Standalone, CPU-only (pyarrow). Any error stops the current operation and is raised.
"""
import gzip
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv_csv
import pyarrow.parquet as pq
from dotenv import dotenv_values

from prometheus.agents.data_clerk import edgar_text
from prometheus.agents.data_clerk.edgar_text import EdgarNotConfigured

S3_ROOT = "s3://flatfiles/us_stocks_sip"
ENDPOINT_URL = "https://files.massive.com"
DATA_TYPES = ["day_aggs_v1", "minute_aggs_v1", "trades_v1", "quotes_v1"]
SUPPORTED_UBUNTU = {"24.04", "26.04"}
AWS_CLI_URLS = {
    "x86_64": "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip",
    "aarch64": "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip",
}

# Column types per data type (header descriptions reused from the main bmd project).
# CSV values are read as strings and cast per batch (see _cast_batch).
SCHEMAS = {
    "quotes_v1": {
        "ticker": "str", "ask_exchange": "int16", "ask_price": "float64", "ask_size": "int64",
        "bid_exchange": "int16", "bid_price": "float64", "bid_size": "int64",
        "conditions": "str", "indicators": "str", "participant_timestamp": "int64",
        "sequence_number": "int64", "sip_timestamp": "int64", "tape": "int8",
        "trf_timestamp": "int64",
    },
    "minute_aggs_v1": {
        "ticker": "str", "open": "float64", "high": "float64", "low": "float64",
        "close": "float64", "volume": "float64", "transactions": "int32", "window_start": "int64",
    },
    "day_aggs_v1": {
        "ticker": "str", "open": "float64", "high": "float64", "low": "float64",
        "close": "float64", "volume": "float64", "transactions": "int32", "window_start": "int64",
    },
    "trades_v1": {
        "ticker": "str", "conditions": "str", "correction": "int8", "exchange": "int16",
        "id": "str", "participant_timestamp": "int64", "price": "float64",
        "sequence_number": "int64", "sip_timestamp": "int64", "size": "int64", "tape": "int8",
        "trf_id": "int16", "trf_timestamp": "int64",
    },
}
ARROW_TYPES = {
    "str": pa.string(), "int8": pa.int8(), "int16": pa.int16(), "int32": pa.int32(),
    "int64": pa.int64(), "float64": pa.float64(),
}

# ---- News (fetch_news) ----
# Aggregated per-symbol file {NEWS_OUTPUT_DIR}/{SYMBOL}.txt. A RAG loader splits it into articles on
# lines that are exactly ARTICLE_START. Each record is:
#   <<<ARTICLE>>>        header lines "key: value" (single line each)
#   <<<SUMMARY>>>        API summary/description text
#   <<<INSIGHTS>>>       one line per ticker: "TICKER | sentiment | reasoning" (Polygon only; may be empty)
#   <<<END>>>
# Any line inside text that begins with "<<<" is prefixed with a space, so markers are unambiguous.
ARTICLE_START, SUMMARY_START, INSIGHTS_START, ARTICLE_END = (
    "<<<ARTICLE>>>", "<<<SUMMARY>>>", "<<<INSIGHTS>>>", "<<<END>>>")
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
FINNHUB_PRESS_URL = "https://finnhub.io/api/v1/press-releases"
FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
FINNHUB_GENERAL_URL = "https://finnhub.io/api/v1/news"
FINNHUB_GENERAL_CATEGORIES = ("general", "merger")
PRESS_RELEASES_DEFAULT_DIR = "data/press_releases"   # PRESS_RELEASES_OUTPUT_DIR overrides it; the RAG loader reads the same setting
PRESS_RELEASES_SUFFIX = "_PRs.txt"                   # one file per symbol, like news: <SYMBOL>_PRs.txt (design choices entry 34)
FILINGS_DIR = "filings"                 # under data/
FILINGS_URL = "https://api.massive.com/stocks/filings"
EDGAR_MIN_INTERVAL = 0.2            # EDGAR allows 10 requests a second; this is 5 (and one request is a whole 10-Q)
EDGAR_MAX_BYTES = 25_000_000        # a complete submission (a real AAPL 10-Q: 5.2 MB) larger than this is refused
EDGAR_TIMEOUT = 120.0
CNBC_RSS_URL = "https://search.cnbc.com/rs/search/combinedcms/view.xml"
CNBC_FEEDS = {"Top News": 100003114, "World News": 100727362, "Business": 10001147, "Earnings": 15839135,
              "Technology": 19854910, "Finance": 10000664, "Investing": 15839069}
FILING_FORMS = ("10-K", "10-Q", "8-K", "PRE 14C", "DEF 14C", "DEF 14A", "SCHEDULE 13D", "SCHEDULE 13D/A",
                "SCHEDULE 13G", "SCHEDULE 13G/A", "144", "144/A")
MARKET_DIR = "market"          # subfolder of the news folder holding the market-wide news store
POLYGON_NEWS_URL = "https://api.massive.com/v2/reference/news"
MASSIVE_AGGS_URL = "https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"   # daily bars of ONE ticker (REST); the same numbers as the day_aggs_v1 flat files
TICKER_BARS_DIR = "ticker_bars_v1"   # under data/historical: one <SYMBOL>.csv of daily bars per ticker, in the columns of the day files; what `fetch_daily_bars` writes and BarStore merges in
BARS_HEADER = ["ticker", "volume", "open", "close", "high", "low", "window_start", "transactions"]
FINNHUB_MIN_INTERVAL = 0.25    # 4 calls/s = 240/min, under the paid 300/min limit
POLYGON_PAGE_SIZE = 1000
POLYGON_MIN_INTERVAL = 0.1     # 10 calls/s, well under the paid limit
FINNHUB_WINDOW_DAYS = 7        # company-news returns a capped number of items per call: use small windows
API_MAX_ATTEMPTS = 5


class _RateLimiter:
    """Blocks so that calls are at least `min_interval` seconds apart."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        delay = self._last + self.min_interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


def _progress(prefix: str, fraction: float, suffix: str = "") -> None:
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(30 * fraction)
    sys.stdout.write(f"\r{prefix} [{'#' * filled}{'-' * (30 - filled)}] {fraction * 100:5.1f}% {suffix}\033[K")
    sys.stdout.flush()


class DataToolbox:
    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.aws_dir = self.base_dir / ".aws"
        self.root_data_dir = self.base_dir / "data" / "historical"
        self.converted_dir = self.root_data_dir / "converted"
        self.aws_bin_dir = Path.home() / ".local" / "bin"
        self.financials = Financials(self)

    # ------------------------------------------------------------------ AWS CLI
    def init_aws_cli(self) -> None:
        """Download and install the latest AWS CLI v2 to ~/.local (no sudo)."""
        if platform.system() != "Linux":
            raise RuntimeError(f"Unsupported OS: {platform.system()} (Linux Ubuntu required)")
        os_release = platform.freedesktop_os_release()
        if os_release.get("ID") != "ubuntu" or os_release.get("VERSION_ID") not in SUPPORTED_UBUNTU:
            raise RuntimeError(
                f"Unsupported OS: {os_release.get('PRETTY_NAME')} (need Ubuntu {', '.join(sorted(SUPPORTED_UBUNTU))})"
            )
        url = AWS_CLI_URLS.get(platform.machine())
        if url is None:
            raise RuntimeError(f"Unsupported CPU architecture: {platform.machine()}")

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "awscliv2.zip"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
                for info in zf.infolist():  # zipfile drops the executable bit
                    mode = info.external_attr >> 16
                    if mode:
                        os.chmod(Path(tmp) / info.filename, mode)
            self.aws_bin_dir.mkdir(parents=True, exist_ok=True)
            self._run([
                str(Path(tmp) / "aws" / "install"),
                "-i", str(Path.home() / ".local" / "aws-cli"),
                "-b", str(self.aws_bin_dir),
                "--update",
            ])
        self._run(["aws", "--version"])
        # The CRT transfer client gets 403 Forbidden from files.massive.com; classic works.
        config_file = self.aws_dir / "config"
        if config_file.exists() and not config_file.read_bytes().endswith(b"\n"):
            with open(config_file, "ab") as f:  # 'configure set' appends to the last line otherwise
                f.write(b"\n")
        self._run(["aws", "configure", "set", "default.s3.preferred_transfer_client", "classic"])
        # Concurrency / chunk size are left as set by the user in .aws/config.

    def _aws_env(self) -> dict:
        env = os.environ.copy()
        if (self.aws_dir / "credentials").exists():  # the project's own .aws folder when there is one; otherwise the user's ~/.aws (naming a file that is not there would make every sync fail)
            env["AWS_SHARED_CREDENTIALS_FILE"] = str(self.aws_dir / "credentials")
        if (self.aws_dir / "config").exists():
            env["AWS_CONFIG_FILE"] = str(self.aws_dir / "config")
        env["AWS_PAGER"] = ""
        # Skip checksum calculation/validation on transfers
        env["AWS_REQUEST_CHECKSUM_CALCULATION"] = "when_required"
        env["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "when_required"
        env["PATH"] = f"{self.aws_bin_dir}{os.pathsep}{env.get('PATH', '')}"
        return env

    def _run(self, cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
        try:
            subprocess.run(cmd, cwd=cwd, env=env or self._aws_env(), check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}: {e}") from e

    # ------------------------------------------------------------------ Download
    def fetch_historical_data(self) -> None:
        """Sync the current month's flat files for all four data types."""
        now = datetime.now()
        month = f"{now.month:02d}"
        year = str(now.year)
        for n, data_type in enumerate(DATA_TYPES, 1):
            target = self.root_data_dir / data_type / year / month
            target.mkdir(parents=True, exist_ok=True)
            print(f"\n== [{n}/{len(DATA_TYPES)}] {data_type} {year}/{month} -> {target}")
            self._run(
                ["aws", "s3", "sync", f"{S3_ROOT}/{data_type}/{year}/{month}", ".",
                 "--endpoint-url", ENDPOINT_URL],
                cwd=target,
            )

    def sync_day_aggs(self, months: list[str]) -> list[Path]:
        """Sync the DAY AGGREGATES (`day_aggs_v1`: one small gzip CSV per trading day, about 0.3 MB, every US ticker) of the given months, and nothing else: the Advisor's indicators need a year
        of daily bars (Advisor v1 spec section 2.1) and `fetch_historical_data` syncs only the current month, for all four data types (minute bars, trades and quotes are gigabytes). `months` are
        "YYYY-MM" texts. It only ADDS files under data/historical/day_aggs_v1 (`aws s3 sync` never deletes here), it does not convert them, and it is not a tool of the Data Clerk's model (it is not in
        DATA_CLERK_TOOLS). It uses the Massive flat-file bucket and the credentials in .aws (the project's, else ~/.aws): a real network call. Returns the folders synced."""
        if not months or len(months) > 36:
            raise ValueError("give 1 to 36 months as 'YYYY-MM'")
        parsed = []
        for text in months:  # every month is checked BEFORE anything is called: a mistyped one must not leave half a range synced
            found = re.fullmatch(r"(\d{4})-(\d{2})", text) if isinstance(text, str) else None
            if not found or not (2000 <= int(found.group(1)) <= 2100 and 1 <= int(found.group(2)) <= 12):
                raise ValueError(f"not a month 'YYYY-MM': {text!r}")
            parsed.append((int(found.group(1)), int(found.group(2))))
        with tempfile.TemporaryDirectory() as tmp:
            env = self._classic_aws_env(Path(tmp))
            folders = []
            for year, month in parsed:
                target = self.root_data_dir / "day_aggs_v1" / str(year) / f"{month:02d}"
                target.mkdir(parents=True, exist_ok=True)
                self._run(["aws", "s3", "sync", f"{S3_ROOT}/day_aggs_v1/{year}/{month:02d}", ".", "--endpoint-url", ENDPOINT_URL], cwd=target, env=env)
                folders.append(target)
            return folders

    def _classic_aws_env(self, tmp: Path) -> dict:
        """The AWS environment with the CLASSIC transfer client set, in a TEMPORARY copy of the config (init_aws_cli sets it in the project's own config: "the CRT transfer client gets 403 Forbidden from
        files.massive.com; classic works"). The user's config file is read, never changed."""
        env = self._aws_env()
        source = Path(env.get("AWS_CONFIG_FILE") or Path.home() / ".aws" / "config")
        text = source.read_bytes() if source.exists() else b"[default]\n"
        config = tmp / "config"
        config.write_bytes(text if text.endswith(b"\n") else text + b"\n")
        env["AWS_CONFIG_FILE"] = str(config)
        self._run(["aws", "configure", "set", "default.s3.preferred_transfer_client", "classic"], env=env)
        return env

    def fetch_daily_bars(self, symbols: list[str], days: int = 400, end: date | None = None) -> dict[str, dict]:
        """Refresh the DAILY BARS of some tickers from Massive's REST aggregates API: one call per ticker, the last `days` calendar days up to `end` (default today), UNADJUSTED like the flat files
        (the numbers of 2026-10-23 are the same as the day_aggs_v1 file's). A real, paid-API network call. For tickers the day files do not cover or that are fresher than they are (the flat files
        arrive in batches; the Advisor's freshness gate wants yesterday's close), and it needs no flat-file credentials. Each ticker is written to data/historical/ticker_bars_v1/<SYMBOL>.csv in
        the columns of the day files, MERGED with what is there (a date fetched again replaces the old row); nothing else under data/historical is touched. Returns {symbol: {fetched, stored,
        first, last}}. A symbol that is not a ticker, or `days` outside 1 to 800, is refused before any call; an API failure raises (the message never holds the key)."""
        wanted = list(dict.fromkeys(s.strip().lstrip("$").upper() for s in symbols))
        if not wanted or any(not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,9}", s) for s in wanted):
            raise ValueError("give one or more ticker symbols")
        if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= 800:
            raise ValueError("days must be a whole number from 1 to 800")
        end = end or datetime.now(timezone.utc).date()
        key = self._api_keys()["POLYGON_API_KEY"]
        folder = self.root_data_dir / TICKER_BARS_DIR
        out = {}
        with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
            limiter = _RateLimiter(POLYGON_MIN_INTERVAL)
            for symbol in wanted:
                url = MASSIVE_AGGS_URL.format(ticker=symbol, start=(end - timedelta(days=days)).isoformat(), end=end.isoformat())
                data = self._get_json(client, limiter, "Massive aggregates", url, {"adjusted": "false", "sort": "asc", "limit": 50000, "apiKey": key})
                rows = {int(r["t"]) * 1_000_000: [symbol, r["v"], r["o"], r["c"], r["h"], r["l"], int(r["t"]) * 1_000_000, int(r.get("n", 0))] for r in data.get("results", []) if {"t", "v", "o", "c", "h", "l"} <= set(r)}
                path = folder / f"{symbol}.csv"
                stored = {}
                if path.exists():
                    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
                        cells = line.split(",")
                        if len(cells) == len(BARS_HEADER):
                            stored[int(cells[6])] = cells
                stored.update({stamp: [str(c) for c in row] for stamp, row in rows.items()})
                folder.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".csv.tmp")
                tmp.write_text("\n".join([",".join(BARS_HEADER), *(",".join(stored[stamp]) for stamp in sorted(stored))]) + "\n", encoding="utf-8")
                tmp.replace(path)
                stamps = sorted(stored)
                out[symbol] = {"fetched": len(rows), "stored": len(stored), "first": datetime.fromtimestamp(stamps[0] / 1e9, timezone.utc).date().isoformat() if stamps else None,
                               "last": datetime.fromtimestamp(stamps[-1] / 1e9, timezone.utc).date().isoformat() if stamps else None}
        return out

    # ------------------------------------------------------------------ Convert
    def convert_to_parquet(self) -> None:
        """Convert the current month's YYYY-MM-DD.csv.gz files to <converted>/<type>/YYYY-MM-DD.parquet."""
        now = datetime.now()
        month, year = f"{now.month:02d}", str(now.year)
        jobs = []
        for data_type in DATA_TYPES:
            src_dir = self.root_data_dir / data_type / year / month
            for csv_path in sorted(src_dir.glob("*.csv.gz")):
                jobs.append((data_type, csv_path, self.converted_dir / data_type / f"{csv_path.name[:-7]}.parquet"))
        if not jobs:
            print("No csv.gz files found to convert.")
            return

        for i, (data_type, csv_path, out_path) in enumerate(jobs):
            label = f"[{i + 1}/{len(jobs)}] {data_type}/{csv_path.name}"
            if out_path.exists():
                _progress(label, 1.0, "exists, skipped")
                print()
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(".parquet.tmp")
            try:
                rows, skipped = self._convert_file(csv_path, tmp_path, data_type, label)
                tmp_path.replace(out_path)
            except Exception as e:
                tmp_path.unlink(missing_ok=True)
                print()
                raise RuntimeError(f"Conversion failed for {csv_path}: {e}") from e
            _progress(label, 1.0, f"{rows:,} rows" + (f", {skipped:,} bad rows skipped" if skipped else ""))
            print()

    def _convert_file(self, csv_path: Path, out_path: Path, data_type: str, label: str):
        schema = SCHEMAS[data_type]
        skipped = {"rows": 0}

        def skip_row(_row):
            skipped["rows"] += 1  # TODO: add better row-level processing (e.g. repair '11,3')
            return "skip"

        convert_options = pv_csv.ConvertOptions(
            strings_can_be_null=True,
            column_types={col: pa.string() for col in schema},
        )
        total_size = csv_path.stat().st_size
        rows = 0
        writer = None
        with open(csv_path, "rb") as raw, gzip.GzipFile(fileobj=raw) as gz:
            reader = pv_csv.open_csv(
                gz,
                read_options=pv_csv.ReadOptions(block_size=64 * 1024 * 1024),
                parse_options=pv_csv.ParseOptions(invalid_row_handler=skip_row),
                convert_options=convert_options,
            )
            try:
                for batch in reader:
                    table, dropped = self._cast_batch(pa.Table.from_batches([batch]), schema)
                    skipped["rows"] += dropped
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
                    writer.write_table(table)
                    rows += table.num_rows
                    _progress(label, raw.tell() / total_size, f"{rows:,} rows")
            finally:
                if writer is not None:
                    writer.close()
        if writer is None:
            raise RuntimeError("no data rows read")
        return rows, skipped["rows"]

    @staticmethod
    def _cast_batch(table: pa.Table, schema: dict):
        """Cast string columns to schema types (vectorized, no per-row validation).
        TODO: add better row-level processing (validate/repair bad values such as '11.3').
        Structurally broken CSV rows (e.g. '11,3') are still skipped by the CSV reader."""
        for col, typ in schema.items():
            if typ == "str" or col not in table.schema.names:
                continue
            target = ARROW_TYPES[typ]
            values = table[col]
            try:
                casted = pc.cast(values, target)
            except pa.ArrowInvalid:
                # e.g. '1.000000' in an integer column: go through float64
                casted = pc.cast(pc.round(pc.cast(values, pa.float64())), target)
            table = table.set_column(table.schema.get_field_index(col), col, casted)
        return table, 0

    # ------------------------------------------------------------------ Summary
    def performance_summary(self, symbol: str, points: int = 61) -> dict:
        """Summarize how `symbol` traded this month from the day_aggs_v1 files.
        Returns (and saves to data/summaries/<SYMBOL>_<YYYY-MM>.json) the date range, volume,
        number of trades, lowest/average/highest price, and a normal-distribution (bell curve)
        built from the daily closes: mean = average price, std = std of daily closes."""
        now = datetime.now()
        month, year = f"{now.month:02d}", str(now.year)
        symbol = symbol.upper().strip()
        src_dir = self.root_data_dir / "day_aggs_v1" / year / month
        files = sorted(src_dir.glob("*.csv.gz"))
        if not files:
            raise RuntimeError(f"No day_aggs_v1 files in {src_dir}")

        rows = []  # (date, volume, close, high, low, transactions)
        for path in files:
            with open(path, "rb") as raw, gzip.GzipFile(fileobj=raw) as gz:
                table = pv_csv.read_csv(gz)
            table = table.filter(pc.equal(table["ticker"], symbol))
            if table.num_rows == 0:
                continue
            r = table.to_pylist()[0]
            rows.append((path.name[:-7], r["volume"], r["close"], r["high"], r["low"], r["transactions"]))
        if not rows:
            raise RuntimeError(f"Symbol {symbol} not found in {src_dir}")

        closes = [r[2] for r in rows]
        n = len(closes)
        mean = sum(closes) / n
        std = (sum((c - mean) ** 2 for c in closes) / (n - 1)) ** 0.5 if n > 1 else 0.0
        low, high = min(r[4] for r in rows), max(r[3] for r in rows)

        # Bell curve over mean +/- 3 std (a single point if there is no spread)
        if std > 0:
            curve = []
            for i in range(points):
                x = mean + (-3 + 6 * i / (points - 1)) * std
                y = math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))
                curve.append({"price": round(x, 4), "density": round(y, 6)})
        else:
            curve = [{"price": round(mean, 4), "density": 1.0}]

        summary = {
            "symbol": symbol,
            "date_range": {"start": rows[0][0], "end": rows[-1][0], "trading_days": n},
            "volume": round(sum(r[1] for r in rows)),
            "number_of_trades": sum(r[5] for r in rows),
            "lowest_price": round(low, 4),
            "average_price": round(mean, 4),
            "highest_price": round(high, 4),
            "distribution": {
                "type": "normal",
                "mean": round(mean, 4),
                "std_dev": round(std, 4),
                "range_1_sigma": [round(mean - std, 4), round(mean + std, 4)],
                "range_2_sigma": [round(mean - 2 * std, 4), round(mean + 2 * std, 4)],
                "curve": curve,
            },
        }
        out = self.base_dir / "data" / "summaries" / f"{symbol}_{year}-{month}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        return summary

    # ------------------------------------------------------------------ Export
    def export_project(self, dest: Path | str, verify: bool = True) -> dict:
        """Copy this whole capstone folder to `dest` with rsync (re-runnable, only changes are copied).
        For the huge tick data (trades_v1, quotes_v1) only the earliest CSV of each is copied, plus its
        converted Parquet; everything else (day/minute aggregates, news, filings, code, docs, .env, .aws)
        is copied. __pycache__ is skipped. With verify=True every copied file is checksum-compared and any
        difference raises. Errors stop the operation and raise. Returns {dest, sample_files, verified}."""
        dest = Path(dest)
        if not dest.parent.exists():
            raise RuntimeError(f"Destination parent does not exist (drive not mounted?): {dest.parent}")
        rel = lambda p: p.relative_to(self.base_dir).as_posix()
        rules, samples = ["- __pycache__/"], []
        for data_type in ("trades_v1", "quotes_v1"):
            csvs = sorted(self.root_data_dir.joinpath(data_type).rglob("????-??-??.csv.gz"))
            if not csvs:
                continue
            keep = [csvs[0], self.converted_dir / data_type / f"{csvs[0].name[:-7]}.parquet"]
            keep = [p for p in keep if p.exists()]
            samples += [rel(p) for p in keep]
            for p in keep:  # include the parent folders (below base_dir), then the file itself
                rules += [f"+ /{rel(d)}/" for d in reversed(p.parents) if self.base_dir in d.parents]
                rules.append(f"+ /{rel(p)}")
            rules += [f"- /{rel(root)}/**" for root in (self.root_data_dir / data_type, self.converted_dir / data_type)]
        filters = [f"--filter={r}" for r in rules]
        base = ["rsync", "-a", "--partial", *filters, f"{self.base_dir}/", f"{dest}/"]
        try:
            subprocess.run(base, check=True)
            diff = []
            if verify:
                out = subprocess.run(["rsync", "-rc", "--dry-run", "-i", *filters, f"{self.base_dir}/", f"{dest}/"],
                                     check=True, capture_output=True, text=True).stdout.strip()
                diff = out.splitlines()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(f"Export failed: {e}") from e
        if diff:
            raise RuntimeError(f"Export verification found {len(diff)} differing files, e.g. {diff[0]}")
        return {"dest": str(dest), "sample_files": samples, "verified": verify}

    # ------------------------------------------------------------------ News
    def fetch_news(self, company: str | None = None, symbol: str | None = None,
                   keywords: list[str] | None = None, from_date: int = 30) -> dict:
        """Fetch news for `symbol` from `from_date` days ago until now (Finnhub + Polygon/Massive),
        keep articles whose headline/summary match the company name AND any keyword (whichever of
        the two are given), and append them to {NEWS_OUTPUT_DIR}/{SYMBOL}.txt.
        Articles already in the file (same URL) are skipped. API errors stop the operation and raise."""
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required (Finnhub and Polygon query by symbol)")
        if not isinstance(from_date, int) or isinstance(from_date, bool) or from_date < 1:
            raise ValueError("from_date must be a positive integer (days back from today)")
        symbol = symbol.strip().upper()
        keywords = [k.strip() for k in (keywords or []) if k and k.strip()]
        company = company.strip() if company and company.strip() else None
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=from_date)

        env = dotenv_values(self.base_dir / ".env")
        keys = self._api_keys(env)

        news_dir = self._news_dir(env)
        out_path = news_dir / f"{symbol}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seen = self._read_seen_urls(out_path)
        stats = {"symbol": symbol, "path": str(out_path), "fetched": 0, "matched": 0,
                 "written": 0, "duplicates": 0}

        print(f"News {symbol}: last {from_date} days ({start:%Y-%m-%d} -> {now:%Y-%m-%d})")
        with httpx.Client(timeout=httpx.Timeout(30.0)) as api, open(out_path, "a", encoding="utf-8") as out:
            sources = (
                self._iter_finnhub(api, keys["FINNHUB_API_KEY"], symbol, start, now),
                self._iter_polygon(api, keys["POLYGON_API_KEY"], symbol, start, now),
            )
            for source in sources:
                for page in source:
                    for art in page:
                        stats["fetched"] += 1
                        if art["published"] < start:
                            continue
                        if not self._matches(self._match_text(art), company, keywords):
                            continue
                        stats["matched"] += 1
                        key = self._url_key(art["url"])
                        if key in seen:
                            stats["duplicates"] += 1
                            continue
                        seen.add(key)
                        out.write(self._format_article(art, symbol, company))
                        out.flush()
                        stats["written"] += 1
                    sys.stdout.write(f"\r  fetched {stats['fetched']}  matched {stats['matched']}  "
                                     f"written {stats['written']}\033[K")
                    sys.stdout.flush()
        print(f"\nNews {symbol}: wrote {stats['written']} articles -> {out_path}")
        return stats

    def get_quote(self, symbol: str) -> dict:
        """The current price of `symbol` from Finnhub /quote (one call, paid key, nothing is written to disk).
        Returns {symbol, price, change, change_percent, high, low, open, previous_close, as_of_utc, source}.
        `as_of_utc` is when the last trade happened: outside market hours it is the last trade of the session,
        not "now". Finnhub answers an unknown symbol with all zeros, which is raised here as an error rather
        than returned as a price of 0. Errors (no key, bad symbol, API failure) raise."""
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required")
        symbol = symbol.strip().upper()
        key = self._api_keys()["FINNHUB_API_KEY"]
        with httpx.Client(timeout=httpx.Timeout(30.0)) as api:
            data = self._get_json(api, _RateLimiter(FINNHUB_MIN_INTERVAL), "Finnhub quote", FINNHUB_QUOTE_URL,
                                  {"symbol": symbol}, headers={"X-Finnhub-Token": key})
        if not isinstance(data, dict) or not data.get("c") or not data.get("t"):
            raise RuntimeError(f"No quote returned for {symbol} (Finnhub answers an unknown symbol with zeros)")
        return {"symbol": symbol, "price": round(float(data["c"]), 4),
                "change": round(float(data.get("d") or 0.0), 4), "change_percent": round(float(data.get("dp") or 0.0), 4),
                "high": round(float(data.get("h") or 0.0), 4), "low": round(float(data.get("l") or 0.0), 4),
                "open": round(float(data.get("o") or 0.0), 4), "previous_close": round(float(data.get("pc") or 0.0), 4),
                "as_of_utc": datetime.fromtimestamp(int(data["t"]), timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "source": "Finnhub /quote"}

    def fetch_press_releases(self, symbol: str, from_date: int = 365) -> dict:
        """Fetch the official company press releases (Finnhub /press-releases) for `symbol` from
        `from_date` days ago until now and append them, oldest first, to
        {PRESS_RELEASES_OUTPUT_DIR}/{SYMBOL}_PRs.txt (default data/press_releases/). Fully independent
        of fetch_news (own request, parser, file and record format). Releases already in the file
        (same id) are skipped, so reruns are safe. Errors stop the operation and raise.
        Returns {symbol, path, fetched, written, duplicates}."""
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required")
        if not isinstance(from_date, int) or isinstance(from_date, bool) or from_date < 1:
            raise ValueError("from_date must be a positive integer (days back from today)")
        symbol = symbol.strip().upper()
        key = self._api_keys()["FINNHUB_API_KEY"]
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=from_date)
        out_path = self._press_dir() / f"{symbol}{PRESS_RELEASES_SUFFIX}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                seen = {line[4:].strip() for line in f if line.startswith("id: ")}
        stats = {"symbol": symbol, "path": str(out_path), "fetched": 0, "written": 0, "duplicates": 0}

        print(f"Press releases {symbol}: last {from_date} days ({start:%Y-%m-%d} -> {now:%Y-%m-%d})")
        with httpx.Client(timeout=httpx.Timeout(30.0)) as api:
            data = self._get_json(
                api, _RateLimiter(FINNHUB_MIN_INTERVAL), "Finnhub press releases", FINNHUB_PRESS_URL,
                {"symbol": symbol, "from": f"{start:%Y-%m-%d}", "to": f"{now:%Y-%m-%d}"},
                headers={"X-Finnhub-Token": key})
        items = data.get("majorDevelopment", []) if isinstance(data, dict) else data
        releases = []
        for it in items:
            if not it.get("headline") or not it.get("datetime"):
                continue
            ts = it["datetime"]
            published = (datetime.fromtimestamp(ts, tz=timezone.utc) if isinstance(ts, (int, float))
                         else datetime.fromisoformat(str(ts).replace("Z", "+00:00").replace(" ", "T")))
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            rid = hashlib.sha1(f"{symbol}|{published.isoformat()}|{it['headline']}".encode()).hexdigest()[:16]
            releases.append((published, rid, it))
        with open(out_path, "a", encoding="utf-8") as out:
            for published, rid, it in sorted(releases, key=lambda r: r[0]):
                stats["fetched"] += 1
                if rid in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(rid)
                text = self._block(it.get("description"))
                out.write(f"<<<PRESS_RELEASE>>>\nid: {rid}\nsymbol: {symbol}\n"
                          f"published_utc: {published:%Y-%m-%dT%H:%M:%SZ}\nurl: {self._one_line(it.get('url'))}\n"
                          f"headline: {self._one_line(it['headline'])}\n<<<TEXT>>>\n{text}\n<<<END>>>\n")
                stats["written"] += 1
        print(f"Press releases {symbol}: {stats['fetched']} fetched, {stats['written']} written, "
              f"{stats['duplicates']} duplicates -> {out_path}")
        return stats

    def fetch_filings(self, symbol: str, from_date: int = 365, forms: tuple[str, ...] = FILING_FORMS) -> dict:
        """Fetch SEC EDGAR filings for `symbol` filed in the last `from_date` days from the Massive
        (Polygon) filings endpoints and append them, oldest first, to data/filings/{SYMBOL}.txt.
        One record per filing (accession number). Text is included where the API provides it:
        10-K (Business and Risk Factors sections) and 8-K (Items text). Other forms (10-Q ...) are
        listed with their SEC URL and no text. Reruns skip accession numbers already in the file.
        Fully independent of the news and press-release code. Errors stop the operation and raise.
        Returns {symbol, path, fetched, written, duplicates}."""
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required")
        if not isinstance(from_date, int) or isinstance(from_date, bool) or from_date < 1:
            raise ValueError("from_date must be a positive integer (days back from today)")
        symbol = symbol.strip().upper()
        key = self._api_keys()["POLYGON_API_KEY"]
        now = datetime.now(timezone.utc)
        since = f"{now - timedelta(days=from_date):%Y-%m-%d}"
        out_path = self._filings_dir() / f"{symbol}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                seen = {line[11:].strip() for line in f if line.startswith("accession: ")}
        stats = {"symbol": symbol, "path": str(out_path), "fetched": 0, "written": 0, "duplicates": 0}

        print(f"Filings {symbol}: {', '.join(forms)} filed since {since}")
        limiter = _RateLimiter(POLYGON_MIN_INTERVAL)

        def query(path: str, extra: dict, limit: int, sort: str = "filing_date.asc") -> list[dict]:
            params = {"ticker": symbol, "filing_date.gte": since, "sort": sort,
                      "limit": limit, "apiKey": key, **extra}
            results = self._get_json(client, limiter, "Massive filings", f"{FILINGS_URL}{path}", params)["results"]
            if len(results) >= limit:
                raise RuntimeError(f"{path} returned {limit}+ results for {symbol}; use a smaller from_date")
            return results

        with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
            filings: dict[str, dict] = {}
            for form in forms:
                for r in query("/vX/index", {"form_type": form}, 1000):
                    filings[r["accession_number"]] = {
                        "form": r["form_type"], "filed": r["filing_date"], "period_end": "",
                        "url": r["filing_url"], "sections": []}
            if "10-K" in forms:  # 10-K text comes as sections; match them to filings by URL
                by_url = {f["url"]: f for f in filings.values()}
                for r in query("/10-K/vX/sections", {}, 100, "period_end.asc"):
                    f = by_url.get(r["filing_url"])
                    if f is not None:
                        f["period_end"] = r.get("period_end", "")
                        f["sections"].append((r["section"], r["text"]))
            if "8-K" in forms:
                for r in query("/8-K/vX/text", {}, 100):
                    f = filings.get(r["accession_number"])
                    if f is not None:
                        f["sections"].append(("items", r["items_text"]))

        with open(out_path, "a", encoding="utf-8") as out:
            for acc, f in sorted(filings.items(), key=lambda kv: (kv[1]["filed"], kv[0])):
                stats["fetched"] += 1
                if acc in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(acc)
                out.write(self._filing_record(acc, symbol, f["form"], f["filed"], f["period_end"], f["url"], f["sections"]))
                stats["written"] += 1
        print(f"Filings {symbol}: {stats['fetched']} fetched, {stats['written']} written, "
              f"{stats['duplicates']} duplicates -> {out_path}")
        return stats

    def _filing_record(self, accession: str, symbol: str, form: str, filed: str, period_end: str, url: str, sections: list[tuple[str, str]]) -> str:
        """One filing record in the file's format (the contract with prometheus/rag/loaders.py): the header lines, <<<TEXT>>>, one `== name ==` block per section,
        <<<END>>>. `text_status` is `ok` exactly when there is text. Shared by fetch_filings (Massive) and fetch_filing_text (EDGAR)."""
        text = "\n\n".join(f"== {name} ==\n{self._block(body)}" for name, body in sections)
        return (f"<<<FILING>>>\naccession: {accession}\nsymbol: {symbol}\nform: {form}\nfiled: {filed}\nperiod_end: {period_end}\nurl: {self._one_line(url)}\n"
                f"text_status: {'ok' if text else 'not_available'}\n<<<TEXT>>>\n{text}\n<<<END>>>\n")

    def fetch_filing_text(self, symbol: str, forms: tuple[str, ...] = edgar_text.TEXT_FORMS, from_date: int = 365) -> dict:
        """Fetch from SEC EDGAR the TEXT of the stored filings of `forms` (the 10-Q) that have none, filed in the last `from_date` days, and append it to
        data/filings/{SYMBOL}.txt. The Massive API gives no text for a 10-Q; the record the file already holds says `text_status: not_available`. For each such
        filing the url of the record (EDGAR's complete submission) is downloaded, the 10-Q document is taken out of it and split into its Items (edgar_text.py), and a SECOND
        record of the same accession is appended with `text_status: ok` and the period of the report (the loader reads the last record that has text; the file stays append-only).
        A filing that already has text is skipped, so reruns are safe. Needs SEC_USER_AGENT (a name and a contact: EDGAR asks every client to say who it is): without it
        this raises EdgarNotConfigured, unless there is nothing to fetch. Every other error stops the operation and raises. Returns {symbol, path, candidates, written}."""
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required")
        if not isinstance(from_date, int) or isinstance(from_date, bool) or from_date < 1:
            raise ValueError("from_date must be a positive integer (days back from today)")
        unknown = [f for f in forms if f not in edgar_text.TEXT_FORMS]
        if not forms or unknown:
            raise ValueError(f"forms must be some of {', '.join(edgar_text.TEXT_FORMS)} (got {', '.join(unknown) or 'none'})")
        symbol = symbol.strip().upper()
        path = self._filings_dir() / f"{symbol}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist: fetch the filings of {symbol} first")
        since = f"{datetime.now(timezone.utc) - timedelta(days=from_date):%Y-%m-%d}"
        has_text, wanted = set(), {}
        for head, _ in self._read_records(path, "<<<FILING>>>"):
            if head.get("text_status") == "ok":
                has_text.add(head.get("accession"))
            elif head.get("form") in forms and head.get("filed", "") >= since and head.get("accession"):
                wanted.setdefault(head["accession"], head)
        candidates = [head for acc, head in wanted.items() if acc not in has_text]
        stats = {"symbol": symbol, "path": str(path), "candidates": len(candidates), "written": 0}
        print(f"Filing text {symbol}: {', '.join(forms)} with no text filed since {since}: {len(candidates)}")
        if not candidates:
            return stats
        agent = self._sec_user_agent()
        limiter = _RateLimiter(EDGAR_MIN_INTERVAL)
        with httpx.Client(timeout=httpx.Timeout(EDGAR_TIMEOUT), headers={"User-Agent": agent}) as client:
            for head in candidates:
                acc, url = head["accession"], head.get("url", "")
                if not url.startswith(edgar_text.SEC_PREFIX):
                    raise ValueError(f"{acc}: the url is not an EDGAR archive url, so nothing was sent to it")
                submission = self._get_bytes(client, limiter, f"EDGAR {acc}", url).decode("utf-8", errors="replace")
                facts = edgar_text.submission_facts(submission)
                if facts["accession"] != acc:
                    raise RuntimeError(f"EDGAR {acc}: the downloaded submission is for accession {facts['accession'] or '(none)'}")
                try:
                    sections = edgar_text.filing_sections(submission, head["form"])
                except edgar_text.EdgarLayoutError as error:
                    raise ValueError(f"{acc} ({head['form']}): {error}") from None
                with open(path, "a", encoding="utf-8") as out:  # one record at a time: a failure on the third 10-Q keeps the first two
                    out.write(self._filing_record(acc, symbol, head["form"], head["filed"], facts["period_end"], url, sections))
                stats["written"] += 1
                print(f"  {acc} {head['form']} filed {head['filed']}: {len(sections)} sections, {sum(len(b) for _, b in sections)} characters")
        print(f"Filing text {symbol}: {stats['written']} written -> {path}")
        return stats

    def _sec_user_agent(self, env: dict | None = None) -> str:
        """SEC_USER_AGENT from the environment or .env: a name and a contact (an e-mail), as EDGAR asks. Never has a default: raises EdgarNotConfigured."""
        env = env if env is not None else dotenv_values(self.base_dir / ".env")
        value = " ".join((os.environ.get("SEC_USER_AGENT") or env.get("SEC_USER_AGENT") or "").split())
        if not value:
            raise EdgarNotConfigured(f"SEC_USER_AGENT is not set in {self.base_dir / '.env'}: EDGAR needs a User-Agent that names you and gives a contact, "
                                     "for example SEC_USER_AGENT=Your Name you@example.com")
        if "@" not in value or not re.fullmatch(r"[ -~]+", value):
            raise EdgarNotConfigured("SEC_USER_AGENT must be printable ASCII text that contains a contact e-mail, for example Your Name you@example.com")
        return value

    @staticmethod
    def _get_bytes(client: httpx.Client, limiter: _RateLimiter, name: str, url: str) -> bytes:
        """GET a page and return its bytes: rate limited, backs off on 429/5xx, refuses a body over EDGAR_MAX_BYTES. The message never holds the identity header."""
        for attempt in range(1, API_MAX_ATTEMPTS + 1):
            limiter.wait()
            try:
                r = client.get(url, follow_redirects=False)
            except httpx.RequestError as e:
                if attempt == API_MAX_ATTEMPTS:
                    raise RuntimeError(f"{name} request failed: {type(e).__name__}") from None
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                if len(r.content) > EDGAR_MAX_BYTES:
                    raise RuntimeError(f"{name}: the response is {len(r.content)} bytes, over the limit of {EDGAR_MAX_BYTES}")
                return r.content
            if r.status_code in (429, 500, 502, 503, 504) and attempt < API_MAX_ATTEMPTS:
                time.sleep(5 * 2 ** (attempt - 1))
                continue
            raise RuntimeError(f"{name} HTTP {r.status_code}")
        raise RuntimeError(f"{name}: retries exhausted")

    def _api_keys(self, env: dict | None = None) -> dict:
        env = env if env is not None else dotenv_values(self.base_dir / ".env")
        keys = {}
        for name in ("FINNHUB_API_KEY", "POLYGON_API_KEY"):
            keys[name] = os.environ.get(name) or env.get(name)
            if not keys[name]:
                raise RuntimeError(f"{name} is not set in {self.base_dir / '.env'}")
        return keys

    def _news_dir(self, env: dict | None = None) -> Path:
        """NEWS_OUTPUT_DIR from .env (or environment); falls back to ./news. A relative path is
        resolved against the folder containing data_toolbox.py."""
        env = env if env is not None else dotenv_values(self.base_dir / ".env")
        d = Path(os.environ.get("NEWS_OUTPUT_DIR") or env.get("NEWS_OUTPUT_DIR") or "news").expanduser()
        return d if d.is_absolute() else self.base_dir / d

    def _press_dir(self, env: dict | None = None) -> Path:
        """PRESS_RELEASES_OUTPUT_DIR from .env (or environment); falls back to ./data/press_releases. Resolved like _news_dir."""
        env = env if env is not None else dotenv_values(self.base_dir / ".env")
        d = Path(os.environ.get("PRESS_RELEASES_OUTPUT_DIR") or env.get("PRESS_RELEASES_OUTPUT_DIR")
                 or PRESS_RELEASES_DEFAULT_DIR).expanduser()
        return d if d.is_absolute() else self.base_dir / d

    def _filings_dir(self, env: dict | None = None) -> Path:
        """FILINGS_OUTPUT_DIR from .env (or environment); falls back to ./data/filings (where the files always were). Resolved like _news_dir.
        The RAG loader reads the same setting (prometheus/rag/config.py), so the file this writes is the file it loads."""
        env = env if env is not None else dotenv_values(self.base_dir / ".env")
        d = Path(os.environ.get("FILINGS_OUTPUT_DIR") or env.get("FILINGS_OUTPUT_DIR") or f"data/{FILINGS_DIR}").expanduser()
        return d if d.is_absolute() else self.base_dir / d

    @staticmethod
    def _progress(label: str, frac: float, detail: str, t0: float) -> None:
        """One-line progress bar: label [####----] 42%  detail  elapsed m:ss  ETA m:ss."""
        def clock(s: float) -> str:
            return f"{int(s) // 60}:{int(s) % 60:02d}"
        frac = min(max(frac, 0.0), 1.0)
        elapsed = time.monotonic() - t0
        filled = int(frac * 30)
        eta = f"  ETA {clock(elapsed * (1 - frac) / frac)}" if frac > 0.01 else ""
        sys.stdout.write(f"\r  {label} [{'#' * filled}{'-' * (30 - filled)}] {frac * 100:3.0f}%  "
                         f"{detail}  elapsed {clock(elapsed)}{eta}\033[K")
        sys.stdout.flush()

    # ------------------------------------------------------- Market news store
    # Market-wide (ticker-less) news kept locally as one JSON-lines file per UTC publish day:
    #   {NEWS_OUTPUT_DIR}/market/YYYY-MM-DD.jsonl   plus   market/_coverage.json {"from", "to"}
    # sync_market_news() fills it incrementally; discover() searches it without any API calls.
    def sync_market_news(self, from_date: int = 7) -> dict:
        """Download market-wide news (Polygon, every page; Finnhub general + merger feeds, latest
        items) for the last `from_date` days into the local store. Already-synced time is skipped
        (the stored range is tracked in _coverage.json), articles are de-duplicated by URL, and a
        progress bar shows the Polygon download. Errors stop the operation and raise; re-running
        is safe. Returns {path, fetched, written, duplicates}."""
        if not isinstance(from_date, int) or isinstance(from_date, bool) or from_date < 1:
            raise ValueError("from_date must be a positive integer (days back from today)")
        keys = self._api_keys()
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=from_date)
        store = self._news_dir() / MARKET_DIR
        store.mkdir(parents=True, exist_ok=True)
        cov_path = store / "_coverage.json"
        cov = json.loads(cov_path.read_text()) if cov_path.exists() else None
        cov_from = datetime.fromisoformat(cov["from"]) if cov else None
        cov_to = datetime.fromisoformat(cov["to"]) if cov else None
        # Resume from the end of what is already stored (1h overlap) if it reaches back far enough.
        fetch_from = max(start, cov_to - timedelta(hours=1)) if cov and cov_from <= start else start
        seen = self._market_seen(store, fetch_from)
        stats = {"path": str(store), "fetched": 0, "written": 0, "duplicates": 0}

        print(f"Market news sync: {fetch_from:%Y-%m-%d %H:%M} -> {now:%Y-%m-%d %H:%M} UTC "
                f"({'incremental' if fetch_from > start else 'full'}) -> {store}")
        span = (now - fetch_from).total_seconds() or 1.0
        t0 = time.monotonic()
        with httpx.Client(timeout=httpx.Timeout(30.0)) as api:
            pages = 0
            for page in self._iter_polygon(api, keys["POLYGON_API_KEY"], None, fetch_from, now):
                pages += 1
                stats["fetched"] += len(page)
                self._market_write(store, page, seen, stats)
                last = page[-1]["published"] if page else fetch_from
                self._progress("Polygon", (last - fetch_from).total_seconds() / span,
                               f"{stats['fetched']:,} articles  {stats['written']:,} new  page {pages}", t0)
            sys.stdout.write("\n")
            for cat in FINNHUB_GENERAL_CATEGORIES:
                for page in self._iter_finnhub_general(api, keys["FINNHUB_API_KEY"], cat):
                    page = [a for a in page if a["published"] >= fetch_from]
                    stats["fetched"] += len(page)
                    self._market_write(store, page, seen, stats)
                print(f"  Finnhub {cat}: done  ({stats['written']:,} new in total)")
        cov_path.write_text(json.dumps({
            "from": (min(cov_from, start) if cov and cov_from else start).isoformat(),
            "to": now.isoformat()}))
        print(f"Market news sync: {stats['fetched']:,} fetched, {stats['written']:,} new, "
              f"{stats['duplicates']:,} duplicates")
        return stats

    def sync_cnbc_news(self) -> dict:
        """Read CNBC's public RSS feeds (CNBC_FEEDS) and append the items to the market news store
        ({NEWS_OUTPUT_DIR}/market/YYYY-MM-DD.jsonl) in the same article schema the Polygon and Finnhub
        articles use (api='cnbc-rss', publisher='CNBC', category=feed name). RSS gives only headline,
        short description, link, id and publish time, and no tickers, so tickers/keywords/insights are
        empty and search() finds these items by text. Feeds hold only the latest items (no history);
        run it regularly. De-duplicated by URL. Errors raise. Returns {path, fetched, written, duplicates}."""
        store = self._news_dir() / MARKET_DIR
        store.mkdir(parents=True, exist_ok=True)
        seen = self._market_seen(store, datetime.now(timezone.utc) - timedelta(days=30))
        stats = {"path": str(store), "fetched": 0, "written": 0, "duplicates": 0}
        ns = {"m": "http://search.cnbc.com/rss/2.0/modules/siteContentMetadata"}
        with httpx.Client(timeout=httpx.Timeout(30.0), follow_redirects=True) as api:
            for name, feed_id in CNBC_FEEDS.items():
                r = api.get(CNBC_RSS_URL, params={"partnerId": "wrss01", "id": feed_id})
                if r.status_code != 200:
                    raise RuntimeError(f"CNBC RSS '{name}' HTTP {r.status_code}")
                try:
                    items = ET.fromstring(r.content).findall("./channel/item")
                except ET.ParseError as e:
                    raise RuntimeError(f"CNBC RSS '{name}' is not valid XML: {e}") from None
                page = []
                for it in items:
                    url, title, pub = (it.findtext(t) for t in ("link", "title", "pubDate"))
                    if not url or not title or not pub:
                        continue
                    page.append({
                        "id": it.findtext("m:id", namespaces=ns) or it.findtext("guid") or url,
                        "api": "cnbc-rss", "type": it.findtext("m:type", namespaces=ns) or "article",
                        "published": parsedate_to_datetime(pub).astimezone(timezone.utc),
                        "publisher": "CNBC", "author": "", "url": url.strip(), "headline": title.strip(),
                        "summary": (it.findtext("description") or "").strip(), "category": name,
                        "image_url": "", "tickers": [], "keywords": [], "insights": []})
                stats["fetched"] += len(page)
                self._market_write(store, page, seen, stats)
                print(f"  CNBC {name}: {len(page)} items  ({stats['written']} new in total)")
        print(f"CNBC news sync: {stats['fetched']} fetched, {stats['written']} new, "
              f"{stats['duplicates']} duplicates -> {store}")
        return stats

    @staticmethod
    def _market_files(store: Path, start: datetime) -> list[Path]:
        first = f"{start:%Y-%m-%d}"
        return sorted(p for p in store.glob("????-??-??.jsonl") if p.stem >= first)

    def _market_seen(self, store: Path, start: datetime) -> set[str]:
        seen: set[str] = set()
        for path in self._market_files(store, start):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    seen.add(self._url_key(json.loads(line)["url"]))
        return seen

    def _market_write(self, store: Path, page: list[dict], seen: set[str], stats: dict) -> None:
        by_day: dict[str, list[str]] = {}
        for art in page:
            key = self._url_key(art["url"])
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            rec = dict(art, published=art["published"].isoformat())
            by_day.setdefault(f"{art['published']:%Y-%m-%d}", []).append(json.dumps(rec, ensure_ascii=False))
            stats["written"] += 1
        for day, lines in by_day.items():
            with open(store / f"{day}.jsonl", "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    def discover(self, phrase: str, from_date: int = 7, top: int = 10, refresh: bool = True) -> dict:
        """Find which companies a trend/keyword phrase (e.g. "weight loss", "GLP-1") is about.
        With refresh=True (default) it first runs sync_market_news(from_date) (cheap when already
        synced), then searches the local market-news store: articles whose headline, summary, API
        keywords or insight reasoning contain the phrase (whole word, case-insensitive) are kept and
        the tickers they mention ranked. If an article has Polygon insights, only the tickers with
        their own insight count (the ones the article is really about); otherwise all its tickers do.
        Returns {phrase, from_date, scanned, matched, tickers: [{symbol, articles, positive, neutral,
        negative, providers, headlines}]}, best first, at most `top`. Errors raise."""
        if not phrase or not phrase.strip():
            raise ValueError("phrase is required")
        if not isinstance(from_date, int) or isinstance(from_date, bool) or from_date < 1:
            raise ValueError("from_date must be a positive integer (days back from today)")
        if not isinstance(top, int) or isinstance(top, bool) or top < 1:
            raise ValueError("top must be a positive integer")
        phrase = phrase.strip()
        if refresh:
            self.sync_market_news(from_date)
        store = self._news_dir() / MARKET_DIR
        start = datetime.now(timezone.utc) - timedelta(days=from_date)
        files = self._market_files(store, start) if store.exists() else []
        if not files:
            raise RuntimeError(f"no market news stored in {store}; run sync_market_news() first")

        print(f"Discover '{phrase}': searching {len(files)} day files since {start:%Y-%m-%d}")
        found: dict[str, dict] = {}
        scanned = matched = 0
        t0 = time.monotonic()
        for n, path in enumerate(files, 1):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    art = json.loads(line)
                    art["published"] = datetime.fromisoformat(art["published"])
                    scanned += 1
                    if art["published"] < start or not self._matches(self._match_text(art), phrase, []):
                        continue
                    matched += 1
                    sentiments = {i["ticker"].upper(): (i.get("sentiment") or "").lower()
                                  for i in art["insights"] if i.get("ticker")}
                    for sym in set(sentiments) or {t.upper() for t in art["tickers"]}:
                        rec = found.setdefault(sym, {"symbol": sym, "articles": 0, "positive": 0,
                                                     "neutral": 0, "negative": 0,
                                                     "providers": set(), "headlines": []})
                        rec["articles"] += 1
                        if sentiments.get(sym) in ("positive", "neutral", "negative"):
                            rec[sentiments[sym]] += 1
                        rec["providers"].add(art["api"])
                        if len(rec["headlines"]) < 3:
                            rec["headlines"].append(art["headline"])
            self._progress("Searching", n / len(files), f"{n}/{len(files)} days  {scanned:,} scanned  {matched} matched", t0)

        ranked = sorted(found.values(), key=lambda r: (-r["articles"], r["symbol"]))[:top]
        for r in ranked:
            r["providers"] = sorted(r["providers"])
        print(f"\nDiscover '{phrase}': {matched} matching articles of {scanned:,} stored")
        for r in ranked:
            print(f"  {r['symbol']:<6} {r['articles']:>3} articles  +{r['positive']} ={r['neutral']} -{r['negative']}  "
                  f"{r['headlines'][0][:70]}")
        return {"phrase": phrase, "from_date": from_date, "scanned": scanned,
                "matched": matched, "tickers": ranked}

    def search(self, symbol: str, terms: list[str], from_date: int = 30, refresh: bool = True) -> dict:
        """Combined news + filings search for one ticker. With refresh=True (default) it first runs
        fetch_news(symbol, keywords=terms) and fetch_filings(symbol) for the window, then searches
        the stored files data: news/{SYMBOL}.txt and data/filings/{SYMBOL}.txt. A record matches when
        ANY term occurs (whole word, case-insensitive) in its text; filings without text (10-Q,
        14C, 144 ...) are searched on form and URL only, so they rarely match.
        Returns {symbol, terms, from_date, news: [{published, publisher, headline, url}],
        filings: [{filed, form, url, text_status}], filings_in_window}. Errors raise."""
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required")
        terms = [t.strip() for t in (terms or []) if t and t.strip()]
        if not terms:
            raise ValueError("at least one search term is required")
        if not isinstance(from_date, int) or isinstance(from_date, bool) or from_date < 1:
            raise ValueError("from_date must be a positive integer (days back from today)")
        symbol = symbol.strip().upper()
        if refresh:
            self.fetch_news(symbol=symbol, keywords=terms, from_date=from_date)
            self.fetch_filings(symbol, from_date)
            self.sync_cnbc_news()
        start = datetime.now(timezone.utc) - timedelta(days=from_date)

        news = []
        for head, body in self._read_records(self._news_dir() / f"{symbol}.txt", ARTICLE_START):
            if head.get("published_utc", "") >= f"{start:%Y-%m-%dT%H:%M:%SZ}" and \
                    self._matches(f"{head.get('headline', '')}\n{body}", None, terms):
                news.append({"published": head["published_utc"], "publisher": head.get("publisher", ""),
                             "headline": head.get("headline", ""), "url": head.get("url", "")})
        for path in self._market_files(self._news_dir() / MARKET_DIR, start):
            for line in path.read_text(encoding="utf-8").splitlines():
                art = json.loads(line)
                if art["api"] == "cnbc-rss" and self._matches(f"{art['headline']}\n{art['summary']}", None, terms):
                    when = datetime.fromisoformat(art["published"])
                    if when >= start:
                        news.append({"published": f"{when:%Y-%m-%dT%H:%M:%SZ}", "publisher": "CNBC",
                                     "headline": art["headline"], "url": art["url"]})
        filings, in_window = [], 0
        by_accession: dict[str, tuple[dict, str]] = {}
        for head, body in self._read_records(self._filings_dir() / f"{symbol}.txt", "<<<FILING>>>"):
            key = head.get("accession") or f"record {len(by_accession)}"
            known = by_accession.get(key)
            if known is None or head.get("text_status") == "ok" or known[0].get("text_status") != "ok":
                by_accession[key] = (head, body)  # a filing whose text was fetched later (entry 36) is listed once, with its text; a record without text never replaces one with text
        for head, body in by_accession.values():
            if head.get("filed", "") < f"{start:%Y-%m-%d}":
                continue
            in_window += 1
            if self._matches(f"{head.get('form', '')}\n{head.get('url', '')}\n{body}", None, terms):
                filings.append({"filed": head["filed"], "form": head["form"], "url": head.get("url", ""),
                                "text_status": head.get("text_status", "")})
        news.sort(key=lambda r: r["published"], reverse=True)
        filings.sort(key=lambda r: r["filed"], reverse=True)
        print(f"Search {symbol} {terms}: {len(news)} news, {len(filings)} filings "
              f"({in_window} filings in the last {from_date} days)")
        for n in news[:10]:
            print(f"  NEWS    {n['published'][:10]}  {n['publisher'][:14]:<14} {n['headline'][:80]}")
        for f in filings[:10]:
            print(f"  FILING  {f['filed']}  {f['form']:<10} {f['url']}")
        return {"symbol": symbol, "terms": terms, "from_date": from_date, "news": news,
                "filings": filings, "filings_in_window": in_window}

    @staticmethod
    def _read_records(path: Path, start_marker: str):
        """Yield (header dict, body text) for each record of a splittable text file (news, filings)."""
        if not path.exists():
            return
        head: dict | None = None
        body: list[str] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == start_marker:
                    head, body = {}, []
                elif head is None:
                    continue
                elif line == "<<<END>>>":
                    yield head, "\n".join(body)
                    head = None
                elif line.startswith("<<<"):
                    body.append("")  # section marker (SUMMARY, TEXT, INSIGHTS): header ends
                elif body or ": " not in line:
                    body.append(line)
                else:
                    k, _, v = line.partition(": ")
                    head[k] = v

    @staticmethod
    def _url_key(url: str) -> str:
        p = urlsplit(url.strip())
        if p.netloc.lower().endswith("finnhub.io"):  # Finnhub redirect links differ only by ?id=...
            return f"{p.netloc.lower()}{p.path.rstrip('/')}?{p.query}"
        return f"{p.netloc.lower()}{p.path.rstrip('/')}"

    def _read_seen_urls(self, path: Path) -> set[str]:
        seen: set[str] = set()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("url: "):
                        seen.add(self._url_key(line[5:]))
        return seen

    @staticmethod
    def _matches(text: str, company: str | None, keywords: list[str]) -> bool:
        def has(term: str) -> bool:
            return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE) is not None
        if company and not has(company):
            return False
        if keywords and not any(has(k) for k in keywords):
            return False
        return True

    @staticmethod
    def _get_json(client: httpx.Client, limiter: _RateLimiter, name: str, url: str,
                  params: dict, headers: dict | None = None) -> dict | list:
        """GET with rate limiting; backs off on 429/5xx (honours Retry-After). Never puts the URL
        (which may hold an API key) in an error message."""
        for attempt in range(1, API_MAX_ATTEMPTS + 1):
            limiter.wait()
            try:
                r = client.get(url, params=params, headers=headers)
            except httpx.RequestError as e:
                if attempt == API_MAX_ATTEMPTS:
                    raise RuntimeError(f"{name} request failed: {type(e).__name__}") from None
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504) and attempt < API_MAX_ATTEMPTS:
                retry_after = r.headers.get("Retry-After", "")
                time.sleep(float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 5 * 2 ** (attempt - 1))
                continue
            raise RuntimeError(f"{name} HTTP {r.status_code}: {r.text[:200]}")
        raise RuntimeError(f"{name}: retries exhausted")

    def _iter_finnhub(self, client, api_key, symbol, start, end):
        limiter = _RateLimiter(FINNHUB_MIN_INTERVAL)
        win_start = start
        while win_start <= end:
            win_end = min(win_start + timedelta(days=FINNHUB_WINDOW_DAYS - 1), end)
            items = self._get_json(
                client, limiter, "Finnhub", FINNHUB_NEWS_URL,
                {"symbol": symbol, "from": f"{win_start:%Y-%m-%d}", "to": f"{win_end:%Y-%m-%d}"},
                headers={"X-Finnhub-Token": api_key},
            )
            page = []
            for it in items:
                if not it.get("url") or not it.get("headline") or not it.get("datetime"):
                    continue
                page.append({
                    "id": f"finnhub:{it.get('id', '')}", "api": "finnhub", "type": "news",
                    "published": datetime.fromtimestamp(it["datetime"], tz=timezone.utc),
                    "publisher": it.get("source", ""), "author": "", "url": it["url"],
                    "headline": it["headline"], "summary": it.get("summary", ""),
                    "category": it.get("category", ""), "image_url": it.get("image", ""),
                    "tickers": [t.strip() for t in str(it.get("related", "")).split(",") if t.strip()],
                    "keywords": [], "insights": [],
                })
            yield page
            win_start = win_end + timedelta(days=1)

    def _iter_finnhub_general(self, client, api_key, category):
        """Market-wide Finnhub feed (no symbol, latest items only). `related` is often empty."""
        limiter = _RateLimiter(FINNHUB_MIN_INTERVAL)
        items = self._get_json(client, limiter, "Finnhub general", FINNHUB_GENERAL_URL,
                               {"category": category}, headers={"X-Finnhub-Token": api_key})
        page = []
        for it in items:
            if not it.get("url") or not it.get("headline") or not it.get("datetime"):
                continue
            page.append({
                "id": f"finnhub:{it.get('id', '')}", "api": "finnhub", "type": "news",
                "published": datetime.fromtimestamp(it["datetime"], tz=timezone.utc),
                "publisher": it.get("source", ""), "author": "", "url": it["url"],
                "headline": it["headline"], "summary": it.get("summary", ""),
                "category": it.get("category", ""), "image_url": it.get("image", ""),
                "tickers": [t.strip() for t in str(it.get("related", "")).split(",") if t.strip()],
                "keywords": [], "insights": [],
            })
        yield page

    def _iter_polygon(self, client, api_key, symbol, start, end):
        """Pages oldest to newest by moving the published_utc.gte bound forward (Polygon's next_url
        cursor repeats pages in practice), de-duplicating by article id. Raises if one full page
        holds nothing new (more than 1000 articles with the same timestamp)."""
        limiter = _RateLimiter(POLYGON_MIN_INTERVAL)
        end_s = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        gte = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        seen_ids: set[str] = set()
        while True:
            params = {"published_utc.gte": gte, "published_utc.lte": end_s,
                      "order": "asc", "sort": "published_utc", "limit": POLYGON_PAGE_SIZE, "apikey": api_key}
            if symbol:  # no ticker = market-wide news (used by sync_market_news)
                params["ticker"] = symbol
            results = self._get_json(client, limiter, "Polygon", POLYGON_NEWS_URL, params).get("results", [])
            page = []
            for it in results:
                if it.get("id") in seen_ids:
                    continue
                seen_ids.add(it.get("id"))
                if not it.get("article_url") or not it.get("title") or not it.get("published_utc"):
                    continue
                page.append({
                    "id": f"polygon:{it.get('id', '')}", "api": "polygon", "type": "news",
                    "published": datetime.fromisoformat(it["published_utc"].replace("Z", "+00:00")),
                    "publisher": (it.get("publisher") or {}).get("name", ""),
                    "author": it.get("author", "") or "", "url": it["article_url"],
                    "headline": it["title"], "summary": it.get("description", "") or "",
                    "category": "", "image_url": it.get("image_url", "") or "",
                    "tickers": it.get("tickers") or [], "keywords": it.get("keywords") or [],
                    "insights": it.get("insights") or [],
                })
            new_count = len(page)
            if results:
                yield page
            if len(results) < POLYGON_PAGE_SIZE:
                return
            if new_count == 0:
                raise RuntimeError(f"Polygon returned {POLYGON_PAGE_SIZE}+ articles at {gte}; cannot page past them")
            gte = results[-1]["published_utc"]

    @staticmethod
    def _match_text(art: dict) -> str:
        """Text the company/keyword filter looks at: headline, summary, API keywords and insight reasoning."""
        reasoning = " ".join(i.get("sentiment_reasoning", "") or "" for i in art["insights"])
        return f"{art['headline']}\n{art['summary']}\n{' '.join(art['keywords'])}\n{reasoning}"

    @staticmethod
    def _one_line(v) -> str:
        """Collapse whitespace so a value fits on one header line (shared by news and press releases)."""
        return re.sub(r"\s+", " ", str(v or "")).strip()

    @staticmethod
    def _block(v) -> str:
        """Keep text multi-line, but never let a line look like a record marker (shared)."""
        return re.sub(r"(?m)^<<<", " <<<", (v or "").strip())

    @staticmethod
    def _format_article(art: dict, symbol: str, company: str | None) -> str:
        one_line, block = DataToolbox._one_line, DataToolbox._block

        head = [
            f"id: {one_line(art['id'])}", f"symbol: {symbol}", f"company: {one_line(company)}",
            f"api: {art['api']}", f"type: {art['type']}", f"published_utc: {art['published']:%Y-%m-%dT%H:%M:%SZ}",
            f"publisher: {one_line(art['publisher'])}", f"author: {one_line(art['author'])}",
            f"url: {one_line(art['url'])}", f"image_url: {one_line(art['image_url'])}",
            f"category: {one_line(art['category'])}",
            f"tickers: {', '.join(one_line(t) for t in art['tickers'])}",
            f"keywords: {', '.join(one_line(k) for k in art['keywords'])}",
            f"headline: {one_line(art['headline'])}",
        ]
        insights = "\n".join(
            f"{one_line(i.get('ticker'))} | {one_line(i.get('sentiment'))} | {one_line(i.get('sentiment_reasoning'))}"
            for i in art["insights"])
        return (f"{ARTICLE_START}\n" + "\n".join(head) + f"\n{SUMMARY_START}\n{block(art['summary'])}\n"
                f"{INSIGHTS_START}\n{block(insights)}\n{ARTICLE_END}\n")


FINNHUB_METRIC_URL = "https://finnhub.io/api/v1/stock/metric"
# name -> (Finnhub 'metric' key, unit, description). Same sources the main bmd FundamentalTicker uses.
# The first ten are the original ones. The rest (Advisor v1 spec section 5.4, design choices entry 51) are Finnhub's own fields, kept as Finnhub gives them; a field Finnhub does
# not return for a company is None. `FINANCIAL_SCALE` converts a unit Finnhub quotes (market capitalization is in millions of dollars) and `FINANCIAL_FALLBACK` names the key
# to try when the first is absent (price to book is quoted quarterly and annually).
FINANCIAL_METRICS = {
    "pe_ratio": ("peTTM", "x", "Price/Earnings, trailing 12 months"),
    "eps": ("epsTTM", "USD", "Earnings per share, trailing 12 months"),
    "net_profit_margin": ("netProfitMarginTTM", "%", "Net income / revenue, trailing 12 months"),
    "gross_margin": ("grossMarginTTM", "%", "Gross income / revenue, trailing 12 months"),
    "operating_margin": ("operatingMarginTTM", "%", "Operating income / revenue, trailing 12 months"),
    "roe": ("roeTTM", "%", "Return on equity, trailing 12 months"),
    "debt_to_equity": ("totalDebt/totalEquityQuarterly", "ratio", "Total debt / total equity, latest quarter"),
    "current_ratio": ("currentRatioQuarterly", "x", "Current assets / current liabilities, latest quarter"),
    "dividend_yield": ("currentDividendYieldTTM", "%", "Dividends per share / price, trailing 12 months"),
    "revenue_growth": ("revenueGrowthTTMYoy", "%", "Revenue growth vs. a year earlier (TTM)"),
    "market_cap": ("marketCapitalization", "USD", "Market capitalization (Finnhub quotes millions; converted to dollars)"),
    "beta_provider": ("beta", "x", "Beta as Finnhub computes it"),
    "high_52w": ("52WeekHigh", "USD", "52-week high"),
    "low_52w": ("52WeekLow", "USD", "52-week low"),
    "adv_10d_provider": ("10DayAverageTradingVolume", "M shares", "Average daily volume over 10 days, in millions of shares"),
    "ps_ratio": ("psTTM", "x", "Price/Sales, trailing 12 months"),
    "pb_ratio": ("pbQuarterly", "x", "Price/Book, latest quarter (the annual figure when the quarterly one is absent)"),
    "payout_ratio_provider": ("payoutRatioTTM", "%", "Dividend payout ratio, trailing 12 months"),
    "eps_growth_ttm": ("epsGrowthTTMYoy", "%", "EPS growth vs. a year earlier (TTM)"),
}
FINANCIAL_SCALE = {"market_cap": 1_000_000}
FINANCIAL_FALLBACK = {"pb_ratio": "pbAnnual"}


def parse_metric_payload(symbol: str, raw: dict, as_of: str) -> dict:
    """Finnhub's `metric` object -> {symbol, as_of, metrics: {name: {value, unit, description}}}. A metric Finnhub has no value for is None. Pure: the same payload gives the same
    result, so a test can feed it a payload that was recorded from the real API."""
    metrics = {}
    for name, (field, unit, description) in FINANCIAL_METRICS.items():
        value = raw.get(field)
        if value is None and name in FINANCIAL_FALLBACK:
            value = raw.get(FINANCIAL_FALLBACK[name])
        value = None if isinstance(value, bool) or not isinstance(value, (int, float)) else float(value) * FINANCIAL_SCALE.get(name, 1)
        metrics[name] = {"value": round(value, 4) if value is not None else None, "unit": unit, "description": description}
    return {"symbol": symbol, "as_of": as_of, "metrics": metrics}


class Financials:
    """The commonly asked financial metrics (see FINANCIAL_METRICS: the original ten, and the ones the Advisor's strategies read), used through DataToolbox:
    agent.financials.get('AAPL')  or  agent.financials.pe_ratio('AAPL'). One Finnhub call per request."""

    def __init__(self, agent: DataToolbox):
        self.agent = agent

    def get(self, symbol: str) -> dict:
        """Return {symbol, as_of, metrics: {name: {value, unit, description}}}. A metric Finnhub has no
        value for is returned with value None. Errors (bad symbol, API failure) raise."""
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required")
        symbol = symbol.strip().upper()
        key = self.agent._api_keys()["FINNHUB_API_KEY"]
        with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
            data = self.agent._get_json(
                client, _RateLimiter(FINNHUB_MIN_INTERVAL), "Finnhub metric", FINNHUB_METRIC_URL,
                {"symbol": symbol, "metric": "all"}, headers={"X-Finnhub-Token": key})
        raw = data.get("metric") if isinstance(data, dict) else None
        if not raw:
            raise RuntimeError(f"No financial metrics returned for {symbol}")
        return parse_metric_payload(symbol, raw, f"{datetime.now(timezone.utc):%Y-%m-%d}")

    def _one(self, symbol: str, name: str):
        return self.get(symbol)["metrics"][name]["value"]

    def pe_ratio(self, symbol: str): return self._one(symbol, "pe_ratio")
    def eps(self, symbol: str): return self._one(symbol, "eps")
    def net_profit_margin(self, symbol: str): return self._one(symbol, "net_profit_margin")
    def gross_margin(self, symbol: str): return self._one(symbol, "gross_margin")
    def operating_margin(self, symbol: str): return self._one(symbol, "operating_margin")
    def roe(self, symbol: str): return self._one(symbol, "roe")
    def debt_to_equity(self, symbol: str): return self._one(symbol, "debt_to_equity")
    def current_ratio(self, symbol: str): return self._one(symbol, "current_ratio")
    def dividend_yield(self, symbol: str): return self._one(symbol, "dividend_yield")
    def revenue_growth(self, symbol: str): return self._one(symbol, "revenue_growth")


def main() -> int:
    agent = DataToolbox()
    try:
        agent.init_aws_cli()
        agent.fetch_historical_data()
        agent.convert_to_parquet()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
