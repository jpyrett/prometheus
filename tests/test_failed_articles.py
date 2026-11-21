"""A news article that FAILS is discarded completely (Jacob, 2026-11-15; design choices entry 83).

"If a news article fails (cannot be processed due to formatting etc), then skip and discard it (ignore it completely) from the RAG and returned scoring and other results to the Advisor agent. Do not use failed news articles."

The loaders and the ingest are tested in test_rag_loaders.py, test_press_releases.py and test_rag_ingest.py. This file is the other end, the DataToolbox: what it does with an item an API gives that cannot be made into
an article, and with a line of the market-news store that is not a whole article. The API answers are REAL records (the fixtures of the loader tests, `tests/fixtures/news_sample.txt` and
`press_releases_sample.txt`, and the first lines of a real day file of `news/market/`) put back into the shape each API returns, with malformed items added among them. No network, no key (a fake key in the
environment), and every folder is a temporary one.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prometheus.agents.data_clerk import data_toolbox
from prometheus.agents.data_clerk.data_toolbox import DataToolbox
from prometheus.rag.config import news_dir
from prometheus.rag.loaders import load_news, load_press_releases

FIXTURES = Path(__file__).parent / "fixtures"
NEWS = FIXTURES / "news_sample.txt"
PRESS = FIXTURES / "press_releases_sample.txt"
STAMP = "%Y-%m-%dT%H:%M:%SZ"


def real_records(path: Path, marker: str) -> list[tuple[dict, str, str]]:
    """(header fields, summary text, insights text) of each REAL record of a fixture."""
    records = []
    for record in path.read_text(encoding="utf-8").split(f"{marker}\n")[1:]:
        head_text, _, rest = record.partition("<<<SUMMARY>>>\n")
        summary, _, insights = rest.partition("<<<INSIGHTS>>>\n")
        records.append((dict(line.split(": ", 1) for line in head_text.strip().splitlines() if ": " in line), summary.strip(), insights.replace("<<<END>>>", "").strip()))
    return records


def epoch(text: str) -> int:
    return int(datetime.strptime(text, STAMP).replace(tzinfo=timezone.utc).timestamp())


def finnhub_item(head: dict, summary: str) -> dict:
    """A real Finnhub record put back into the shape /company-news returns."""
    return {"id": int(head["id"].split(":")[1]), "url": head["url"], "headline": head["headline"], "datetime": epoch(head["published_utc"]), "source": head.get("publisher", ""),
            "summary": summary, "category": head.get("category", ""), "image": head.get("image_url", ""), "related": head["tickers"]}


def polygon_item(head: dict, summary: str, insights: str) -> dict:
    """A real Polygon record put back into the shape /v2/reference/news returns."""
    return {"id": head["id"].split(":")[1], "article_url": head["url"], "title": head["headline"], "published_utc": head["published_utc"], "publisher": {"name": head.get("publisher", "")},
            "author": head.get("author", ""), "description": summary, "image_url": head.get("image_url", ""), "tickers": [t.strip() for t in head["tickers"].split(",")], "keywords": [],
            "insights": [dict(zip(("ticker", "sentiment", "sentiment_reasoning"), (part.strip() for part in line.split("|", 2)))) for line in insights.splitlines() if line.count("|") >= 2]}


@pytest.fixture
def toolbox(tmp_path, monkeypatch):
    """The real toolbox with the two news APIs answered from the real records (plus whatever `box.finnhub` and `box.polygon` hold for one test), once each (a window after the first gets nothing)."""
    monkeypatch.setenv("NEWS_OUTPUT_DIR", str(tmp_path / "news"))
    monkeypatch.setenv("PRESS_RELEASES_OUTPUT_DIR", str(tmp_path / "prs"))
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key-not-real")
    monkeypatch.setenv("POLYGON_API_KEY", "test-key-not-real")
    first, second, third = real_records(NEWS, "<<<ARTICLE>>>")
    box = DataToolbox(base_dir=tmp_path)
    box.finnhub = [finnhub_item(first[0], first[1]), finnhub_item(third[0], third[1])]
    box.polygon = [polygon_item(*second)]
    box.press = {"symbol": "AAPL", "majorDevelopment": []}
    served = set()

    def fake_get_json(client, limiter, name, url, params, headers=None):
        if url == data_toolbox.FINNHUB_NEWS_URL:
            return box.finnhub if "finnhub" not in served and not served.add("finnhub") else []
        if url == data_toolbox.POLYGON_NEWS_URL:
            return {"results": box.polygon if "polygon" not in served and not served.add("polygon") else []}
        if url == data_toolbox.FINNHUB_PRESS_URL:
            return box.press
        raise AssertionError(f"unexpected call {url}")

    monkeypatch.setattr(DataToolbox, "_get_json", staticmethod(fake_get_json))
    return box


def written_ids(tmp_path: Path, symbol: str = "AAPL") -> list[str]:
    skipped = []
    ids = [d.id for d in load_news(tmp_path / "news" / f"{symbol}.txt", skipped)]
    assert skipped == []                                                                     # what the toolbox writes always loads: nothing it wrote is a failed record
    return ids


@pytest.mark.parametrize("which", ["finnhub", "polygon"])
def test_an_item_the_api_gave_that_cannot_be_made_into_an_article_is_discarded_and_the_rest_are_written(toolbox, tmp_path, capsys, which):
    good = {"finnhub": toolbox.finnhub[0], "polygon": toolbox.polygon[0]}[which]
    broken = [
        {**good, "id": 1, "url": "https://example.com/1", **({"datetime": "garbage"} if which == "finnhub" else {"published_utc": "garbage"})},     # a time that is not a time
        {**good, "id": 2, "url": "https://example.com/2", **({"datetime": 10 ** 30} if which == "finnhub" else {"published_utc": 12345})},         # a time out of range, or of the wrong kind
        {**good, "id": 3, "url": "https://example.com/3", **({"datetime": [1, 2]} if which == "finnhub" else {"article_url": ["not", "text"]})},    # a field of the wrong kind
    ]
    getattr(toolbox, which).extend(broken)
    stats = toolbox.fetch_news(symbol="AAPL", from_date=60)
    capsys.readouterr()
    assert stats["failed"] >= 2 and stats["written"] == 3                                      # the failed ones were counted and not written; the three real articles were
    text = (tmp_path / "news" / "AAPL.txt").read_text(encoding="utf-8")
    assert "example.com/1" not in text and "example.com/2" not in text and sorted(written_ids(tmp_path)) == sorted(d.id for d in load_news(NEWS))


def test_a_press_release_that_cannot_be_read_is_discarded_and_the_real_ones_are_written(toolbox, tmp_path, capsys):
    real = [{"symbol": head["symbol"], "datetime": epoch(head["published_utc"]), "headline": head["headline"], "description": summary, "url": head["url"]}
            for head, summary, _ in ((h, rest.rsplit("\n<<<END>>>", 1)[0], "") for h, rest, _ in
                                     [(dict(line.split(": ", 1) for line in part.partition("<<<TEXT>>>\n")[0].strip().splitlines()), part.partition("<<<TEXT>>>\n")[2], "")
                                      for part in PRESS.read_text(encoding="utf-8").split("<<<PRESS_RELEASE>>>\n")[1:]])]
    toolbox.press = {"symbol": "AAPL", "majorDevelopment": [*real, {**real[0], "datetime": "not a time", "headline": "never written A"}, {**real[0], "datetime": 10 ** 30, "headline": "never written B"}]}
    stats = toolbox.fetch_press_releases("AAPL", from_date=400)
    capsys.readouterr()
    assert (stats["written"], stats["failed"]) == (3, 2)
    written = load_press_releases(tmp_path / "prs" / "AAPL_PRs.txt")
    assert [d.id for d in written] == [d.id for d in load_press_releases(PRESS)] and not any("never written" in d.page_content for d in written)


# ---------------------------------------------------------------- the market-news store (JSON lines), read by `discover` and `search`

def real_market_lines(count: int = 25) -> tuple[str, list[str]]:
    """(the day of the first real day file, its first `count` real lines)."""
    path = sorted((news_dir() / "market").glob("????-??-??.jsonl"))
    if not path:
        pytest.skip("no stored market news on this machine")
    lines = [line for line in path[-1].read_text(encoding="utf-8").splitlines() if line.strip()][:count]
    return path[-1].stem, lines


def market_store(tmp_path: Path, monkeypatch, lines: list[str], day: str) -> DataToolbox:
    folder = tmp_path / "news" / "market"
    folder.mkdir(parents=True)
    (folder / f"{day}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("NEWS_OUTPUT_DIR", str(tmp_path / "news"))
    return DataToolbox(base_dir=tmp_path)


def damaged(line: str) -> list[str]:
    """Real line, damaged in the ways a stored line can be: cut off, not JSON, a field gone, a time that is not a time, a list that is not a list."""
    art = json.loads(line)
    return [line[: len(line) // 2], "not json at all", json.dumps({k: v for k, v in art.items() if k != "headline"}), json.dumps({**art, "published": "garbage"}),
            json.dumps({**art, "insights": "positive"}), json.dumps({**art, "tickers": None}), json.dumps(["a", "list"])]


def test_the_seen_urls_of_a_market_store_ignore_a_damaged_line_instead_of_stopping_the_sync(tmp_path, monkeypatch):
    day, lines = real_market_lines(10)
    box = market_store(tmp_path, monkeypatch, [*damaged(lines[0]), *lines], day)
    seen = box._market_seen(tmp_path / "news" / "market", datetime(2000, 1, 1))
    assert seen == {box._url_key(json.loads(line)["url"]) for line in lines}
