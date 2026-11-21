"""The as-of clock (design choices entry 127, design/proving_ground_spec.md section 4): while `trading_clock.as_of(D)` is open, every door of the desk returns nothing dated after D.

REAL DATA ONLY. D is Monday 2026-11-09 (and, for the market-wide news, which is stored for October only, Tuesday 2026-10-20). The text doors are tried on the REAL Chroma index (a COPY of `data/processed/vectorstore`:
the real one is never opened by a test), the filing evidence on the real stored filings, the toolbox's stored-text search on the real news and filings files, the quote and the metrics on the real stored daily bars, the ledger
on real record lines copied from `data/ledger/recommendations.jsonl`, the Advisor on a real capture (`tests/fixtures/advisor_real`), and the Critic on a real recommendation of a real session
(`tests/fixtures/session_real`). Each test first shows that data exists AFTER D (the door has something to refuse), then that the bound door returns none of it. Nothing here calls a model or a paid API: the one
Finnhub call a review makes is made to fail on purpose (a tripwire), and the fetchers' windows are checked on the arguments they hand to the provider's call, which is replaced by a recorder.

Unbound, every door is exactly what it was: the existing tests of the retriever, the tools, the toolbox, the ledger, the Advisor and the Critic are the proof and pass unchanged.
"""
import copy
import json
import re
import shutil
import threading
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest
from advisor_helpers import Scripted
from critic_helpers import ReviewerModel, view
from test_advisor_sndk_pg import FIXTURES as ADVISOR_CAPTURES
from test_advisor_sndk_pg import RecordedClerk

from prometheus.agents.advisor.advisor import Advisor
from prometheus.agents.critic.critic import Critic
from prometheus.agents.data_clerk.bars import BarStore
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.agents.data_clerk.data_toolbox import Financials, DataToolbox, fetch_window
from prometheus.common import trading_clock as tc
from prometheus.common import work_order as wo
from prometheus.memory.ledger import Ledger
from prometheus.rag import retriever
from prometheus.rag.config import REPO_ROOT
from prometheus.rag.store import get_embeddings
from prometheus.tools.data_clerk import DATA_CLERK_TOOLS, DataClerkTools, ToolInputError, ToolLimitError

FIXTURES = Path(__file__).parent / "fixtures"
D = date(2026, 11, 9)                                                                  # a Monday: news, press releases, filings and bars exist on both sides of it
END_OF_D = datetime(2026, 11, 10, 4, 59, 59, 999999, tzinfo=timezone.utc)              # 23:59:59.999999 in New York (EST): the horizon
D_MARKET = date(2026, 10, 20)                                                          # the market-wide news is stored from 2026-10-13 to 2026-10-24
STORE = REPO_ROOT / "data" / "processed" / "vectorstore"
BARS = REPO_ROOT / "data" / "historical" / "day_aggs_v1"


def stamp(hit) -> datetime:
    return datetime.fromisoformat(hit.published_utc.replace("Z", "+00:00"))


def need_bars():
    if not (BARS / "2026" / "11" / "2026-11-09.csv.gz").exists():
        pytest.skip("the stored daily bars of 2026-11-09 are not on this machine")


# ---------------------------------------------------------------- the clock itself


def test_a_day_that_is_not_a_past_trading_day_is_refused_before_anything_is_bound():
    far = date.today() + timedelta(days=400)
    for day, why in ((date(2026, 11, 8), "weekend"), (date(2026, 11, 26), "Thanksgiving"), (far, "future"), (datetime(2026, 11, 9, 10, 0), "date")):
        with pytest.raises(ValueError):
            with tc.as_of(day):
                pytest.fail(f"as_of accepted {day} ({why})")
        assert tc.bound_day() is None
    with tc.as_of(D), pytest.raises(ValueError, match="already working as of 2026-11-09"):
        with tc.as_of(date(2026, 11, 10)):
            pass


# ---------------------------------------------------------------- the context follows the threads that copy it


def test_every_thread_site_of_the_package_copies_the_context_or_runs_no_agent_code():
    """`ThreadPoolExecutor`, `threading.Thread`: each site under prometheus/ either copies the context (`copy_context`) or is one of the two that start a run's own thread, which binds the day itself, and a poller that
    only reads the account. A new site that does neither fails here."""
    runs_own_day = {"application/web/runs.py", "application/web/desk_run.py"}
    for path in sorted((REPO_ROOT / "prometheus").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"ThreadPoolExecutor\(|threading\.Thread\(|\(threading\.Thread\)|run_in_executor|to_thread\(", text):
            assert "copy_context" in text or path.relative_to(REPO_ROOT / "prometheus").as_posix() in runs_own_day, f"{path} starts a thread without copying the context"


# ---------------------------------------------------------------- the text index: a COPY of the real one

@pytest.fixture(scope="module")
def real_embeddings():
    try:
        return get_embeddings()
    except Exception as error:   # no local copy of the model and no network
        pytest.skip(f"the embedding model is not available: {type(error).__name__}")


@pytest.fixture(scope="module")
def real_index(real_embeddings, tmp_path_factory):
    """The real stored index, COPIED to a temporary folder and opened there (Chroma writes to the folder it opens: the real one is never opened by a test)."""
    if not (STORE / "chroma.sqlite3").exists():
        pytest.skip("the real vector store is not on this machine")
    folder = tmp_path_factory.mktemp("vectorstore")
    shutil.copytree(STORE, folder / "index")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VECTOR_STORE_PATH", str(folder / "index"))
        yield folder / "index"
    shutil.rmtree(folder, ignore_errors=True)


def newest(search, symbol, **options):
    return search(f"{symbol} latest news and announcements", tickers=symbol, k=20, min_score=None, newest_first=True, **options)


def test_news_after_the_bound_day_is_never_returned_and_the_window_counts_back_from_it(real_index):
    for symbol in ("MSFT", "XOM"):
        assert any(stamp(h) > END_OF_D for h in newest(retriever.search_news, symbol)), f"the index has no {symbol} news after {D}: pick another symbol"
        with tc.as_of(D):
            bound = newest(retriever.search_news, symbol)
            assert bound and all(stamp(h) <= END_OF_D for h in bound)
            later = newest(retriever.search_news, symbol, until=datetime(2026, 11, 20, tzinfo=timezone.utc))      # a later `until` is cut to the horizon
            assert later and all(stamp(h) <= END_OF_D for h in later)
            recent = newest(retriever.search_news, symbol, since_days=2, now=datetime(2026, 11, 20, tzinfo=timezone.utc))
            assert recent and all(END_OF_D - timedelta(days=2) <= stamp(h) <= END_OF_D for h in recent)         # `now` is cut too: two days counted back from D, not from the real today
    assert all(stamp(h) <= datetime(2026, 11, 6, 23, 59, 59, tzinfo=timezone.utc) for h in newest(retriever.search_news, "MSFT", until=datetime(2026, 11, 6, 23, 59, 59, tzinfo=timezone.utc)))
    with pytest.raises(ValueError, match="until must be a datetime with a timezone"):
        newest(retriever.search_news, "MSFT", until=datetime(2026, 11, 6))


def test_a_filing_has_a_date_so_the_day_after_is_not_known_even_though_it_is_before_the_new_york_midnight_in_utc(real_index):
    """A filing is stamped 00:00 UTC of its filing date. The end of the New York day is already the NEXT day in UTC, so the horizon alone would let a filing of 2026-11-10 through on 2026-11-09 (found on Costco's real
    8-K of 2026-11-10): the filings collection is cut at 23:59:59 UTC of the bound day, and a filing of the bound day itself (Apple's of 2026-11-09) counts."""
    day_after = [h for h in newest(retriever.search_filings, "COST") if h.published_utc[:10] == "2026-11-10"]
    assert day_after and stamp(day_after[0]) < END_OF_D                                                  # the leak the horizon alone would have: 00:00 UTC of the 10th is before 04:59 UTC of the 10th
    with tc.as_of(D):
        assert all(h.published_utc[:10] <= "2026-11-09" for h in newest(retriever.search_filings, "COST"))
        assert any(h.published_utc[:10] == "2026-11-09" for h in newest(retriever.search_filings, "AAPL"))    # filed on D: known on D


# ---------------------------------------------------------------- the filing evidence and the stored-text search (the real stored files)


def test_the_toolbox_searches_the_stored_text_as_of_the_bound_day_and_never_refreshes(monkeypatch, capsys):
    toolbox = DataToolbox(base_dir=REPO_ROOT)
    free = toolbox.search("XOM", ["the"], from_date=90, refresh=False)
    capsys.readouterr()
    assert any(n["published"] > "2026-11-10T04:59:59Z" for n in free["news"])                          # XOM has news after D, and a filing after D too (counted in the window, though no text of it matches)
    for name in ("fetch_news", "fetch_filings", "sync_cnbc_news", "sync_market_news"):
        monkeypatch.setattr(DataToolbox, name, lambda *a, **k: pytest.fail("a past day refreshed the stored text"))
    with tc.as_of(D):
        bound = toolbox.search("XOM", ["the"], from_date=90)                                        # refresh=True by default, and not done
        capsys.readouterr()
        assert bound["news"] and all(n["published"] <= "2026-11-10T04:59:59Z" for n in bound["news"])
        assert bound["filings"] and all(f["filed"] <= D.isoformat() for f in bound["filings"]) and bound["filings_in_window"] < free["filings_in_window"]
    capsys.readouterr()


def test_the_market_wide_news_is_discovered_as_of_the_bound_day(capsys):
    toolbox = DataToolbox(base_dir=REPO_ROOT)
    horizon = tc.end_of_day(D_MARKET)

    def matching(start, cutoff):
        """The stored market articles published from `start` to `cutoff` that name "oil" (headline, summary, keywords or an insight's reasoning), counted straight from the day files."""
        count = 0
        for path in sorted((REPO_ROOT / "news" / "market").glob("????-??-??.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                art = json.loads(line) if line.strip() else None
                if art and all(art.get(k) is not None for k in ("url", "headline", "published", "api", "summary", "tickers", "keywords", "insights")):
                    text = f"{art['headline']}\n{art['summary']}\n{' '.join(art['keywords'])}\n{' '.join(i.get('sentiment_reasoning') or '' for i in art['insights'])}"
                    count += start <= datetime.fromisoformat(art["published"]) <= cutoff and re.search(r"(?<!\w)oil(?!\w)", text, re.I) is not None
        return count

    window_start = horizon - timedelta(days=30)
    bound_count = matching(window_start, horizon)
    assert 0 < bound_count < matching(window_start, datetime(2026, 10, 25, tzinfo=timezone.utc))      # there are articles about oil after D_MARKET
    with tc.as_of(D_MARKET):
        found = toolbox.discover("oil", from_date=30, top=5)                                        # refresh=True by default, and not done: no key is read, nothing is fetched
    capsys.readouterr()
    assert found["matched"] == bound_count and found["tickers"]


def test_this_months_price_summary_is_the_summary_of_the_bound_day_and_writes_nothing(tmp_path):
    need_bars()
    toolbox = DataToolbox(base_dir=REPO_ROOT)
    summary_file = REPO_ROOT / "data" / "summaries" / "XOM_2026-11.json"
    before = summary_file.stat().st_mtime_ns if summary_file.exists() else None
    with tc.as_of(D):
        summary = toolbox.performance_summary("XOM")
    assert summary["date_range"]["end"] == D.isoformat() and summary["date_range"]["start"] == "2026-11-02"
    assert (summary_file.stat().st_mtime_ns if summary_file.exists() else None) == before


# ---------------------------------------------------------------- the bars, the quote and the metrics (the real stored bars)

def day_bar(symbol, day):
    bars = DataClerkTools().daily_bars([symbol], day, day)["bars"][symbol]
    assert len(bars) == 1, f"no stored {symbol} bar on {day}"
    return bars[0]


def test_a_quote_as_of_the_bound_day_is_that_days_stored_close_marked_simulated_and_the_live_one_is_never_asked(monkeypatch):
    need_bars()

    def asked(self, symbol):
        raise AssertionError("Finnhub was asked for a live quote")

    monkeypatch.setattr(DataToolbox, "get_quote", asked)
    close = day_bar("PG", D)["close"]
    order = wo.new_work_order("prices", requester="account_manager", action_type="fetch_quote", subject=["PG", "MSFT"], live=True)
    clerk = DataClerk(toolbox=DataToolbox(base_dir=REPO_ROOT), model=object())
    with tc.as_of(D):
        clerk.request_quotes(order)
        tool_quote = clerk.tools.get_quote("PG")                                                    # the model's own `get_quote` tool answers the same
        with pytest.raises(ToolInputError, match="the desk works as of 2026-11-09"):
            clerk.tools.close_quote("PG", date(2026, 11, 10))                                       # the close of a later day is the future
    assert order.status == "data_ready" and order.data.quotes["PG"]["price"] == round(close, 4) == tool_quote["price"]
    assert order.data.quotes["MSFT"]["price"] == round(day_bar("MSFT", D)["close"], 4)
    assert all(q["simulated"] is True and q["as_of_utc"] == "2026-11-09T20:29:00Z" for q in order.data.quotes.values())
    assert "market close of 2026-11-09" in order.data.notes[0]


def test_the_metrics_are_computed_as_of_the_bound_day_whatever_date_is_asked_for():
    need_bars()
    tools = DataClerkTools()
    unbound = tools.metrics_for(["PG"], [{"name": "last_close"}], date(2026, 11, 19))
    with tc.as_of(D):
        bound = tools.metrics_for(["PG"], [{"name": "last_close"}], date(2026, 11, 19))             # a caller that passes the real today is cut to D
        no_date = tools.calculate_metrics(["PG"], ["last_close"])
        newest = BarStore(BARS).load(["PG"]).bars("PG")[-1].date                                    # no `end` while bound: not the newest stored day
    assert unbound["as_of"] == "2026-11-19" and unbound["results"]["PG"]["last_close"]["as_of"] > D.isoformat()
    assert bound["as_of"] == no_date["as_of"] == D.isoformat() and bound["results"]["PG"]["last_close"]["as_of"] == D.isoformat() == newest.isoformat()
    assert bound["results"]["PG"]["last_close"]["value"] == day_bar("PG", D)["close"]


def test_only_the_trader_and_the_follower_ask_for_a_range_of_bars_not_the_agents_the_model_or_the_critic():
    """`request_bars` with an explicit range is the one door to the FUTURE of the day (the backtest's next open, the follower's path). Nothing the desk researches with may call it: the Advisor, the Critic and the model's
    own tools never name it."""
    for folder in ("advisor", "critic"):
        for path in (REPO_ROOT / "prometheus" / "agents" / folder).glob("*.py"):
            assert not re.search(r"request_bars|daily_bars", path.read_text(encoding="utf-8")), path
    assert not {"daily_bars", "request_bars", "close_quote"} & set(DATA_CLERK_TOOLS)


# ---------------------------------------------------------------- nothing is fetched while a day is bound

def test_a_past_day_never_fetches_at_any_door(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("a past day reached the network")

    monkeypatch.setattr(DataToolbox, "_api_keys", boom)
    monkeypatch.setattr(DataToolbox, "_run", boom)
    toolbox = DataToolbox(base_dir=REPO_ROOT)
    calls = {"fetch_news": lambda: toolbox.fetch_news(symbol="PG"), "fetch_press_releases": lambda: toolbox.fetch_press_releases("PG"), "fetch_filings": lambda: toolbox.fetch_filings("PG"),
             "fetch_filing_text": lambda: toolbox.fetch_filing_text("PG"), "sync_market_news": lambda: toolbox.sync_market_news(), "sync_cnbc_news": lambda: toolbox.sync_cnbc_news(),
             "fetch_daily_bars": lambda: toolbox.fetch_daily_bars(["PG"]), "fetch_splits": lambda: toolbox.fetch_splits(["PG"]), "get_quote": lambda: toolbox.get_quote("PG"),
             "fetch_historical_data": lambda: toolbox.fetch_historical_data(), "sync_day_aggs": lambda: toolbox.sync_day_aggs(["2026-11"])}
    tools = DataClerkTools(toolbox=toolbox)
    tool_calls = {"fetch_news": lambda: tools.fetch_news("PG"), "fetch_press_releases": lambda: tools.fetch_press_releases("PG"), "fetch_filings": lambda: tools.fetch_filings("PG")}
    clerk = DataClerk(toolbox=toolbox, model=object(), allow_fetch=True)
    with tc.as_of(D):
        for name, call in calls.items():
            with pytest.raises(RuntimeError, match=f"the Proving Ground never fetches: {name} was refused"):
                call()
        for name, call in tool_calls.items():
            with pytest.raises(ToolLimitError, match=f"the Proving Ground never fetches: {name} is turned off"):
                call()
        report = clerk.refresh_data(["PG"])
        assert [(r["step"], r["outcome"]) for r in report] == [("all", "refused")] and "never fetches" in report[0]["detail"]


def test_the_financials_say_they_are_not_point_in_time_only_while_a_day_is_bound(monkeypatch):
    """Finnhub's metrics are today's snapshot (Jacob: keep and disclose). The call is the real `Financials.get`; only the provider's answer is replaced, by the shape Finnhub gives."""
    monkeypatch.setattr(DataToolbox, "_api_keys", lambda self, env=None: {"FINNHUB_API_KEY": "none", "POLYGON_API_KEY": "none"})
    monkeypatch.setattr(DataToolbox, "_get_json", staticmethod(lambda client, limiter, name, url, params, headers=None: {"metric": {"peTTM": 25.0}}))
    toolbox = DataToolbox(base_dir=REPO_ROOT)
    assert "point_in_time" not in Financials(toolbox).get("PG")
    with tc.as_of(D):
        assert Financials(toolbox).get("PG")["point_in_time"] is False


# ---------------------------------------------------------------- the ledger, the Advisor and the portfolio

REAL_LEDGER = REPO_ROOT / "data" / "ledger" / "recommendations.jsonl"


@pytest.fixture
def ledger(tmp_path):
    """A ledger of real record lines (YUM of 2026-11-06 and 2026-11-12, PG of 2026-11-12), copied from the real one."""
    if not REAL_LEDGER.exists():
        pytest.skip("the real ledger is not on this machine")
    wanted = {"3689dc86ea50", "2b50c581aa94", "6d8959fcf54e"}
    lines = [line for line in REAL_LEDGER.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line)["identity"]["record_id"] in wanted]
    assert len(lines) == 3, "the real ledger no longer holds the three lines this test copies"
    path = tmp_path / "recommendations.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Ledger(path)


def advisor_replay(name, ledger, today=None):
    capture = json.loads((ADVISOR_CAPTURES / f"{name}.json").read_text(encoding="utf-8"))
    order = wo.WorkOrder.model_validate(capture["order"])
    advisor = Advisor(model=Scripted(*capture["replies"]), clerk=RecordedClerk(capture["metrics"]), ledger=ledger)
    advisor.generate_recommendations(order, "funds", today)
    return order, capture


def scrubbed(record):
    record = copy.deepcopy(record)
    record["identity"]["record_id"] = record["identity"]["request_id"] = record["audit"]["metric_order"] = None
    return record


def test_the_advisor_asked_with_no_date_works_on_the_bound_day_and_its_cooldown_reads_no_later_line(ledger, tmp_path):
    """The real PG capture of 2026-11-05 (the Advisor's own recorded model replies and metrics), run with NO date inside `as_of(2026-11-05)`: the records are the ones the dated run makes, and the real PG line of
    2026-11-12 that the ledger holds (a conviction recommendation whose time stop is in January) does NOT put PG on cooldown: on 2026-11-05 it had not been written."""
    day = date(2026, 11, 5)
    dated, capture = advisor_replay("PG_open", Ledger(tmp_path / "empty.jsonl"), today=day)
    assert [r["symbol"] for r in dated.recommendations] == ["PG"] and capture["today"] == day.isoformat()
    with tc.as_of(day):
        bound, _ = advisor_replay("PG_open", ledger)
        assert bound.recommendations and bound.recommendations[0]["identity"]["as_of_utc"] == "2026-11-05T00:00:00Z"
        assert [scrubbed(r) for r in bound.recommendations] == [scrubbed(r) for r in dated.recommendations]
    later, _ = advisor_replay("PG_open", ledger, today=date(2026, 11, 20))                          # unbound, on a day after the line: the cooldown is what it always was
    assert later.recommendations == [] and "cooldown" in json.dumps(later.candidates)


# ---------------------------------------------------------------- the Critic: a real recommendation, a past day, no denial for timing

TIMING_CODES = {"market_closed", "record_expired", "time_stop_passed", "bars_stale", "no_live_price", "quote_stale", "price_left_zone", "stop_already_hit"}


class NoModel:
    def invoke(self, *args, **kwargs):
        raise AssertionError("the Data Clerk's model is never asked for a review")

    bind_tools = invoke


def test_the_critic_judges_a_real_recommendation_against_the_bound_day_and_not_against_the_real_one(real_index, monkeypatch, tmp_path):
    """The real MSFT recommendation of `auto_ai_quoted_theme.json` (made for 2026-11-13), reviewed by the real Critic with the real Data Clerk, the real stored bars, the real index and the Critic's recorded-style model
    (`ReviewerModel`, the one every Critic test uses). Bound to 2026-11-13 the clock is that day's regular session, the quote is that day's stored close, the record is for the day and nothing of timing denies it. The same review on
    the simulated clock of today (entry 76) is denied for timing (the record is a week old): that is what `as_of` removes. The Critic's other findings (the capture holds no sources, Finnhub is switched off here) are not the question."""
    need_bars()
    day = date(2026, 11, 13)
    asked = []

    def tripwire(self, symbol):
        asked.append(symbol)
        raise RuntimeError("no network in a test")

    monkeypatch.setattr(Financials, "get", tripwire)                                                # the one provider call a review makes (Finnhub's metrics) fails on purpose
    directory = json.loads((FIXTURES / "finnhub_company_search.json").read_text(encoding="utf-8"))["queries"]
    capture = json.loads((FIXTURES / "session_real" / "auto_ai_quoted_theme.json").read_text(encoding="utf-8"))

    def reviewed():
        order = wo.WorkOrder.model_validate(copy.deepcopy(capture["order"]))
        order.recommendations = [r for r in order.recommendations if r["symbol"] == "MSFT"]
        order.status, order.reviews, order.trade_decisions, order.trades, order.trader_runs, order.critic_runs = "recommended", [], [], [], [], []
        clerk = DataClerk(toolbox=DataToolbox(base_dir=REPO_ROOT), model=NoModel(), symbol_search=lambda query: directory[query])
        clerk.lookup_interval = 0
        critic = Critic(model=ReviewerModel(), clerk=clerk, ledger=Ledger(tmp_path / "reviews.jsonl"), history=Ledger(tmp_path / "history.jsonl"))
        return critic.review_recommendations(order, view())

    with tc.as_of(day):
        bound = reviewed()
    review, run = bound.reviews[0], bound.critic_runs[-1]
    assert run["now_et"] == "2026-11-13 15:30 ET" and run["session"] == "regular"
    assert review["timing"]["trading_date"] == "2026-11-13" and review["timing"]["record_date"] == "2026-11-13" and review["timing"]["price_basis"] == "quote"
    assert review["timing"]["quote_price"] == round(day_bar("MSFT", day)["close"], 4) and review["timing"]["quote_as_of_utc"] == "2026-11-13T20:29:00Z"
    assert not TIMING_CODES & {f["code"] for f in review["blockers"]} and not TIMING_CODES & set(review["denial_reasons"])
    monkeypatch.setenv("USE_LIVE_TRADING", "false")
    today = reviewed().reviews[0]
    assert {"record_expired", "bars_stale"} <= set(today["denial_reasons"]) and asked                # the clock of today: expired


# ---------------------------------------------------------------- the history: windows, and the fetchers that take one (no network: the provider's call is a recorder)

class Providers:
    """Stands where the providers' HTTP calls are and records what it is asked: the arguments the fetchers hand over are the windowing, and they are what is tested. It answers nothing (no article, no filing, no split)."""

    def __init__(self, monkeypatch, answer=None):
        self.asked, self.answer = [], answer
        monkeypatch.setattr(DataToolbox, "_api_keys", lambda toolbox, env=None: {"FINNHUB_API_KEY": "none", "POLYGON_API_KEY": "none"})
        monkeypatch.setattr(DataToolbox, "_get_json", staticmethod(self.get_json))
        for name in ("_iter_finnhub", "_iter_polygon"):
            monkeypatch.setattr(DataToolbox, name, lambda toolbox, client, key, symbol, start, end, _n=name: self.iterate(_n, symbol, start, end))
        monkeypatch.setattr(DataToolbox, "_iter_finnhub_general", lambda toolbox, client, key, category: self.iterate("general", category, None, None))

    def get_json(self, client, limiter, name, url, params, headers=None):
        self.asked.append((name, url, dict(params) if params else None))
        return self.answer(name, url, params) if self.answer else ({"results": []} if "filings" in name else [])

    def iterate(self, name, symbol, start, end):
        self.asked.append((name, symbol, start, end))
        return iter(())


def window_of(call):
    return call[2], call[3]


def test_a_window_is_whole_days_from_the_start_to_the_end_of_the_end_day_and_the_old_form_is_unchanged():
    now = datetime(2026, 11, 20, 17, 0, tzinfo=timezone.utc)
    assert fetch_window(30, None, None, now) == (now - timedelta(days=30), now)                                    # the form the fetchers have always had
    assert fetch_window(30, date(2026, 5, 20), date(2026, 6, 18), now) == (datetime(2026, 5, 20, tzinfo=timezone.utc), datetime(2026, 6, 18, 23, 59, 59, tzinfo=timezone.utc))
    assert fetch_window(30, date(2026, 11, 10), None, now) == (datetime(2026, 11, 10, tzinfo=timezone.utc), now)  # no end: up to now
    assert fetch_window(30, date(2026, 11, 10), date(2026, 12, 30), now)[1] == now                                 # an end in the future is cut to now
    for start, end in ((None, date(2026, 5, 1)), (date(2026, 6, 1), date(2026, 5, 1)), (date(2026, 12, 1), None), ("2026-05-01", None), (datetime(2026, 5, 1), None)):
        with pytest.raises(ValueError):
            fetch_window(30, start, end, now)


def test_news_is_fetched_in_the_window_it_is_given_and_a_window_keeps_only_what_was_published_in_it(monkeypatch, tmp_path, capsys):
    providers = Providers(monkeypatch)
    toolbox = DataToolbox(base_dir=tmp_path)
    toolbox.fetch_news(symbol="XOM", start=date(2026, 5, 20), end=date(2026, 6, 18))
    assert [window_of(c) for c in providers.asked if c[0] in ("_iter_finnhub", "_iter_polygon")] == [(datetime(2026, 5, 20, tzinfo=timezone.utc), datetime(2026, 6, 18, 23, 59, 59, tzinfo=timezone.utc))] * 2
    assert "window (2026-05-20 -> 2026-06-18)" in capsys.readouterr().out
    providers.asked.clear()
    toolbox.fetch_news(symbol="XOM", from_date=30)                                                                  # the old form: the last 30 days up to now
    start, end = window_of(providers.asked[0])
    assert timedelta(days=30) <= end - start <= timedelta(days=30, seconds=5) and abs((end - datetime.now(timezone.utc)).total_seconds()) < 5
    with pytest.raises(ValueError, match="end needs a start"):
        toolbox.fetch_news(symbol="XOM", end=date(2026, 6, 18))
    capsys.readouterr()


def test_market_news_in_a_past_window_asks_massive_only_and_leaves_the_coverage_file_alone(monkeypatch, tmp_path, capsys):
    providers = Providers(monkeypatch)
    toolbox = DataToolbox(base_dir=tmp_path)
    toolbox.sync_market_news(start=date(2026, 5, 20), end=date(2026, 6, 18))
    assert [c[0] for c in providers.asked] == ["_iter_polygon"] and window_of(providers.asked[0]) == (datetime(2026, 5, 20, tzinfo=timezone.utc), datetime(2026, 6, 18, 23, 59, 59, tzinfo=timezone.utc))
    assert not (tmp_path / "news" / "market" / "_coverage.json").exists()                                           # a past window is not part of the incremental range that file records
    providers.asked.clear()
    toolbox.sync_market_news(start=date.today() - timedelta(days=1))                                                # a window that reaches the last two days also asks the two general feeds
    assert [c[0] for c in providers.asked] == ["_iter_polygon", "general", "general"]
    providers.asked.clear()
    toolbox.sync_market_news(from_date=7)                                                                           # the old form: both feeds, and the coverage file
    assert [c[0] for c in providers.asked] == ["_iter_polygon", "general", "general"] and (tmp_path / "news" / "market" / "_coverage.json").exists()
    capsys.readouterr()


def test_the_split_table_is_written_one_file_a_ticker_page_by_page_and_an_empty_table_is_a_table(monkeypatch, tmp_path):
    """Massive's `GET /stocks/v1/splits` (docs: massive.com/docs/rest/stocks/corporate-actions/splits): `ticker`, `sort=execution_date.asc`, `limit`, the cursor in `next_url`. The rows are Apple's real splits."""
    pages = {"AAPL": [{"results": [{"ticker": "AAPL", "execution_date": "2020-08-31", "split_from": 1, "split_to": 4, "adjustment_type": "forward_split", "id": "E1"},
                                   {"ticker": "AAPL", "execution_date": "2005-02-28", "split_from": 1, "split_to": 2, "adjustment_type": "forward_split", "id": "E0"}],
                       "next_url": "https://api.massive.com/stocks/v1/splits?cursor=abc&ticker=AAPL"},
                      {"results": [{"ticker": "AAPL", "execution_date": "2014-06-09", "split_from": 1, "split_to": 7, "adjustment_type": "forward_split", "id": "E2"},
                                   {"ticker": "AAPL", "execution_date": "not a date", "split_from": 1, "split_to": 7}], "status": "OK"}],
             "PG": [{"results": [], "status": "OK"}]}

    def answer(name, url, params):
        return pages[(params or {}).get("ticker") or "AAPL"].pop(0)

    providers = Providers(monkeypatch, answer)
    toolbox = DataToolbox(base_dir=tmp_path)
    out = toolbox.fetch_splits(["AAPL", "PG"])
    stored = json.loads((tmp_path / "data" / "historical" / "splits_v1" / "AAPL.json").read_text(encoding="utf-8"))
    assert [s["execution_date"] for s in stored["splits"]] == ["2005-02-28", "2014-06-09", "2020-08-31"] and stored["symbol"] == "AAPL"
    assert (out["AAPL"]["written"], out["AAPL"]["discarded"], out["AAPL"]["first"], out["AAPL"]["last"]) == (3, 1, "2005-02-28", "2020-08-31")
    assert json.loads((tmp_path / "data" / "historical" / "splits_v1" / "PG.json").read_text(encoding="utf-8"))["splits"] == [] and out["PG"]["written"] == 0
    first, second = providers.asked[0], providers.asked[1]
    assert first[1] == "https://api.massive.com/stocks/v1/splits" and first[2]["ticker"] == "AAPL" and first[2]["sort"] == "execution_date.asc"
    assert second[1] == "https://api.massive.com/stocks/v1/splits?cursor=abc&ticker=AAPL&apiKey=none" and second[2] is None       # the cursor is kept, the key is added to it
    with pytest.raises(ValueError, match="ticker symbols"):
        toolbox.fetch_splits(["not a symbol"])


# ---------------------------------------------------------------- scripts/backfill_history.py

@pytest.fixture(scope="module")
def backfill():
    import importlib.util
    spec = importlib.util.spec_from_file_location("backfill_history", REPO_ROOT / "scripts" / "backfill_history.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_six_months_are_seven_windows_oldest_first_with_no_gap_and_no_overlap(backfill):
    spans = backfill.windows(date(2026, 5, 20), date(2026, 11, 20))
    assert spans[0] == (date(2026, 5, 20), date(2026, 6, 18)) and spans[-1] == (date(2026, 11, 16), date(2026, 11, 20)) and len(spans) == 7
    assert all(b[0] == a[1] + timedelta(days=1) for a, b in zip(spans, spans[1:])) and all((e - s).days + 1 <= 30 for s, e in spans)
    assert backfill.windows(date(2026, 5, 20), date(2026, 5, 20)) == [(date(2026, 5, 20), date(2026, 5, 20))]
    for bad in ((date(2026, 6, 1), date(2026, 5, 1), 30), (date(2026, 5, 1), date(2026, 6, 1), 0), (date(2026, 5, 1), date(2026, 6, 1), 31)):
        with pytest.raises(ValueError):
            backfill.windows(*bad)


