"""Tests for prometheus.langgraph.desk: the LangGraph that passes one work order from agent to agent (Part 9 of the RAG guide).

Run:  pytest tests/test_desk_graph.py -q

No network, no key, no cost. The graph is exercised with fake agents (to check the routing and that the SAME order object travels through every
node) and, in one test, with the real Data Clerk and Advisor driven by scripted models over a small temporary index.
"""
import json

import pytest

from prometheus.agents.advisor.advisor import Advisor
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk, Result
from prometheus.common import work_order as wo
from prometheus.langgraph import desk
from prometheus.tools import symbols
from test_advisor import Scripted, reply, rec, sig
from test_data_clerk_agent import ScriptedModel, call, say
from test_data_clerk_tools import FakeToolbox
from test_rag_retriever import DAY, NOW, KeywordEmbeddings, chunk, fill, make_store


# ---------------------------------------------------------------- the four parameters become a work order

def build(**request):
    return desk.order_from_request(request)


def test_a_symbol_alone_is_a_live_order_asking_for_the_latest_news_on_it():
    o = build(symbol="MSFT", from_days_ago=14)
    assert o.subject == ["MSFT"] and o.live is True and o.since_days == 14 and o.time_period == "last 14 days" and o.source == "news"
    assert o.query.startswith("What is the latest news on MSFT") and o.action_type == "search_news" and o.top_k == 10
    assert o.criteria == {"sentiment": "bullish or bearish"} and o.requester == "desk" and o.status == "new" and len(o.request_id) == 12


def test_a_term_alone_is_not_live_because_there_is_no_ticker_to_fetch():
    o = build(term="flying car", from_days_ago=7)
    assert o.subject == "flying car" and o.query == "flying car" and o.live is False and o.since_days == 7


def test_a_symbol_and_a_term_search_the_term_inside_that_symbols_news_and_still_fetch_the_whole_symbol():
    o = build(symbol="SNDK", term="long-term contracts")
    assert o.subject == ["SNDK"] and o.query == "long-term contracts news for SNDK" and o.live is True
    assert o.since_days == 7  # the default


def test_no_fetch_turns_a_symbol_order_into_a_search_of_the_stored_news():
    assert build(symbol="MSFT", fetch=False).live is False


def test_the_defaults_are_news_and_seven_days_and_blank_values_count_as_missing():
    o = build(symbol="MSFT", term="  ", source="", from_days_ago="")
    assert o.source == "news" and o.since_days == 7 and o.query.startswith("What is the latest news on MSFT")


@pytest.mark.parametrize("given, symbol", [("msft", "MSFT"), ("$MSFT", "MSFT"), ("  $aapl ", "AAPL"), ("BRK.B", "BRK.B")])
def test_a_symbol_is_cleaned_like_the_tools_do(given, symbol):
    assert build(symbol=given).subject == [symbol]


@pytest.mark.parametrize("request_, message", [
    ({}, "give a symbol, a term, or both"),
    ({"symbol": "", "term": "  "}, "give a symbol, a term, or both"),
    ({"symbol": "../etc"}, "not a valid ticker symbol"),
    ({"symbol": "MSFT", "from_days_ago": "soon"}, "from_days_ago must be a whole number of days"),
    ({"symbol": "MSFT", "from_days_ago": 7.5}, "from_days_ago must be a whole number of days"),
    ({"symbol": "MSFT", "from_days_ago": True}, "from_days_ago must be a whole number of days"),
    ({"term": "x" * 201}, "the term is longer than 200 characters"),
])
def test_a_request_that_cannot_become_an_order_is_refused_with_a_message_for_the_user(request_, message):
    with pytest.raises(ValueError, match=message):
        desk.order_from_request(request_)


def test_days_given_as_text_are_accepted_because_the_script_passes_text():
    assert build(symbol="MSFT", from_days_ago="14").since_days == 14


# ---------------------------------------------------------------- the graph

def test_the_graph_has_the_three_agents_and_the_routing_that_stops_early():
    g = desk.build_desk(clerk_factory=lambda order: None, advisor_factory=lambda order: None).get_graph()
    assert set(g.nodes) == {"__start__", "account_manager", "data_clerk", "advisor", "__end__"}
    edges = {(e.source, e.target, bool(e.conditional)) for e in g.edges}
    assert edges == {("__start__", "account_manager", False), ("account_manager", "data_clerk", True), ("account_manager", "__end__", True),
                     ("data_clerk", "advisor", True), ("data_clerk", "__end__", True), ("advisor", "__end__", False)}


class FakeClerk:
    """Records the order it is given and fills it as a Data Clerk that found one source would."""
    seen = None

    def __init__(self, order=None, fail=False):
        self.fail = fail

    def request_data(self, order):
        FakeClerk.seen = order
        result = Result(question=order.query, answer="Yes [1].", has_citations=True, references=[{"number": 1, "chunk_id": "a#0", "article_id": "a", "headline": "Microsoft is up on AI demand", "publisher": "Yahoo",
                                                                  "published_utc": "2026-10-23T10:00:00Z", "url": "https://x/a", "score": 0.61}],
                        sources=[{"number": 1, "chunk_id": "a#0"}], tool_calls=[{"name": wo.SEARCH_TOOL[order.source][0], "args": {}, "outcome": "ok"}],  # the search tool of the order's source
                        problems=["cites [9], which are not among the sources"] if self.fail else [])
        return wo.attach_result(order, result)


class FakeAdvisor:
    seen = None

    def advise(self, order):
        FakeAdvisor.seen = order
        order.recommendations = [{"symbol": "MSFT", "company": "Microsoft", "stance": "bullish", "model_score": 70, "score": 67, "supported": True,
                                  "signals": [], "signals_verified": 1, "signals_claimed": 1}]
        order.status = "recommended"
        return order


NAMES = {"procter&gamble": ("PG", "Procter & Gamble Co", "name"), "procter & gamble": ("PG", "Procter & Gamble Co", "name"),
         "sndk": ("SNDK", "Sandisk Corp", "ticker"), "msft": ("MSFT", "Microsoft Corp", "ticker"), "microsoft": ("MSFT", "Microsoft Corp", "name")}


class FakeResolver:
    """The name lookup, answering from a small table and recording what it was asked. It never reaches the network."""

    def __init__(self):
        self.asked = []

    def __call__(self, text):
        self.asked.append(text)
        key = text.strip().lstrip("$").lower()
        if key == "alphabet":
            raise symbols.SymbolLookupError("'alphabet' matches several companies: GOOGL (Alphabet Inc-Cl A), GOOG (Alphabet Inc-Cl C). Type the ticker symbol instead")
        if key not in NAMES:
            raise symbols.SymbolLookupError(f"no company found for {text!r}; type the ticker symbol instead")
        symbol, company, how = NAMES[key]
        return symbols.Resolution(typed=text.strip(), symbol=symbol, company=company, how=how)


def run(request, clerk=None, advisor=None, resolver=None, **options):
    FakeClerk.seen = FakeAdvisor.seen = None
    return desk.run_desk(request, clerk_factory=clerk or (lambda order: FakeClerk()), advisor_factory=advisor or (lambda order: FakeAdvisor()),
                         resolver=resolver or FakeResolver(), **options)


def test_one_work_order_travels_through_all_three_agents_and_is_the_same_object():
    state = run({"symbol": "MSFT", "from_days_ago": 14})
    order = state["order"]
    assert state["path"] == ["account_manager", "data_clerk", "advisor"] and state["errors"] == []
    assert FakeClerk.seen is order and FakeAdvisor.seen is order  # the SAME object, not a copy: the state of the graph is the work order
    assert order.status == "recommended" and order.data.answer == "Yes [1]." and order.recommendations[0]["score"] == 67
    assert set(state["seconds"]) == {"account_manager", "data_clerk", "advisor"} and all(isinstance(v, float) for v in state["seconds"].values())


def test_the_order_the_account_manager_builds_is_the_order_that_comes_out_not_a_copy(monkeypatch):
    built = []
    real = desk.order_from_request
    monkeypatch.setattr(desk, "order_from_request", lambda request: built.append(real(request)) or built[-1])
    state = run({"symbol": "MSFT"})
    assert len(built) == 1 and state["order"] is built[0] and FakeClerk.seen is built[0] and FakeAdvisor.seen is built[0]


def test_a_request_with_no_symbol_and_no_term_stops_at_the_account_manager():
    state = run({"symbol": "", "term": ""}, clerk=lambda order: pytest.fail("the Data Clerk must not be built"))
    assert state["path"] == ["account_manager"] and state["order"] is None and state["errors"] == ["give a symbol, a term, or both"]


def test_a_source_that_is_not_wired_stops_at_the_account_manager_with_the_reason_on_the_order():
    state = run({"symbol": "MSFT", "source": "social"}, clerk=lambda order: pytest.fail("the Data Clerk must not be built"))  # 'filings' was the example until entry 35 wired it
    assert state["path"] == ["account_manager"] and state["errors"] == ["source 'social' is not wired yet (only: news, press_releases, filings)"]
    assert state["order"].errors == state["errors"] and state["order"].status == "new"


def test_a_window_outside_one_to_ninety_days_is_refused():
    state = run({"symbol": "MSFT", "from_days_ago": 120}, clerk=lambda order: pytest.fail("the Data Clerk must not be built"))
    assert state["path"] == ["account_manager"] and any("since_days must be a whole number of days from 1 to 90" in e for e in state["errors"])


def test_data_that_is_not_ready_stops_before_the_advisor():
    state = run({"term": "flying car"}, clerk=lambda order: FakeClerk(fail=True), advisor=lambda order: pytest.fail("the Advisor must not be built"))
    assert state["path"] == ["account_manager", "data_clerk"] and state["order"].status == "data_failed" and state["order"].recommendations == []


def test_without_the_advisor_the_graph_stops_after_the_data_clerk():
    state = run({"term": "flying car"}, advisor=lambda order: pytest.fail("the Advisor must not be built"), with_advisor=False)
    assert state["path"] == ["account_manager", "data_clerk"] and state["order"].status == "data_ready"


def test_the_data_clerk_may_fetch_only_when_the_order_is_live(monkeypatch):
    made = []

    class Spy:
        def __init__(self, allow_fetch=False):
            made.append(allow_fetch)

        def request_data(self, order):
            return order

    monkeypatch.setattr(desk, "DataClerk", Spy)
    desk.run_desk({"symbol": "MSFT"}, with_advisor=False)
    desk.run_desk({"symbol": "MSFT", "fetch": False}, with_advisor=False)
    desk.run_desk({"term": "flying car"}, with_advisor=False)
    assert made == [True, False, False]  # a live symbol order may fetch; --no-fetch and a term order may not


def test_an_agent_that_cannot_start_is_reported_on_the_order_and_not_raised():
    def broken(order):
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env; see .env.example)")

    state = run({"symbol": "MSFT"}, clerk=broken, advisor=lambda order: pytest.fail("the Advisor must not be built"))
    assert state["path"] == ["account_manager", "data_clerk"] and state["order"].status == "data_failed"
    assert state["order"].errors == ["the Data Clerk could not start: OPENROUTER_API_KEY is not set (put it in .env; see .env.example)"]
    state = run({"symbol": "MSFT"}, advisor=broken)
    assert state["path"] == ["account_manager", "data_clerk", "advisor"] and state["order"].status == "data_ready"
    assert state["order"].errors == ["the Advisor could not start: OPENROUTER_API_KEY is not set (put it in .env; see .env.example)"]


# ---------------------------------------------------------------- the real agents, scripted

@pytest.fixture
def index(tmp_path):
    return fill(make_store(tmp_path / "chroma", KeywordEmbeddings()), [
        chunk("a1", 0, 1, "Apple has sold a million iphone phone units", ["AAPL"], NOW - 2 * DAY),
        chunk("t1", 0, 1, "tesla car", ["TSLA"], NOW - 1 * DAY)])


@pytest.fixture
def toolbox(tmp_path, monkeypatch):
    news = tmp_path / "news"
    news.mkdir()
    monkeypatch.setenv("NEWS_OUTPUT_DIR", str(news))
    return FakeToolbox(news)


def test_the_real_data_clerk_and_advisor_pass_one_order_through_the_graph(index, toolbox):
    """A term with no symbol (a stored search, nothing fetched): the real DataClerk and Advisor, scripted models, a small index."""
    clerk_model = ScriptedModel([call("retrieve_news", query="apple iphone phone", k=2), say("Apple has sold a million phones [1].")])
    advisor_model = Scripted(reply(rec("AAPL", "Apple", signals=[sig(1, "bullish", "Apple has sold a million iphone phone units")])))
    made = []

    def clerk(order):
        made.append(order.live)
        return DataClerk(toolbox=toolbox, store=index, model=clerk_model, allow_fetch=order.live)

    state = desk.run_desk({"term": "apple iphone phone", "from_days_ago": 30}, clerk_factory=clerk, advisor_factory=lambda order: Advisor(model=advisor_model))
    order = state["order"]
    assert state["path"] == ["account_manager", "data_clerk", "advisor"] and made == [False] and toolbox.calls == []  # nothing was fetched
    assert order.status == "recommended" and [r["chunk_id"] for r in order.data.references] == ["a1#0"]
    r = order.recommendations[0]
    assert (r["symbol"], r["score"], r["supported"]) == ("AAPL", 67, True) and r["signals"][0]["chunk_id"] == "a1#0"
    assert "last 30 days" in clerk_model.seen[0][1][1].content and "subject: apple iphone phone" in clerk_model.seen[0][1][1].content
    assert wo.WorkOrder.model_validate_json(order.model_dump_json()) == order  # still plain JSON at the end


def test_a_term_answered_without_any_search_is_caught_and_the_advisor_never_runs(index, toolbox):
    """The "flying car" failure: the Data Clerk answers a bare term from memory. The order is data_failed and the graph stops."""
    clerk_model = ScriptedModel([say("Flying cars are an emerging technology.")])
    state = desk.run_desk({"term": "flying car"}, clerk_factory=lambda order: DataClerk(toolbox=toolbox, store=index, model=clerk_model),
                          advisor_factory=lambda order: pytest.fail("the Advisor must not be built"))
    assert state["path"] == ["account_manager", "data_clerk"] and state["order"].status == "data_failed"
    assert any("no search of the stored news succeeded" in e for e in state["order"].errors)


# ---------------------------------------------------------------- a company name in the symbol field

def test_a_company_name_is_looked_up_and_the_order_is_for_its_ticker_with_the_company_on_it():
    resolver = FakeResolver()
    state = run({"symbol": "Procter&Gamble", "from_days_ago": 7}, resolver=resolver)
    order = state["order"]
    assert resolver.asked == ["Procter&Gamble"] and state["path"] == ["account_manager", "data_clerk", "advisor"]
    assert order.subject == ["PG"] and order.company == "Procter & Gamble Co" and order.live is True and order.since_days == 7
    assert state["resolution"] == {"typed": "Procter&Gamble", "symbol": "PG", "company": "Procter & Gamble Co", "how": "name"}
    assert order.query.startswith("What is the latest news on PG")  # the query is about the ticker the lookup found, never about the typed text


def test_a_ticker_in_capitals_is_used_as_it_is_with_no_lookup():
    resolver = FakeResolver()
    for typed in ("MSFT", "$AAPL", "BRK.B", "PG"):
        state = run({"symbol": typed}, resolver=resolver)
        assert state["resolution"] is None and state["order"].company is None
    assert resolver.asked == []


@pytest.mark.parametrize("typed", ["msft", "$sndk", "microsoft"])
def test_a_ticker_in_lower_case_and_a_name_are_looked_up(typed):
    resolver = FakeResolver()
    state = run({"symbol": typed}, resolver=resolver)
    assert resolver.asked == [typed] and state["order"].subject[0] in ("MSFT", "SNDK") and state["resolution"]["how"] in ("ticker", "name")


def test_a_name_that_cannot_be_resolved_stops_at_the_account_manager_and_builds_nothing():
    for typed, message in (("alphabet", "matches several companies: GOOGL"), ("Xyzzy Corp", "no company found for 'Xyzzy Corp'")):
        state = run({"symbol": typed}, clerk=lambda order: pytest.fail("the Data Clerk must not be built"))
        assert state["path"] == ["account_manager"] and state["order"] is None and state["resolution"] is None and message in state["errors"][0]


def test_a_company_name_typed_as_the_term_is_a_topic_and_is_not_looked_up():
    resolver = FakeResolver()
    state = run({"term": "Procter&Gamble"}, resolver=resolver)
    assert resolver.asked == [] and state["order"].live is False and state["order"].subject == "Procter&Gamble" and state["order"].company is None


def test_no_fetch_still_looks_the_name_up_and_searches_the_stored_news_for_its_ticker():
    state = run({"symbol": "Procter&Gamble", "fetch": False})
    assert state["order"].subject == ["PG"] and state["order"].live is False


def test_the_real_lookup_is_used_when_no_resolver_is_given_and_it_is_found_when_called(monkeypatch):
    monkeypatch.setattr(desk.symbols, "resolve_company", FakeResolver())
    state = desk.run_desk({"symbol": "microsoft"}, clerk_factory=lambda order: FakeClerk(), advisor_factory=lambda order: FakeAdvisor())
    assert state["order"].subject == ["MSFT"]


def test_the_data_clerk_is_told_the_company_name_next_to_the_ticker(index, toolbox):
    clerk_model = ScriptedModel([call("retrieve_news", query="Procter & Gamble news", tickers="PG"), say("The stored news has nothing on Procter & Gamble.")])
    state = run({"symbol": "Procter&Gamble"}, clerk=lambda order: DataClerk(toolbox=toolbox, store=index, model=clerk_model, allow_fetch=order.live), with_advisor=False)
    question = clerk_model.seen[0][1][1].content
    assert "subject: PG (Procter & Gamble Co)" in question and "call fetch_news once for each ticker in the subject first, with from_date=7" in question
    assert state["order"].status == "data_ready" and state["order"].data.no_supporting_text


# ---------------------------------------------------------------- the report and the command line

def test_the_report_shows_the_request_the_path_the_answer_the_sources_the_signals_and_the_timing():
    state = run({"symbol": "MSFT", "from_days_ago": 14})
    state["order"].recommendations[0]["signals"] = [{"source": 1, "direction": "bullish", "quote": "Microsoft is up", "verified": True, "counted": True, "reason": None},
                                                    {"source": 2, "direction": "bullish", "quote": "Microsoft is up", "verified": True, "counted": False, "reason": None},
                                                    {"source": 3, "direction": "bearish", "quote": "made up", "verified": False, "counted": False, "reason": "the quote is not in source 3"}]
    text = desk.report(state, 12.3)
    for part in ("REQUEST  symbol MSFT | term - | source news | from 14 days ago", "PATH     account_manager -> data_clerk -> advisor", "LIVE (fetching on)",
                 "window last 14 days", "TOOLS    retrieve_news(ok)", "DATA CLERK ANSWER\nYes [1].", "ADVISOR RECOMMENDATIONS", "calculated score 67",
                 "OK   [1] bullish", "SAME [2] bullish", "FAIL [3] bearish", "<- the quote is not in source 3", "TIMING (seconds)", "| total 12.3"):
        assert part in text, part
    assert "(stopped here)" not in text


def test_the_report_shows_what_a_company_name_was_looked_up_as():
    text = desk.report(run({"symbol": "Procter&Gamble", "from_days_ago": 7}), 1.0)
    assert "RESOLVED 'Procter&Gamble' -> PG (Procter & Gamble Co), looked up and matched as a name" in text
    assert text.index("RESOLVED") < text.index("PATH") and "subject ['PG'] (Procter & Gamble Co)" in text
    assert "RESOLVED" not in desk.report(run({"symbol": "MSFT"}), 1.0)  # a ticker in capitals is not looked up, so there is nothing to show


def test_the_report_of_a_refused_request_says_why_and_that_it_stopped():
    text = desk.report(run({"symbol": "", "term": ""}))
    assert "PATH     account_manager  (stopped here)" in text and "REFUSED  give a symbol, a term, or both" in text


def test_the_report_marks_where_the_graph_stopped_and_why():
    text = desk.report(run({"term": "flying car"}, clerk=lambda order: FakeClerk(fail=True)))
    assert "PATH     account_manager -> data_clerk  (stopped here)" in text and "status data_failed" in text and "ERRORS   cites [9]" in text


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """The command line with the graph's agents faked and its output folder in tmp_path. `seen` records what main passed to run_desk."""
    monkeypatch.setattr(desk, "OUT_DIR", tmp_path / "out" / "data" / "processed")
    real_run = desk.run_desk  # taken before the patch, or the fake would call itself
    seen = {}

    def fake(request, **options):
        seen.update(request=request, options=options)
        return real_run(request, clerk_factory=lambda order: FakeClerk(), advisor_factory=lambda order: FakeAdvisor(), resolver=FakeResolver(),
                        with_advisor=options.get("with_advisor", True))

    monkeypatch.setattr(desk, "run_desk", fake)
    return seen


def test_the_command_line_passes_the_four_parameters_to_the_graph_and_prints_the_report(cli, capsys, tmp_path):
    assert desk.main(["--symbol", "MSFT", "--term", "copilot", "--source", "news", "--from-days-ago", "14"]) == 0
    assert cli["request"] == {"symbol": "MSFT", "term": "copilot", "source": "news", "from_days_ago": "14", "fetch": True}
    out = capsys.readouterr().out
    assert "REQUEST  symbol MSFT | term copilot | source news | from 14 days ago" in out and "ADVISOR RECOMMENDATIONS" in out and "saved:" in out
    saved = json.loads((tmp_path / "out" / "data" / "processed" / "desk_MSFT_copilot.json").read_text())  # the symbol AND the term are in the name
    assert saved["path"] == ["account_manager", "data_clerk", "advisor"] and saved["order"]["status"] == "recommended"


@pytest.mark.parametrize("args, name", [(["--symbol", "MSFT"], "desk_MSFT.json"), (["--symbol", "$sndk", "--term", "long-term contracts"], "desk_SNDK_long_term_contracts.json"),
                                          (["--term", "flying car"], "desk_flying_car.json")])
def test_runs_with_a_symbol_a_term_or_both_are_saved_under_different_names_so_none_overwrites_another(cli, capsys, tmp_path, args, name):
    assert desk.main(args) == 0
    assert (tmp_path / "out" / "data" / "processed" / name).exists(), sorted(p.name for p in (tmp_path / "out" / "data" / "processed").iterdir())
    capsys.readouterr()


def test_a_company_name_on_the_command_line_is_passed_on_as_typed_and_the_order_is_saved_under_its_ticker(cli, capsys, tmp_path):
    assert desk.main(["--symbol", "Procter&Gamble", "--from-days-ago", "7"]) == 0
    assert cli["request"]["symbol"] == "Procter&Gamble"
    out = capsys.readouterr().out
    assert "RESOLVED 'Procter&Gamble' -> PG" in out
    saved = json.loads((tmp_path / "out" / "data" / "processed" / "desk_PG.json").read_text())
    assert saved["resolution"]["symbol"] == "PG" and saved["order"]["company"] == "Procter & Gamble Co" and saved["order"]["subject"] == ["PG"]


def test_the_defaults_of_the_command_line_are_news_and_seven_days(cli, capsys):
    desk.main(["--term", "flying car"])
    assert cli["request"] == {"symbol": "", "term": "flying car", "source": "news", "from_days_ago": "7", "fetch": True}
    capsys.readouterr()


def test_no_fetch_and_no_advisor_reach_the_graph(cli, capsys):
    assert desk.main(["--symbol", "MSFT", "--no-fetch", "--no-advisor"]) == 0  # data_ready is the goal when the Advisor is turned off
    assert cli["request"]["fetch"] is False and cli["options"] == {"with_advisor": False}
    capsys.readouterr()


def test_a_refused_request_exits_with_1_and_saves_nothing(cli, capsys, tmp_path):
    assert desk.main([]) == 1
    assert "REFUSED  give a symbol, a term, or both" in capsys.readouterr().out and not (tmp_path / "out").exists()


def test_a_refused_request_that_has_an_order_with_errors_is_not_saved_either(cli, capsys, tmp_path):
    assert desk.main(["--symbol", "MSFT", "--source", "social"]) == 1  # the order exists (with its errors), the request is still refused
    assert "REFUSED  source 'social' is not wired yet (only: news, press_releases, filings)" in capsys.readouterr().out and not (tmp_path / "out").exists()


def test_json_prints_the_finished_order(cli, capsys):
    assert desk.main(["--symbol", "MSFT", "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["path"] == ["account_manager", "data_clerk", "advisor"] and printed["order"]["recommendations"][0]["symbol"] == "MSFT"
    assert "resolution" in printed and printed["resolution"] is None  # MSFT was a ticker in capitals: no lookup


def test_show_graph_prints_a_mermaid_diagram_and_runs_nothing(monkeypatch, capsys):
    monkeypatch.setattr(desk, "run_desk", lambda *a, **k: pytest.fail("nothing may run"))
    assert desk.main(["--show-graph"]) == 0
    out = capsys.readouterr().out
    assert "graph TD" in out and all(name in out for name in ("account_manager", "data_clerk", "advisor"))


# ================================================================ press releases (design choices entry 34)
#
# The Account Manager passes a symbol and/or a company name on the work order, unchecked as a ticker: the Data Clerk looks them up (symbol first, then the name).

from test_symbols import META as META_ROWS, PG_NAME as PG_NAME_ROWS, PG_TICKER as PG_TICKER_ROWS, FakeSearch  # noqa: E402  (the REAL Finnhub rows)

PG_CO = "Procter & Gamble Co"


def press(**request):
    return desk.order_from_request({"source": "press_releases", **request})


def test_a_press_release_request_with_a_symbol_is_a_live_order_with_a_year_window_and_no_company():
    o = press(symbol="AAPL")
    assert o.source == "press_releases" and o.subject == ["AAPL"] and o.company is None and o.live is True and o.action_type == "search_news"
    assert o.since_days == 365 and o.time_period == "last 365 days" and o.top_k == desk.TOP_K
    assert o.query.startswith("What do AAPL's latest press releases say") and "bullish or bearish" in o.query


def test_a_press_release_request_with_only_a_company_name_has_no_subject_and_carries_the_name():
    o = press(company="Apple Inc")
    assert o.subject is None and o.company == "Apple Inc" and o.live is True and o.query.startswith("What do Apple Inc's latest press releases say")


def test_a_press_release_request_with_both_keeps_both_for_the_data_clerk_to_choose():
    o = press(symbol="AAPL", company="Apple Inc")
    assert o.subject == ["AAPL"] and o.company == "Apple Inc" and "AAPL's latest" in o.query  # the symbol is what the question names; the lookup decides later


def test_a_term_narrows_a_press_release_question_and_cannot_replace_the_company():
    assert press(symbol="AAPL", term="dividend").query == "dividend press releases for AAPL"
    with pytest.raises(ValueError, match="a press release request needs a symbol, a company name, or both"):
        press(term="dividend")


def test_fetch_off_makes_a_press_release_order_not_live():
    assert press(symbol="AAPL", fetch=False).live is False and press(company="Apple Inc", fetch=False).live is False


def test_the_window_defaults_to_a_year_for_press_releases_and_to_a_week_for_news():
    assert press(symbol="AAPL").since_days == 365 and press(symbol="AAPL", from_days_ago="").since_days == 365 and press(symbol="AAPL", from_days_ago=30).since_days == 30
    assert build(symbol="AAPL").since_days == 7 and build(symbol="AAPL", source="news").since_days == 7


def test_the_symbol_of_a_press_release_request_is_not_checked_as_a_ticker_here_so_the_company_name_can_still_take_over():
    o = press(symbol="Procter and Gamble!", company=PG_CO)  # a name typed into the symbol field: not a symbol, but the company name is good
    assert o.subject == ["Procter and Gamble!"] and o.company == PG_CO and wo.verify_request(o) == []
    with pytest.raises(ValueError, match="not a valid ticker symbol"):
        build(symbol="Procter and Gamble!")  # for NEWS the symbol is still checked, as before
    assert press(symbol="$aapl").subject == ["aapl"]  # the dollar sign and spaces go; the case is the lookup's business


@pytest.mark.parametrize("request_, message", [({"symbol": "", "company": ""}, "a press release request needs a symbol, a company name, or both"),
                                               ({"symbol": "x" * 61}, "the symbol is longer than 60 characters"),
                                               ({"company": "x" * 61}, "the company name is longer than 60 characters"),
                                               ({"symbol": "AAPL", "from_days_ago": "soon"}, "from_days_ago must be a whole number of days")])
def test_a_bad_press_release_request_is_refused_with_its_reason(request_, message):
    with pytest.raises(ValueError, match=message):
        press(**request_)


@pytest.mark.parametrize("days, ok", [(1, True), (365, True), (730, True), (731, False), (0, False)])
def test_the_account_manager_accepts_a_press_release_window_up_to_two_years(days, ok):
    state = run({"symbol": "AAPL", "source": "press_releases", "from_days_ago": days})
    assert (state["errors"] == []) is ok and (ok or any("from 1 to 730" in e for e in state["errors"]))


def test_a_press_release_order_travels_through_all_three_agents_like_a_news_order():
    state = run({"symbol": "AAPL", "source": "press_releases"})
    order = state["order"]
    assert state["path"] == ["account_manager", "data_clerk", "advisor"] and FakeClerk.seen is order and FakeAdvisor.seen is order
    assert order.source == "press_releases" and order.subject == ["AAPL"] and order.status == "recommended"


def test_the_account_manager_does_not_look_a_press_release_symbol_up_the_data_clerk_does():
    resolver = FakeResolver()
    state = run({"symbol": "microsoft", "source": "press_releases"}, resolver=resolver)  # for news this would be looked up and become MSFT
    assert resolver.asked == [] and state["resolution"] is None and state["order"].subject == ["microsoft"]
    assert FakeClerk.seen.subject == ["microsoft"]  # unchanged on its way to the Data Clerk
    news = FakeResolver()
    run({"symbol": "microsoft"}, resolver=news)
    assert news.asked == ["microsoft"]  # news is as it was


def real_clerk(search):
    """The REAL Data Clerk with a model that must never be asked: a refused order must not reach it."""
    model = ScriptedModel([])
    return lambda order: DataClerk(toolbox=FakeToolbox(), model=model, symbol_search=search)


@pytest.mark.parametrize("request_, message", [({"symbol": "ZZZZ"}, "Symbol: ZZZZ is not found"),
                                               ({"company": "Nonexistent Holdings"}, "Company Name not found."),
                                               ({"symbol": "ZZZZ", "company": "Nonexistent Holdings"}, "Company Name not found.")])
def test_a_symbol_or_company_that_is_not_found_stops_after_the_data_clerk_with_the_message_and_the_advisor_never_runs(request_, message):
    state = run({"source": "press_releases", **request_}, clerk=real_clerk(FakeSearch()), advisor=lambda order: pytest.fail("the Advisor must not be built"))
    order = state["order"]
    assert state["path"] == ["account_manager", "data_clerk"] and state["errors"] == []  # the request was fine; the lookup found nothing
    assert order.status == "data_failed" and order.errors == [message] and order.data is None
    text = desk.report(state)
    assert f"ERRORS   {message}" in text and "(stopped here)" in text and "RESOLVED" not in text


def test_the_symbol_wins_and_the_name_is_the_fallback_through_the_whole_desk_and_the_report_says_which():
    class ResolvingClerk(FakeClerk):
        def request_data(self, order):
            order.resolved = {"typed_symbol": "ZZZZ", "typed_company": PG_CO, "symbol": "PG", "company": PG_CO, "how": "name"}
            order.subject, order.company = ["PG"], PG_CO
            return super().request_data(order)

    state = run({"symbol": "ZZZZ", "company": PG_CO, "source": "press_releases"}, clerk=lambda order: ResolvingClerk())
    text = desk.report(state, 1.0)
    assert "REQUEST  symbol ZZZZ | term - | source press_releases | from 365 days ago | company Procter & Gamble Co" in text
    assert "RESOLVED press releases for PG (Procter & Gamble Co), found by its company name (typed: symbol ZZZZ, company Procter & Gamble Co)" in text
    assert text.index("ORDER") < text.index("QUERY") < text.index("RESOLVED") and "subject ['PG'] (Procter & Gamble Co)" in text  # the lookup's answer follows the question


def test_a_news_order_has_no_resolved_line_and_no_company_in_the_request_line():
    text = desk.report(run({"symbol": "MSFT"}), 1.0)
    assert "RESOLVED press releases" not in text and "| company" not in text


# ---------------------------------------------------------------- the command line

def test_the_command_line_takes_a_company_and_a_press_release_source_with_a_year_default(cli, capsys, tmp_path):
    assert desk.main(["--source", "press_releases", "--company", "Apple Inc"]) == 0
    assert cli["request"] == {"symbol": "", "term": "", "source": "press_releases", "from_days_ago": "365", "fetch": True, "company": "Apple Inc"}
    assert "| company Apple Inc" in capsys.readouterr().out
    assert (tmp_path / "out" / "data" / "processed" / "desk_press_releases.json").exists()  # nothing resolved the symbol in this fake run: named by the source alone


def test_a_run_for_press_releases_is_saved_apart_from_a_news_run_of_the_same_symbol(cli, capsys, tmp_path):
    assert desk.main(["--symbol", "AAPL"]) == 0 and desk.main(["--source", "press_releases", "--symbol", "AAPL", "--from-days-ago", "30"]) == 0
    assert cli["request"] == {"symbol": "AAPL", "term": "", "source": "press_releases", "from_days_ago": "30", "fetch": True}  # no company key when none was given
    capsys.readouterr()
    assert sorted(p.name for p in (tmp_path / "out" / "data" / "processed").iterdir()) == ["desk_AAPL.json", "desk_AAPL_press_releases.json"]


def test_a_press_release_order_is_saved_under_the_symbol_the_lookup_found(monkeypatch, capsys, tmp_path):
    class ResolvingClerk(FakeClerk):
        def request_data(self, order):
            order.resolved = {"typed_symbol": None, "typed_company": PG_CO, "symbol": "PG", "company": PG_CO, "how": "name"}
            return super().request_data(order)

    monkeypatch.setattr(desk, "OUT_DIR", tmp_path / "out" / "data" / "processed")
    real_run = desk.run_desk
    monkeypatch.setattr(desk, "run_desk", lambda request, **o: real_run(request, clerk_factory=lambda order: ResolvingClerk(), advisor_factory=lambda order: FakeAdvisor(),
                                                                         resolver=FakeResolver(), with_advisor=o.get("with_advisor", True)))
    assert desk.main(["--source", "press_releases", "--company", PG_CO]) == 0
    capsys.readouterr()
    assert (tmp_path / "out" / "data" / "processed" / "desk_PG_press_releases.json").exists()  # found by its name, saved under its ticker


def test_the_news_defaults_of_the_command_line_are_unchanged(cli, capsys):
    desk.main(["--symbol", "MSFT"])
    assert cli["request"] == {"symbol": "MSFT", "term": "", "source": "news", "from_days_ago": "7", "fetch": True}  # still no company key, still a week
    capsys.readouterr()


def test_a_company_name_on_the_command_line_is_not_looked_up_by_the_desk(cli, capsys):
    assert desk.main(["--source", "press_releases", "--company", "Procter & Gamble"]) == 0
    assert "RESOLVED 'Procter" not in capsys.readouterr().out  # the desk's own symbol-field lookup (entry 31) is for news only


# ================================================================ filings (design choices entry 35)
#
# The Account Manager passes filings on the work order exactly as it passes press releases: a symbol and/or a company name, unchecked as a ticker, and the Data Clerk looks
# them up. These are the press release tests again with the source changed.

def filings(**request):
    return desk.order_from_request({"source": "filings", **request})


def test_a_filings_request_with_a_symbol_is_a_live_order_with_a_year_window_and_no_company():
    o = filings(symbol="AAPL")
    assert o.source == "filings" and o.subject == ["AAPL"] and o.company is None and o.live is True and o.action_type == "search_filing"  # entry 37 (entry 35 had search_news)
    assert o.since_days == 365 and o.time_period == "last 365 days" and o.top_k == desk.TOP_K and o.requester == "desk" and o.status == "new"
    assert o.query.startswith("What do AAPL's latest SEC filings say") and "bullish or bearish" in o.query and o.criteria == {"sentiment": "bullish or bearish"}


def test_a_filings_request_with_only_a_company_name_has_no_subject_and_carries_the_name():
    o = filings(company="Apple Inc")
    assert o.subject is None and o.company == "Apple Inc" and o.live is True and o.query.startswith("What do Apple Inc's latest SEC filings say")


def test_a_filings_request_with_both_keeps_both_for_the_data_clerk_to_choose():
    o = filings(symbol="AAPL", company="Apple Inc")
    assert o.subject == ["AAPL"] and o.company == "Apple Inc" and "AAPL's latest" in o.query


def test_a_term_narrows_a_filings_question_and_cannot_replace_the_company():
    assert filings(symbol="AAPL", term="risk factors").query == "risk factors SEC filings for AAPL"
    with pytest.raises(ValueError, match="a filings request needs a symbol, a company name, or both"):
        filings(term="risk factors")


def test_the_press_release_wording_of_the_request_is_unchanged_by_filings():
    assert press(symbol="AAPL", term="dividend").query == "dividend press releases for AAPL" and press(symbol="AAPL").query.startswith("What do AAPL's latest press releases say")
    with pytest.raises(ValueError, match="a press release request needs a symbol, a company name, or both"):
        press(term="dividend")


def test_fetch_off_makes_a_filings_order_not_live():
    assert filings(symbol="AAPL", fetch=False).live is False and filings(company="Apple Inc", fetch=False).live is False


def test_the_window_defaults_to_a_year_for_filings_and_may_be_two_years():
    assert desk.DEFAULT_DAYS_BY_SOURCE == {"news": 7, "press_releases": 365, "filings": 365}
    assert filings(symbol="AAPL").since_days == 365 and filings(symbol="AAPL", from_days_ago="").since_days == 365 and filings(symbol="AAPL", from_days_ago=30).since_days == 30


def test_the_symbol_of_a_filings_request_is_not_checked_as_a_ticker_here_so_the_company_name_can_still_take_over():
    o = filings(symbol="Procter and Gamble!", company=PG_CO)
    assert o.subject == ["Procter and Gamble!"] and o.company == PG_CO and wo.verify_request(o) == []
    assert filings(symbol="$aapl").subject == ["aapl"]  # the dollar sign and spaces go; the case is the lookup's business


@pytest.mark.parametrize("request_, message", [({"symbol": "", "company": ""}, "a filings request needs a symbol, a company name, or both"),
                                               ({"symbol": "x" * 61}, "the symbol is longer than 60 characters"),
                                               ({"company": "x" * 61}, "the company name is longer than 60 characters"),
                                               ({"symbol": "AAPL", "from_days_ago": "soon"}, "from_days_ago must be a whole number of days")])
def test_a_bad_filings_request_is_refused_with_its_reason(request_, message):
    with pytest.raises(ValueError, match=message):
        filings(**request_)


@pytest.mark.parametrize("days, ok", [(1, True), (365, True), (730, True), (731, False), (0, False)])
def test_the_account_manager_accepts_a_filings_window_up_to_two_years(days, ok):
    state = run({"symbol": "AAPL", "source": "filings", "from_days_ago": days})
    assert (state["errors"] == []) is ok and (ok or any("from 1 to 730" in e for e in state["errors"]))


def test_a_filings_order_travels_through_all_three_agents_like_a_news_order():
    state = run({"symbol": "AAPL", "source": "filings"})
    order = state["order"]
    assert state["path"] == ["account_manager", "data_clerk", "advisor"] and FakeClerk.seen is order and FakeAdvisor.seen is order
    assert order.source == "filings" and order.subject == ["AAPL"] and order.status == "recommended"


def test_the_account_manager_does_not_look_a_filings_symbol_up_the_data_clerk_does():
    resolver = FakeResolver()
    state = run({"symbol": "microsoft", "source": "filings"}, resolver=resolver)  # for news this would be looked up and become MSFT
    assert resolver.asked == [] and state["resolution"] is None and state["order"].subject == ["microsoft"] and FakeClerk.seen.subject == ["microsoft"]


@pytest.mark.parametrize("request_, message", [({"symbol": "ZZZZ"}, "Symbol: ZZZZ is not found"),
                                               ({"company": "Nonexistent Holdings"}, "Company Name not found."),
                                               ({"symbol": "ZZZZ", "company": "Nonexistent Holdings"}, "Company Name not found.")])
def test_a_filings_symbol_or_company_that_is_not_found_stops_after_the_data_clerk_with_the_message_and_the_advisor_never_runs(request_, message):
    """The error goes back to the user: the report prints it, and the desk's exit code (tested below) is 1."""
    state = run({"source": "filings", **request_}, clerk=real_clerk(FakeSearch()), advisor=lambda order: pytest.fail("the Advisor must not be built"))
    order = state["order"]
    assert state["path"] == ["account_manager", "data_clerk"] and state["errors"] == []  # the request was fine; the lookup found nothing
    assert order.status == "data_failed" and order.errors == [message] and order.data is None
    text = desk.report(state)
    assert f"ERRORS   {message}" in text and "(stopped here)" in text and "RESOLVED" not in text


def test_the_report_of_a_filings_order_says_what_the_lookup_decided_in_the_words_of_filings():
    class ResolvingClerk(FakeClerk):
        def request_data(self, order):
            order.resolved = {"typed_symbol": "ZZZZ", "typed_company": PG_CO, "symbol": "PG", "company": PG_CO, "how": "name"}
            order.subject, order.company = ["PG"], PG_CO
            return super().request_data(order)

    text = desk.report(run({"symbol": "ZZZZ", "company": PG_CO, "source": "filings"}, clerk=lambda order: ResolvingClerk()), 1.0)
    assert "REQUEST  symbol ZZZZ | term - | source filings | from 365 days ago | company Procter & Gamble Co" in text
    assert "RESOLVED filings for PG (Procter & Gamble Co), found by its company name (typed: symbol ZZZZ, company Procter & Gamble Co)" in text
    press_text = desk.report(run({"symbol": "ZZZZ", "company": PG_CO, "source": "press_releases"}, clerk=lambda order: ResolvingClerk()), 1.0)
    assert "RESOLVED press releases for PG (Procter & Gamble Co), found by its company name" in press_text  # unchanged


def test_the_command_line_takes_a_filings_source_with_a_company_and_a_year_default(cli, capsys, tmp_path):
    assert desk.main(["--source", "filings", "--company", "Apple Inc"]) == 0
    assert cli["request"] == {"symbol": "", "term": "", "source": "filings", "from_days_ago": "365", "fetch": True, "company": "Apple Inc"}
    assert "| company Apple Inc" in capsys.readouterr().out
    assert (tmp_path / "out" / "data" / "processed" / "desk_filings.json").exists()


def test_a_run_for_filings_is_saved_apart_from_a_news_run_and_a_press_release_run_of_the_same_symbol(cli, capsys, tmp_path):
    assert desk.main(["--symbol", "AAPL"]) == 0 and desk.main(["--source", "press_releases", "--symbol", "AAPL"]) == 0
    assert desk.main(["--source", "filings", "--symbol", "AAPL", "--from-days-ago", "30"]) == 0
    assert cli["request"] == {"symbol": "AAPL", "term": "", "source": "filings", "from_days_ago": "30", "fetch": True}  # no company key when none was given
    capsys.readouterr()
    assert sorted(p.name for p in (tmp_path / "out" / "data" / "processed").iterdir()) == ["desk_AAPL.json", "desk_AAPL_filings.json", "desk_AAPL_press_releases.json"]


def test_a_filings_order_that_found_nothing_exits_with_1_and_prints_the_error_to_the_user(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(desk, "OUT_DIR", tmp_path / "out" / "data" / "processed")
    real_run = desk.run_desk
    monkeypatch.setattr(desk, "run_desk", lambda request, **o: real_run(request, clerk_factory=real_clerk(FakeSearch()), advisor_factory=lambda order: pytest.fail("no Advisor"),
                                                                         resolver=FakeResolver(), with_advisor=o.get("with_advisor", True)))
    assert desk.main(["--source", "filings", "--symbol", "ZZZZ"]) == 1
    out = capsys.readouterr().out
    assert "ERRORS   Symbol: ZZZZ is not found" in out and "(stopped here)" in out
    assert desk.main(["--source", "filings", "--symbol", "ZZZZ", "--company", "Nonexistent Holdings"]) == 1
    assert "ERRORS   Company Name not found." in capsys.readouterr().out


def test_the_help_names_filings_and_its_company_and_window(capsys):
    with pytest.raises(SystemExit):
        desk.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "news, press_releases, filings" in text and "press_releases and filings only" in text and "365 for press_releases and filings, 1 to 730" in text


def test_the_account_manager_gives_each_source_its_own_action_and_only_filings_have_search_filing():
    assert filings(symbol="AAPL").action_type == "search_filing" and filings(company="Apple Inc").action_type == "search_filing"
    assert press(symbol="AAPL").action_type == "search_news" and build(symbol="AAPL").action_type == "search_news" and build(term="flying car").action_type == "search_news"
    for order in (filings(symbol="AAPL"), press(symbol="AAPL"), build(symbol="AAPL")):
        assert wo.verify_request(order) == []  # each is a valid order for the Data Clerk


def test_a_filings_order_goes_through_the_desk_with_its_action_and_the_saved_order_keeps_it(cli, capsys, tmp_path):
    assert desk.main(["--source", "filings", "--symbol", "AAPL"]) == 0
    capsys.readouterr()
    saved = json.loads((tmp_path / "out" / "data" / "processed" / "desk_AAPL_filings.json").read_text(encoding="utf-8"))
    assert saved["order"]["action_type"] == "search_filing" and saved["order"]["source"] == "filings"


def test_the_report_shows_the_results_of_each_source_and_the_filing_evidence_the_data_clerk_put_on_the_order():
    from prometheus.agents.data_clerk.data_clerk_agent import Result
    sources = [{"number": 1, "chunk_id": "polygon:a#0", "headline": "h", "publisher": "Yahoo", "published_utc": "2026-10-01T00:00:00Z", "score": 0.7, "text": "t"},
               {"number": 2, "chunk_id": "pr:b#0", "headline": "h", "publisher": "Press release", "published_utc": "2026-10-02T00:00:00Z", "score": 0.6, "text": "t"},
               {"number": 3, "chunk_id": "sec:0000000001-25-000001:items#0", "headline": "h", "publisher": "SEC filing", "published_utc": "2026-10-03T00:00:00Z", "score": 0.5, "text": "t"}]
    result = Result(question="q", answer="A [1] [3].", sources=sources, references=[{"number": n, "chunk_id": sources[n - 1]["chunk_id"], "headline": "h", "publisher": sources[n - 1]["publisher"],
                                                                                    "published_utc": sources[n - 1]["published_utc"], "score": 0.5} for n in (1, 3)],
                    tool_calls=[{"name": "retrieve_filings", "args": {}, "outcome": "ok"}],
                    filing_evidence={"META": {"symbol": "META", "evidence": [{"id": "E1"}, {"id": "E2"}], "balance_sheet": None, "gaps": ["no balance sheet: ..."], "digest": "d"}})
    order = wo.attach_result(wo.new_work_order("q", source="filings", action_type="search_filing", subject=["META"]), result)
    text = desk.report({"request": {"symbol": "META", "source": "filings"}, "order": order, "errors": [], "resolution": None, "path": ["account_manager", "data_clerk"], "seconds": {}})
    assert "RESULTS  news: 1 shown, 1 cited | press releases: 1 shown, 0 cited | filings: 1 shown, 1 cited" in text
    assert "EVIDENCE filings of META: 2 passages, balance sheet not available, 1 gap(s) (scores nothing: the Advisor does)" in text
    assert text.index("DATA ") < text.index("RESULTS") < text.index("EVIDENCE") < text.index("DATA CLERK ANSWER")
    news_only = wo.attach_result(wo.new_work_order("q", subject=["MSFT"]), Result(question="q", answer="A [1].", sources=sources[:1], references=[{"number": 1, "chunk_id": "polygon:a#0", "headline": "h", "publisher": "Yahoo",
                                 "published_utc": "2026-10-01T00:00:00Z", "score": 0.7}], tool_calls=[{"name": "retrieve_news", "args": {}, "outcome": "ok"}]))
    plain = desk.report({"request": {"symbol": "MSFT"}, "order": news_only, "errors": [], "resolution": None, "path": ["account_manager", "data_clerk"], "seconds": {}})
    assert "RESULTS  news: 1 shown, 1 cited | press releases: 0 shown, 0 cited | filings: 0 shown, 0 cited" in plain and "EVIDENCE" not in plain  # no evidence line when the tool was not called


def test_the_report_says_who_searched_each_source_when_the_data_clerks_code_swept_them():
    from prometheus.agents.data_clerk.data_clerk_agent import Result
    sources = [{"number": 1, "chunk_id": "polygon:a#0", "headline": "h", "publisher": "Yahoo", "published_utc": "2026-10-01T00:00:00Z", "score": 0.7, "text": "t"},
               {"number": 2, "chunk_id": "pr:b#0", "headline": "h", "publisher": "Press release", "published_utc": "2026-10-02T00:00:00Z", "score": 0.6, "text": "t", "swept": True}]
    result = Result(question="q", answer="A [1].", sources=sources, references=[{"number": 1, "chunk_id": "polygon:a#0", "headline": "h", "publisher": "Yahoo", "published_utc": "2026-10-01T00:00:00Z", "score": 0.7}],
                    tool_calls=[{"name": "retrieve_news", "args": {}, "outcome": "ok"}],
                    sweep={"news": {"by": "model", "sources": 1, "note": ""}, "press_releases": {"by": "code", "sources": 1, "note": ""},
                           "filings": {"by": "code", "sources": 0, "note": "No supporting text: no stored filing scored 0.40 or more for this question (tickers META); the closest scored 0.31."},
                           "filing_evidence": {"by": "code", "sources": 0, "note": "no filings of META are stored"}})
    order = wo.attach_result(wo.new_work_order("q", subject=["META"]), result)
    text = desk.report({"request": {"symbol": "META"}, "order": order, "errors": [], "resolution": None, "path": ["account_manager", "data_clerk"], "seconds": {}})
    assert "RESULTS  news: 1 shown, 1 cited | press releases: 0 shown, 0 cited, 1 swept | filings: 0 shown, 0 cited" in text
    assert "SWEEP    news: model, 1 | press releases: code, 1 | filings: code, 0 (No supporting text: no stored filing scored 0.40 or more for) | filing evidence: code, 0 (no filings of META are stored)" in text
    assert text.index("RESULTS") < text.index("SWEEP") < text.index("DATA CLERK ANSWER")
