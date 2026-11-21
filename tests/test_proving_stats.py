"""`stats.compute` (design/proving_ground_spec.md sections 8.3 and 14; design choices entry 126; P9 and P12).

REAL DATA ONLY, and no saved Time Machine run exists yet. The outcomes are the 28 real recommendations of the real captures and the real ledger scored by `outcomes.score` against the stored bars (the ones the desk really bought, in the captures,
with their real fills: traded; the others hypothetical, P8; each with its real Critic verdict), and the equity curve is the real Trader's fills of `auto_even_split_dollars` (five orders in dollars of $6,000, 2026-11-16) walked through `SimPortfolio` day
by day to 2026-11-19, with SPY bought for the same $40,000 at the open of the first fill day. Every statistic is then recomputed HERE, independently, from the outcomes' own returns and the curve's own equity: counts, sums and means in exact Decimal at
60 digits, rounded half up to six places, and the sentences written out by hand from the templates' words. Nothing is compared with `pytest.approx`.
"""
from datetime import date
from decimal import ROUND_HALF_UP, Context, Decimal

import pytest
from proving_helpers import plans_of, real_bars, real_records, replayed, stored_bars_available

from prometheus.proving import record, outcomes, stats
from prometheus.proving.config import load_config
from prometheus.proving.portfolio import SimPortfolio

pytestmark = pytest.mark.skipif(not stored_bars_available(), reason="the stored daily bars (data/historical/day_aggs_v1) are not on this machine")

CTX = Context(prec=60)
WON_SHARE = int((Decimal(5) / Decimal(22) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))      # 5 of the 22 closed ones won: 23%
DAYS = ["2026-11-13", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19"]


def six(x: Decimal) -> float:
    return float(x.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def mean(values) -> float | None:
    values = [Decimal(str(v)) for v in values]
    return six(CTX.divide(sum(values, Decimal(0)), Decimal(len(values)))) if values else None


@pytest.fixture(scope="module")
def scored():
    """The 28 real recommendations scored: traded with their real fills where the capture bought them, hypothetical otherwise."""
    reals = real_records()
    syms = {r.rec["symbol"] for r in reals} | {r.rec["identity"].get("sector_etf") for r in reals if r.rec["identity"].get("sector_etf")}
    bars, found = real_bars(syms), []
    for r in reals:
        who = record.identity_of(r.order, r.rec, r.day)
        if r.fill is None and who["verdict"] == "denied":
            who["not_traded_reason"] = "denied"
        found.append(outcomes.score(record.plan_of(r.rec), bars[r.rec["symbol"]], bars["SPY"], bars.get(r.rec["identity"].get("sector_etf")), r.day, [5, 10, 20], symbol=r.rec["symbol"], identity=who, fill=r.fill))
    return found


@pytest.fixture(scope="module")
def curve():
    """The equity curve of the real fills of `auto_even_split_dollars`, with the benchmark of the same $40,000 in SPY from the open of the first fill day (fractional, marked at each close, to the cent)."""
    order = replayed("auto_even_split_dollars", "40000.00")
    bars = real_bars({t["symbol"] for t in order.trades})
    account, info, points = SimPortfolio(Decimal("40000.00"), user="Bob Jones"), plans_of(order), []
    spy_open = Decimal(str(next(b for b in bars["SPY"] if b["date"] == "2026-11-16")["open"]))
    for day in DAYS:
        account.apply_fills([t for t in order.trades if t["fill"]["date"] == day], info)
        account.step(date.fromisoformat(day), bars)
        point = account.mark(date.fromisoformat(day), bars)
        held = Decimal("40000.00") if day < "2026-11-16" else (Decimal("40000.00") * Decimal(str(next(b for b in bars["SPY"] if b["date"] == day)["close"])) / spy_open).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        points.append(point.model_copy(update={"benchmark": held}))
    return points


def test_under_the_minimum_every_rate_carries_the_small_sample_mark_and_no_tone_and_without_a_curve_the_headline_is_about_the_recommendations(scored):
    """With a minimum of 30 (the real configuration, one number changed) the 22 closed outcomes are a small sample: the headline says so and has no colour of success or failure; the groups all carry the mark. With no equity curve (the Track Record has
    none) there is no portfolio and the headline is the average excess of the recommendations over the benchmark, in the closed set of words; an empty run says it has nothing."""
    config = load_config().model_copy(update={"min_sample": 30})
    small = stats.compute(scored, [], config)
    assert small.portfolio is None and small.outcomes.small_sample and all(g.small_sample for g in [*small.by_verdict, *small.by_score, *small.by_family])
    usable = [o for o in scored if o.excess_return_pct is not None]
    average = mean(o.excess_return_pct for o in usable)
    assert average == 0.005613 and small.headline.tone == "neutral"
    assert small.headline.sentence == f"Small sample: 22 closed so far, 30 needed. On average the recommendations were {format((Decimal(str(average)) * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP), '+f')} points against SPY; 28 recommendations are too few to call it."
    plain = stats.compute(scored, [], load_config())
    assert plain.headline.tone == "pos" and plain.headline.sentence == f"On average the recommendations beat SPY by {format((Decimal(str(average)) * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP), 'f')} points: 28 recommendations, {WON_SHARE}% of the closed ones won." == \
        "On average the recommendations beat SPY by 0.6 points: 28 recommendations, 23% of the closed ones won."
    nothing = stats.compute([], [], load_config())
    assert nothing.headline.sentence == "No recommendation has an outcome yet." and nothing.headline.tone == "neutral" and nothing.critic_value.sentence == "The Critic has no outcomes to judge yet." and nothing.outcomes.n == 0
    assert nothing.outcomes.win_rate is None and nothing.outcomes.profit_factor is None and nothing.by_symbol == []


def test_a_result_is_saved_and_read_back_exactly_as_one_file_under_the_store_and_never_under_the_stored_market_data(scored, curve, tmp_path):
    """`ProvingRun` through `ProvingStore`: written atomically by the one encoder (money as JSON numbers), read back to the same run, listed in the index with its headline, deleted; a cache entry and a session are their own files; a root inside the stored
    market data, the vector store or the real ledgers is refused, and so is a name that could leave its folder."""
    from prometheus.proving.record import ProvingRun, RunRange
    from prometheus.proving.store import ProvingStore
    result = stats.compute(scored, curve, load_config(), starting_cash=Decimal("40000.00"))
    run = ProvingRun(run_id="tm-test-0001", kind="time_machine", status="finished", created_utc="2026-11-20T18:00:00Z", request={"prompt": "I want to buy oil stocks.", "cash": Decimal("40000.00")}, range=RunRange(first_day=date(2026, 11, 13), last_day=date(2026, 11, 19),
                     last_bar_day=date(2026, 11, 19)), equity=curve, outcomes=scored, stats=result, caveats=["small_sample", "open_positions_marked"])
    store = ProvingStore(tmp_path / "proving")
    assert not (tmp_path / "proving").exists()
    path = store.save_run(run)
    assert path == (tmp_path / "proving" / "runs" / "tm-test-0001.json").resolve() and store.load_run("tm-test-0001") == run
    text = path.read_text(encoding="utf-8")
    assert '"start_equity": 40000,' in text and '"cash": 40000,' in text and '"40000.00"' not in text and "Decimal" not in text           # a number (a whole one an integer), never a string
    [row] = store.runs()
    assert row["run_id"] == "tm-test-0001" and row["prompt"] == "I want to buy oil stocks." and row["headline"] == result.headline.sentence and row["return_pct"] == result.portfolio.return_pct
    assert store.cache_get("k1") is None and store.cache_put("k1", {"status": "reviewed", "score": 73.0}).is_file() and store.cache_get("k1") == {"status": "reviewed", "score": 73.0}
    store.save_session("req-1", {"order": {"request_id": "req-1"}})
    assert store.sessions() == ["req-1"] and store.load_session("req-1") == {"order": {"request_id": "req-1"}} and store.load_session("nope") is None
    assert store.delete_run("tm-test-0001") is True and store.runs() == [] and store.delete_run("tm-test-0001") is False and store.cache_get("k1") is not None and store.sessions() == ["req-1"]
    assert sorted(p.name for p in (tmp_path / "proving").iterdir()) == ["cache", "runs", "sessions"] and not list((tmp_path / "proving").rglob("*.tmp"))
    from prometheus.rag.config import REPO_ROOT
    for bad in (REPO_ROOT / "data" / "historical" / "x", REPO_ROOT / "data" / "processed", REPO_ROOT / "data" / "ledger", REPO_ROOT / "news"):
        with pytest.raises(ValueError, match="never writes under"):
            ProvingStore(bad)
    for name in ("../x", "a/b", "", ".hidden", "index"):
        with pytest.raises(ValueError):
            store.cache_put(name, {}) if name != "index" else store.load_run(name)
    assert ProvingStore().root == load_config().store_root.resolve() and str(ProvingStore().root).endswith("data/proving")
