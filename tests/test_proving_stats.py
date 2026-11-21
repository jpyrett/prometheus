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

from prometheus.common.portfolio import trade_amount
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


def ratio(part, whole) -> float | None:
    return six(CTX.divide(Decimal(part), Decimal(whole))) if whole else None


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


def test_every_statistic_of_a_real_set_of_outcomes_is_what_a_hand_count_of_them_says_with_its_n_and_the_small_sample_mark(scored, curve):
    config = load_config()
    result = stats.compute(scored, curve, config, starting_cash=Decimal("40000.00"))
    won, lost = [o for o in scored if o.result == "won"], [o for o in scored if o.result == "lost"]
    opened, closed = [o for o in scored if o.result == "open"], [o for o in scored if o.result in ("won", "lost")]
    assert len(scored) == 28 and (len(won), len(lost), len(opened)) == (5, 17, 6) and sum(o.traded for o in scored) == 12 and sum(o.hypothetical for o in scored) == 16

    block = result.outcomes                                                                     # the outcomes block: every outcome, an open position never a win
    assert (block.n, block.closed, block.open, block.won, block.lost, block.not_filled, block.no_data) == (28, 22, 6, 5, 17, 0, 0)
    assert block.win_rate == ratio(5, 22) and block.avg_win_pct == mean(o.return_pct for o in won) and block.avg_loss_pct == mean(o.return_pct for o in lost)
    assert block.profit_factor == six(CTX.divide(sum((Decimal(str(o.return_pct)) for o in won), Decimal(0)), -sum((Decimal(str(o.return_pct)) for o in lost), Decimal(0))))
    assert block.avg_sessions_held == mean(o.exit.sessions_held for o in closed)
    assert (block.exits.stop, block.exits.target, block.exits.time_stop, block.exits.open) == (sum(o.exit.reason == "stop" for o in closed), sum(o.exit.reason == "target" for o in closed), 0, 6)
    assert block.small_sample is False and block.closed >= config.min_sample                                     # 22 closed: over the minimum of 20

    usable = [o for o in scored if o.return_pct is not None and o.excess_return_pct is not None]
    for row in result.by_verdict:                                                                  # the Critic's value: approved, denied, neutral, unreviewed
        group = [o for o in usable if o.verdict == row.verdict]
        done = [o for o in group if o.result in ("won", "lost")]
        assert [r.verdict for r in result.by_verdict] == ["approved", "denied", "neutral", "unreviewed"]
        assert (row.n, row.closed, row.mean_return_pct, row.mean_excess_pct, row.win_rate, row.small_sample) == (len(group), len(done), mean(o.return_pct for o in group), mean(o.excess_return_pct for o in group),
                                                                                                                 ratio(sum(o.result == "won" for o in done), len(done)), len(group) < 20)
    assert [(r.verdict, r.n) for r in result.by_verdict] == [("approved", 12), ("denied", 12), ("neutral", 0), ("unreviewed", 4)]
    assert [(r.low, r.high) for r in result.by_score] == [(0, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 101)]
    for row in result.by_score:                                                                    # the Advisor's score in the buckets of the config: [low, high)
        group = [o for o in usable if o.advisor.score is not None and row.low <= o.advisor.score < row.high]
        done = [o for o in group if o.result in ("won", "lost")]
        assert (row.n, row.closed, row.mean_return_pct, row.mean_excess_pct, row.win_rate, row.small_sample) == (len(group), len(done), mean(o.return_pct for o in group), mean(o.excess_return_pct for o in group),
                                                                                                                 ratio(sum(o.result == "won" for o in done), len(done)), len(group) < 20)
    assert sum(r.n for r in result.by_score) == len(usable) == 28
    assert [r.family for r in result.by_family] == sorted({o.advisor.family for o in scored})
    for row in result.by_family:
        group = [o for o in usable if o.advisor.family == row.family]
        assert (row.n, row.mean_excess_pct) == (len(group), mean(o.excess_return_pct for o in group))
    traded = [o for o in scored if o.traded]
    assert [r.symbol for r in result.by_symbol] == sorted({o.symbol for o in traded})
    for row in result.by_symbol:                                                                   # the desk's money: gain in Money, over what was paid
        mine = [o for o in traded if o.symbol == row.symbol]
        gain, paid = sum((o.gain for o in mine), Decimal("0.00")), sum((o.entry.amount for o in mine), Decimal("0.00"))
        assert (row.n, row.gain, row.return_pct) == (len(mine), gain, six(CTX.divide(gain, paid))) and isinstance(row.gain, Decimal)

    folio = result.portfolio                                                                       # the account: from the curve alone
    equity = [p.equity for p in curve]
    assert (folio.start.isoformat(), folio.end.isoformat(), folio.days, folio.start_equity, folio.end_equity) == ("2026-11-13", "2026-11-19", 5, Decimal("40000.00"), equity[-1])
    assert folio.return_pct == six(CTX.divide(equity[-1], Decimal("40000.00")) - 1) and folio.benchmark_return_pct == six(CTX.divide(curve[-1].benchmark, Decimal("40000.00")) - 1)
    assert folio.excess_return_pct == six(Decimal(str(folio.return_pct)) - Decimal(str(folio.benchmark_return_pct)))
    peak, worst, day_returns, base = Decimal("40000.00"), Decimal(0), [], Decimal("40000.00")
    for p in curve:
        peak = max(peak, p.equity)
        worst = min(worst, CTX.divide(p.equity, peak) - 1)
        day_returns.append(six(CTX.divide(p.equity, base) - 1))
        base = p.equity
    assert folio.max_drawdown_pct == six(worst) and (folio.best_day_pct, folio.worst_day_pct) == (max(day_returns), min(day_returns))
    assert folio.drawdown_from.isoformat() == "2026-11-13" and folio.drawdown_to.isoformat() == [p.date for p in curve if CTX.divide(p.equity, Decimal("40000.00")) - 1 == worst][0].isoformat()
    assert folio.invested_avg_pct == six(CTX.divide(sum((CTX.divide(p.invested, p.equity) for p in curve), Decimal(0)), Decimal(5)))

    # the sentences, written out by hand from the templates' words
    approved, denied = result.by_verdict[0], result.by_verdict[1]
    assert (result.critic_value.denied_n, result.critic_value.approved_n) == (12, 12) and approved.small_sample and denied.small_sample          # both groups are under the minimum
    assert result.critic_value.sentence == "Small sample: the Critic denied 12 and approved 12, too few to say whether it was right (20 of each are needed)."
    assert result.critic_value.denied_mean_excess_pct == denied.mean_excess_pct and result.critic_value.approved_mean_excess_pct == approved.mean_excess_pct
    excess = abs(folio.excess_return_pct)
    word = "Beat" if folio.excess_return_pct > 0 else "Trailed"
    assert result.headline.sentence == f"{word} SPY by {format((Decimal(str(excess)) * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP), 'f')} points over 6 days: 28 recommendations, {WON_SHARE}% of the closed ones won."
    assert result.headline.tone == ("pos" if folio.excess_return_pct > 0 else "neg")
    assert result.headline.sentence == "Beat SPY by 1.4 points over 6 days: 28 recommendations, 23% of the closed ones won." and (folio.return_pct, folio.benchmark_return_pct) == (-0.030737, -0.045074)       # the desk lost 3.1% and SPY 4.5%


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
