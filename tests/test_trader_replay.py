"""The replay command (prometheus/agents/trader/replay.py, Trader spec section 16): a saved work order through the Trader, free in backtest mode with no model. A fake Data Clerk (or the real one over day files in a
temporary folder) gives the bars; nothing here calls a model, a quote or a paid API, and nothing is written outside the temporary folder."""
import json
from datetime import date

import pytest
from bar_helpers import write_day_files
from test_trader import BARS, PLACE_ALL
from trader_helpers import MarketClerk, ScriptedModel, account, answer, decided, plan, row, say, synthetic

from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.agents.trader import replay
from prometheus.agents.trader.trader import Trader


def save(tmp_path, order, name="order.json", desk=True, funds=None):
    path = tmp_path / name
    body = {"request": {"symbol": "PG"}, "funds": funds, "path": ["account_manager"], "order": order.model_dump(mode="json")} if desk else order.model_dump(mode="json")
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def run(tmp_path, *flags, order=None, clerk=None, desk=True, funds=None, **kw):
    path = save(tmp_path, order or decided(), desk=desk, funds=funds)
    lines = []
    code = replay.main(["--order", str(path), *flags], clerk=clerk or MarketClerk(bars=BARS, quotes={"PG": 167.0}), out=lines.append, **kw)
    return code, "\n".join(lines)


# ---------------------------------------------------------------- the free command

def test_the_free_command_replays_a_saved_order_against_the_bars_with_no_model_and_prints_the_trader_section(tmp_path):
    code, text = run(tmp_path, "--mode", "backtest", "--no-model", "--cash", "10000")
    assert code == 0
    assert text.splitlines()[0].startswith("TRADER  (no model: the program placed the orders; BACKTEST mode, prices: the stored daily bars")
    assert "FILLED 44 at 166.0830 = $7,307.65   placed by code" in text and "closed by target on 2026-11-09" in text and "account  cash $10,000.00 -> $2,692.35" in text


def test_the_default_mode_is_backtest_and_a_bare_order_is_read_as_well_as_the_desks_saved_json(tmp_path):
    code, text = run(tmp_path, "--no-model", "--cash", "10000", desk=False)
    assert code == 0 and "BACKTEST mode" in text and "FILLED 44" in text


def test_the_moment_is_the_latest_decided_time_of_the_permitted_decisions_unless_now_is_given(tmp_path):
    order = decided()
    assert replay.default_now(order).isoformat() == "2026-11-05T15:30:00+00:00"
    order.trade_decisions.append({**order.trade_decisions[0], "record_id": "x", "decided_utc": "2026-11-05T15:45:00Z", "will_trade": "yes_auto"})
    order.trade_decisions.append({**order.trade_decisions[0], "record_id": "y", "decided_utc": "2026-11-06T09:00:00Z", "will_trade": "no_system_denied"})        # a denial is not a moment of permission
    assert replay.default_now(order).isoformat() == "2026-11-05T15:45:00+00:00"
    code, text = run(tmp_path, "--no-model", "--cash", "10000", "--now", "2026-11-05 08:00")
    assert code == 0 and "FILLED 44 at 165.0825" in text and "run 2026-11-05 08:00 ET" in text                                                                      # before the open: that Thursday's bar
    code, text = run(tmp_path, "--no-model", "--cash", "10000")
    assert "FILLED 44 at 166.0830" in text and "run 2026-11-05 10:30 ET" in text


def test_a_decision_time_after_the_approval_ran_out_is_a_rejection_not_a_trade(tmp_path):
    code, text = run(tmp_path, "--no-model", "--cash", "10000", "--now", "2026-11-05 11:45")
    assert code == 0 and "REJECTED approval_expired" in text and "FILLED" not in text


def test_the_portfolio_the_cash_and_the_profile_come_from_the_flags(tmp_path):
    folio = tmp_path / "portfolio.json"
    folio.write_text(account(9000.0, MSFT=(5, 300.0)).model_dump_json(), encoding="utf-8")
    code, text = run(tmp_path, "--no-model", "--portfolio", str(folio))
    assert code == 0 and "cash $9,000.00 -> $1,692.35; MSFT" not in text and "account  cash $9,000.00 -> $1,692.35; PG 0 -> 44 shares" in text
    code, text = run(tmp_path, "--no-model", "--cash", "5000")
    assert "cash $5,000.00 ->" in text and "FILLED 43 at 166.0830" in text or "FILLED 30 at 166.0830" in text                                                      # a purchase the cash cannot cover is cut
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"user": "Somebody Else", "risk_tolerance": "moderate"}), encoding="utf-8")
    code, text = run(tmp_path, "--no-model", "--portfolio", str(folio), "--profile", str(profile))
    assert code == 1 and "the user profile is for 'Somebody Else' but the portfolio belongs to 'Jacob Pyrett'" in text


def test_the_default_portfolio_is_jacobs_sample_which_cannot_pay_for_44_shares():
    lines = []
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as folder:
        path = save(Path(folder), decided())
        code = replay.main(["--order", str(path), "--no-model"], clerk=MarketClerk(bars=BARS), out=lines.append)
    text = "\n".join(lines)
    assert code == 0 and "REJECTED insufficient_cash" in text and "--raise-funds" in text and "cash $51.23 -> $51.23" in text


# ---------------------------------------------------------------- funding in a replay

def test_the_funding_question_is_answered_no_unless_yes_funding_is_given(tmp_path):
    order = synthetic(row("PG", "open", 20, 100.0))
    funds = plan("PG", ("MSFT", 5, 420.0), ("TSLA", 2, 330.0))
    folio = tmp_path / "p.json"
    folio.write_text(account(51.23, MSFT=(50, 150.0), TSLA=(20, 300.0)).model_dump_json(), encoding="utf-8")
    clerk = MarketClerk(bars={s: [{"date": d, "open": p, "high": p + 1, "low": p - 1, "close": p, "volume": 1} for d in ("2026-11-05", "2026-11-06", "2026-11-09")] for s, p in
                               {"PG": 100.0, "MSFT": 420.0, "TSLA": 330.0, "SPY": 400.0}.items()})
    code, text = run(tmp_path, "--no-model", "--portfolio", str(folio), order=order, funds=funds, clerk=clerk)
    assert "funding sells were not approved: nobody answered" in text and "REJECTED insufficient_cash" in text
    code, text = run(tmp_path, "--no-model", "--portfolio", str(folio), "--yes-funding", order=order, funds=funds, clerk=clerk)
    assert code == 0 and "the user approved the funding sells (you said yes)" in text and "T1-F1  sell 5 of 5 MSFT" in text and text.count("FILLED") >= 3


# ---------------------------------------------------------------- json, save, fresh

def test_json_prints_the_order_and_save_writes_the_desks_own_shape_which_can_be_replayed_again(tmp_path):
    code, text = run(tmp_path, "--no-model", "--cash", "10000", "--json")
    data = json.loads(text)
    assert code == 0 and data["order"]["status"] == "traded" and data["order"]["trades"][0]["status"] == "filled" and "funds" in data
    saved = tmp_path / "out.json"
    code, text = run(tmp_path, "--no-model", "--cash", "10000", "--save", str(saved))
    order, funds = replay.load_order(saved)
    assert order.status == "traded" and order.trades[0]["fill"]["shares"] == 44 and funds is None and "FILLED 44" in text and order.schema_version == 4


def test_an_order_a_desk_run_already_traded_is_not_traded_again_unless_fresh_is_given(tmp_path):
    first, _ = replay.load_order(save(tmp_path, decided()))
    flags = ("--no-model", "--cash", "10000", "--mode", "backtest")
    code, text = run(tmp_path, *flags, order=first)
    saved = tmp_path / "traded.json"
    code, text = run(tmp_path, *flags, "--save", str(saved), order=first)
    again, _ = replay.load_order(saved)
    code, text = run(tmp_path, *flags, order=again)
    assert text.count("FILLED 44") == 1 and "run 2 of this order" in text and len(replay.load_order(saved)[0].trader_runs) == 1
    traded_in_paper, _ = replay.load_order(save(tmp_path, decided()))
    code, text = run(tmp_path, "--no-model", "--cash", "10000", "--mode", "paper", "--no-ledger", "--now", "2026-11-05 10:30", order=traded_in_paper)
    assert code == 0 and "FILLED 44 at 167.0835" in text
    saved_paper = tmp_path / "paper.json"
    run(tmp_path, "--no-model", "--cash", "10000", "--mode", "paper", "--no-ledger", "--save", str(saved_paper), order=decided())
    paper_order, _ = replay.load_order(saved_paper)
    assert paper_order.status == "traded"
    code, text = run(tmp_path, "--no-model", "--cash", "10000", "--fresh", order=paper_order)
    assert code == 0 and "BACKTEST mode" in text and "FILLED 44 at 166.0830" in text and "run 2 of this order" not in text
    assert replay.start_fresh(paper_order).status == "decided" and paper_order.trades == [] and paper_order.trader_runs == []


# ---------------------------------------------------------------- the model and paper mode

def test_a_run_with_the_model_uses_the_trader_it_is_given_and_a_run_with_no_model_available_still_trades(tmp_path):
    model = ScriptedModel([*PLACE_ALL, say(answer("Bought 44 shares of PG at 166.083.", ("T1", "filled")))])
    code, text = run(tmp_path, "--cash", "10000", trader=Trader(mode="backtest", model=model))
    assert code == 0 and "model scripted/trader at temperature 0" in text and "placed by model" in text and "summary (by model): Bought 44 shares of PG at 166.083." in text
    code, text = run(tmp_path, "--cash", "10000")                                                                                                  # no model flag and none can be built (the autouse fixture)
    assert code == 0 and "no model could run" in text and "FILLED 44" in text and "the Trader's model could not start" in text


def test_paper_mode_asks_for_quotes_and_the_no_ledger_flag_keeps_the_real_ledger_untouched(tmp_path):
    clerk = MarketClerk(quotes={"PG": 167.0}, bars=BARS)
    code, text = run(tmp_path, "--mode", "paper", "--no-model", "--cash", "10000", "--no-ledger", clerk=clerk)
    assert code == 0 and "PAPER mode, prices: the current quotes" in text and "FILLED 44 at 167.0835" in text and clerk.asked("quotes") == [["PG"]] and clerk.asked("bars") == []
    clerk = MarketClerk(quotes={"PG": 167.0}, bars=BARS)
    with pytest.raises(AssertionError, match="a test tried to write to the real ledger"):
        run(tmp_path, "--mode", "paper", "--no-model", "--cash", "10000", clerk=clerk)                                                              # without the flag the real ledger is the Trader's: the guard of the tests refuses to let it be written


def test_the_backtest_asks_for_bars_and_never_for_a_quote():
    clerk = MarketClerk(quotes={"PG": 167.0}, bars=BARS)
    lines = []
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as folder:
        replay.main(["--order", str(save(Path(folder), decided())), "--no-model", "--cash", "10000"], clerk=clerk, out=lines.append)
    assert clerk.asked("quotes") == [] and clerk.asked("bars") == [["PG", "SPY"]]


# ---------------------------------------------------------------- the real Data Clerk over day files

def test_the_whole_replay_over_the_real_data_clerk_and_stored_day_files_calls_no_model(tmp_path, monkeypatch):
    from test_data_clerk_bars import ForbiddenModel
    days = [date(2026, 11, 5), date(2026, 11, 6), date(2026, 11, 9)]
    opens = {"PG": [165.0, 166.0, 167.0], "SPY": [400.0, 400.0, 401.0]}
    rows = {d: [{"ticker": s, "open": opens[s][i], "high": opens[s][i] + 2.5, "low": opens[s][i] - 1.0, "close": opens[s][i] + 0.5, "volume": 1000} for s in opens] for i, d in enumerate(days)}
    root = tmp_path / "day_aggs_v1"
    write_day_files(root, rows)
    real = DataClerk(toolbox=object(), model=ForbiddenModel())
    real.tools.bars_root = root
    code, text = run(tmp_path, "--no-model", "--cash", "10000", clerk=real)
    assert code == 0 and "FILLED 44 at 166.0830 = $7,307.65" in text and "BACKTEST mode, prices: the stored daily bars" in text


def test_without_a_clerk_the_replay_builds_the_bars_only_data_clerk_with_a_model_that_is_never_called(tmp_path, monkeypatch):
    built = []

    class Recorder(MarketClerk):
        def __init__(self, **kw):
            built.append(kw)
            super().__init__(bars=BARS)

    monkeypatch.setattr(replay, "DataClerk", Recorder)
    path = save(tmp_path, decided())
    assert replay.main(["--order", str(path), "--no-model", "--cash", "10000"], out=lambda t: None) == 0
    assert len(built) == 1 and set(built[0]) == {"model"} and type(built[0]["model"]) is object


# ---------------------------------------------------------------- what goes wrong

def test_a_file_that_is_missing_or_not_json_or_not_an_order_is_exit_code_2_with_the_reason_on_stderr(tmp_path, capsys):
    assert replay.main(["--order", str(tmp_path / "missing.json")], clerk=MarketClerk()) == 2
    assert "there is no such file" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert replay.main(["--order", str(bad)], clerk=MarketClerk()) == 2 and "cannot be read as JSON (JSONDecodeError)" in capsys.readouterr().err
    notorder = tmp_path / "x.json"
    notorder.write_text('{"order": {"status": "decided", "colour": 1}}', encoding="utf-8")
    assert replay.main(["--order", str(notorder)], clerk=MarketClerk()) == 2 and "not a work order" in capsys.readouterr().err
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2]", encoding="utf-8")
    assert replay.main(["--order", str(arr)], clerk=MarketClerk()) == 2 and "not a work order" in capsys.readouterr().err


def test_a_moment_that_is_not_written_the_way_it_should_be_is_exit_code_2(tmp_path, capsys):
    code, _ = run(tmp_path, "--no-model", "--now", "yesterday")
    assert code == 2 and '--now must be written "YYYY-MM-DD HH:MM"' in capsys.readouterr().err
    with pytest.raises(ValueError):
        replay.moment_of("2026-11-05T10:30")


def test_an_order_with_no_permitted_decision_needs_a_moment_given_and_otherwise_there_is_nothing_to_replay(tmp_path, capsys):
    code, _ = run(tmp_path, "--no-model", "--cash", "10", order=decided(answer=False))
    assert code == 2 and "no decision of this order permits a trade" in capsys.readouterr().err
    code, text = run(tmp_path, "--no-model", "--cash", "10", "--now", "2026-11-05 10:30", order=decided(answer=False))
    assert code == 0 and "no trades: no decision of this order permits one" in text


def test_a_refused_run_is_exit_code_1(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING", "false")
    code, text = run(tmp_path, "--no-model", "--cash", "10000")
    assert code == 1 and "the Trader did not run" in text and "PAPER_TRADING is not true" in text
    monkeypatch.setenv("PAPER_TRADING", "true")
    order = decided()
    order.status = "recommended"
    code, text = run(tmp_path, "--no-model", "--cash", "10000", order=order)
    assert code == 1 and "the Trader needs an order the Account Manager decided on (status recommended)" in text


def test_the_portfolio_and_the_cash_flags_cannot_be_given_together_and_the_mode_must_be_one_of_two(tmp_path):
    path = save(tmp_path, decided())
    with pytest.raises(SystemExit) as error:
        replay.main(["--order", str(path), "--portfolio", "a.json", "--cash", "5"])
    assert error.value.code == 2
    with pytest.raises(SystemExit):
        replay.main(["--order", str(path), "--mode", "live"])
    with pytest.raises(SystemExit):
        replay.main([])


def test_the_replay_does_not_write_the_order_it_was_given_and_leaves_no_file_unless_save_is_asked_for(tmp_path):
    path = save(tmp_path, decided())
    before = path.read_bytes()
    replay.main(["--order", str(path), "--no-model", "--cash", "10000"], clerk=MarketClerk(bars=BARS), out=lambda t: None)
    assert path.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == ["order.json"]


def test_the_module_text_names_the_free_command_and_the_one_trader_module_that_may_import_the_data_clerk():
    text = replay.__doc__
    assert "--mode backtest --no-model" in text and "FREE" in text and "the one Trader module that may import the Data Clerk" in text
