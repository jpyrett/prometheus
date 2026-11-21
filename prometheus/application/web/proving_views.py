"""What the page shows of a Proving Ground run: PURE (design/proving_ground_spec.md section 9; P12: every number is computed in `proving/`, once, and the page draws).

    run_view(run)                  # a saved `ProvingRun` as the page reads it: the run's JSON, its caveats as sentences, the five KPI tiles, the days that have a desk; the work orders themselves are left out (they are served by day)
    run_row(row)                   # one row of the list of saved runs (the store's index row)
    day_view(run, day)             # one decision day's desk: `views.work_order_view` of the work order the Time Machine decided on that day (the same table and panel as the Desk tab)
    record_of(run, day, key)       # the raw records behind one row of that desk (`views.records_of`)
    replay_events(run)             # a saved run as the stream of `pv_*` events a live run sends, in the order a live run sends them

No model, no network, no clock. This module is the ONLY place of the page's bridge that knows the shape of a `ProvingRun` (as `views.py` is the only one that knows a recommendation). It adds no statistic: the tiles are fields of `Stats`.
"""
from prometheus.application.web import views
from prometheus.proving.record import CAVEATS, ProvingRun

INDEX_FIELDS = ("run_id", "kind", "status", "created_utc", "prompt", "range", "headline", "return_pct", "excess_return_pct", "speed")      # `speed` (fast or faithful) was added with the speeds: a row written before it has None


def caveat_list(codes) -> list[dict]:
    """The caveats of a run as the page shows them, each one sentence as written (P3: never hidden)."""
    return [{"code": code, "sentence": CAVEATS[code]} for code in codes]


def kpis(run: ProvingRun) -> dict:
    """The five tiles of the headline strip (spec 13.2): Return, vs SPY, Win rate (with its n and the small-sample mark), Max drawdown, Recommendations (traded, denied, open). Every figure is a field of `Stats`."""
    portfolio, outcomes = run.stats.portfolio, run.stats.outcomes
    return {"return_pct": portfolio.return_pct if portfolio else None, "excess_return_pct": portfolio.excess_return_pct if portfolio else None, "benchmark_return_pct": portfolio.benchmark_return_pct if portfolio else None,
            "win_rate": {"value": outcomes.win_rate, "n": outcomes.closed, "small_sample": outcomes.small_sample},
            "max_drawdown_pct": portfolio.max_drawdown_pct if portfolio else None,
            "recommendations": {"total": len(run.outcomes), "traded": sum(1 for o in run.outcomes if o.traded), "denied": sum(1 for o in run.outcomes if o.verdict == "denied"), "open": outcomes.open}}


def run_row(row: dict) -> dict:
    """A row of the index of saved runs, with the fields the list shows and no more."""
    return {k: row.get(k) for k in INDEX_FIELDS}


def run_view(run: ProvingRun) -> dict:
    """A saved run as the page reads it (see the module text). The work orders of the days are not in it: `desks` lists the days that have one."""
    body = run.model_dump(mode="json")
    desks = sorted((body.get("coverage") or {}).get("desks") or {})
    body["coverage"] = {k: v for k, v in (body.get("coverage") or {}).items() if k != "desks"}
    return {**body, "caveats": caveat_list(run.caveats), "caveat_codes": list(run.caveats), "kpis": kpis(run), "desks": desks}


def _desk(run: ProvingRun, day: str) -> dict | None:
    desk = (run.coverage.get("desks") or {}).get(day)
    return desk if isinstance(desk, dict) else None


def day_view(run: ProvingRun, day: str) -> dict | None:
    """The desk of one decision day: the same rows the Desk tab shows (`views.work_order_view`), and the day's own row of the run. None when the run has no desk for that day (a failed day, or a day that was not a decision day)."""
    desk = _desk(run, day)
    if desk is None:
        return None
    row = next((d for d in run.decision_days if d.day.isoformat() == day), None)
    return {"day": day, "title": f"The desk on {day}", "view": views.work_order_view(desk, understood=run.understood), "decision_day": row.model_dump(mode="json") if row else None}


def record_of(run: ProvingRun, day: str, key: str) -> dict | None:
    desk = _desk(run, day)
    return None if desk is None else views.records_of(desk, key)


def replay_events(run: ProvingRun):
    """A saved run as the events a live run sends: `pv_started`; `pv_research` for each decision day; then for each walk day `pv_day` (the equity point, the day's markers, and the decision day's row on a decision day) with the day's `pv_decision`
    and the `pv_outcome`s that close on it; and `pv_finished`. Yields (type, data, pause) where `pause` is how long a page watching the replay would like to wait after the event, in decision days of pacing (the caller scales it)."""
    first = run.range.first_day.isoformat() if run.range else None
    yield "pv_started", {"run_id": run.run_id, "request": run.request, "understood": run.understood, "range": run.range.model_dump(mode="json") if run.range else None,
                         "decision_days": [d.day.isoformat() for d in run.decision_days], "starting_cash": run.equity[0].cash if run.equity else None, "caveats": caveat_list(run.caveats),
                         "parallel_days": (run.coverage.get("timing") or {}).get("parallel_days"), "phase": "research", "speed": run.speed,
                         "timing": {"speed": run.speed, "parallel_days": (run.timing or run.coverage.get("timing") or {}).get("parallel_days"), "estimate_seconds": (run.timing or {}).get("research_seconds")}}, 0
    cached = set((run.coverage.get("cache") or {}).get("cached_days") or [])
    total = len(run.decision_days)
    for number, row in enumerate(run.decision_days, 1):
        yield "pv_research", {"day": row.day.isoformat(), "status": "cached" if row.day.isoformat() in cached else ("failed" if row.status == "failed" else "done"), "cached": row.day.isoformat() in cached, "recommended": row.recommended,
                              "approved": row.approved, "denied": row.denied, "seconds": 0, "reason": row.reason if row.status == "failed" else None, "done": number, "total": total, "usage": ((run.coverage.get("usage") or {}).get("per_day") or {}).get(row.day.isoformat()),
                              "stages": (((run.timing or run.coverage.get("timing") or {}).get("per_day") or {}).get(row.day.isoformat()) or {}).get("stages") or {}}, 1
    by_day = {d.day: d for d in run.decision_days}
    markers: dict = {}
    for event in run.events:
        markers.setdefault(event.date, []).append(event)
    closing: dict = {}
    for outcome in run.outcomes:
        closing.setdefault(outcome.exit.date if outcome.exit else (outcome.path[-1].date if outcome.path else outcome.decision_day), []).append(outcome)
    for index, point in enumerate(run.equity, 1):
        day = by_day.get(point.date)
        yield "pv_day", {"point": point.model_dump(mode="json"), "events": [e.model_dump(mode="json") for e in markers.get(point.date, [])], "decision_day": day.model_dump(mode="json") if day else None, "index": index, "total": len(run.equity)}, 1
        if day is not None and _desk(run, point.date.isoformat()) is not None:
            yield "pv_decision", {"day": point.date.isoformat(), "order": _desk(run, point.date.isoformat()), "understood": run.understood}, 0
        for outcome in closing.pop(point.date, []):
            yield "pv_outcome", {"outcome": outcome.model_dump(mode="json")}, 0
    for left in sorted(closing):
        for outcome in closing[left]:
            yield "pv_outcome", {"outcome": outcome.model_dump(mode="json")}, 0
    yield "pv_finished", {"run_id": run.run_id, "status": run.status, "stats": run.stats.model_dump(mode="json"), "headline": run.stats.headline.model_dump(mode="json"), "caveats": caveat_list(run.caveats),
                          "equity": [p.model_dump(mode="json") for p in run.equity], "saved_id": run.run_id, "langsmith": run.langsmith.model_dump(mode="json") if run.langsmith else None, "seconds": run.seconds,
                          "timing": run.timing or run.coverage.get("timing"), "speed": run.speed, "range_first_day": first}, 0
