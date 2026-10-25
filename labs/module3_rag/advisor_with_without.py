"""What does retrieval change in the Advisor's recommendation and score? The week 3 submission's Case 2, end to end (Part 8 of the RAG guide).

Run from the repo root with the `cmu` conda env active:

    python labs/module3_rag/advisor_with_without.py                       # 5 runs of each condition
    python labs/module3_rag/advisor_with_without.py --runs 1              # a quick try
    python labs/module3_rag/advisor_with_without.py --model openai/gpt-oss-120b      # another model for the Advisor

The work order is Case 2, with no ticker given up front:
    "Case 2: AI memory shortage. Which companies are affected, and what does the recent news say?"

Three conditions, the same model, the same JSON and the same checks on whatever signals come back:
    A  PROMPT ONLY        the ordinary prompt-only Advisor the week 2 design argues against: "recommend as well as you can, with a score
                          from 0 to 100". No sources, no rules about evidence
    B  ADVISOR, NO SOURCES   the designed Advisor (signals must be exact quotes from sources, never invent) given no sources
    C  ADVISOR WITH RETRIEVAL   the Data Clerk first answers the order (DataClerk.request_data: the search, the citation check, the
                          chunks put on the order), then the same designed Advisor reads the CHUNKS (not the Data Clerk's answer)
B against C isolates retrieval. A is added because the design's claim is about the ordinary prompt-only approach, and B (told never to
invent) may simply decline to answer. A was added after a first single run of B and C, and before any multi-run result was seen.

Each recommendation has the model's own `model_score` (recorded, never used) and a `score` CALCULATED by code from the signals that pass
the checks (a signal is a quote copied from a source, checked to be in that source and to name the company).
Fetching is off and the other tools are not available, so nothing is fetched and no file is written.

WHAT IS DECIDED BEFORE THE RUN (so that the numbers cannot be chosen afterwards):
  BY CONSTRUCTION: without sources no signal can pass, so in A and B no recommendation can get a calculated score. That is the design
  working as intended, not a finding. What the run can show is everything else:
    1. how many companies each condition recommends, and whether the sets agree from run to run
    2. how confident the model sounds without evidence (its own model_score, in A and B), and how much that number moves between runs
    3. with retrieval: how many of the signals the model proposed survive the checks, and why the others fail (the quote is not in the
       source; the quote does not name the company; the source does not exist)
    4. with retrieval: how many recommendations get a calculated score, how much it moves between runs, and whether the model's stance
       agrees with it
    5. cost: tokens per run
  A claim "retrieval helped" would need (3) and (4) to show that a real share of the signals are verified and that the calculated scores
  are more stable than the model's own. If most signals fail, that is the finding.
Cost: roughly a few cents for 5 runs of each condition (the Advisor model is the paid nemotron 120B).
"""
import argparse
import json
import re
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so "import prometheus" works from any folder

from prometheus.agents.advisor.advisor import Advisor  # noqa: E402
from prometheus.agents.data_clerk.agent import DataClerk  # noqa: E402
from prometheus.common.work_order import WorkOrder, new_work_order  # noqa: E402

QUERY = "Case 2: AI memory shortage. Which companies are affected, and what does the recent news say?"
OUT = Path(__file__).resolve().parents[2] / "data" / "processed" / "advisor_with_without.json"


class Unavailable:
    """The Data Clerk's toolbox here: every tool that needs it fails, so nothing is fetched (fetching is off anyway)."""

    def __getattr__(self, name):
        def refuse(*args, **kwargs):
            raise RuntimeError("This tool is not available in this comparison")

        return type("F", (), {"get": staticmethod(refuse)})() if name == "financials" else refuse


def spread(values: list[float]) -> float | None:
    return round(max(values) - min(values), 1) if len(values) >= 2 else None


def failure_kind(reason: str | None) -> str:
    """The reason a signal failed, with the source number taken out so that 'source 9 is not one of the sources shown' and 'source 4 ...' count together."""
    return re.sub(r"^source \S+ is not one of", "source N is not one of", reason or "")


def summarize(orders: list[WorkOrder]) -> dict:
    """The numbers of the list above, for one condition, from the finished orders."""
    recs = [r for o in orders for r in o.recommendations]
    signals = [s for r in recs for s in r["signals"]]
    by_symbol: dict[str, list[dict]] = {}
    for o in orders:
        for r in o.recommendations:
            by_symbol.setdefault(r["symbol"], []).append(r)
    model_spreads = [spread([r["model_score"] for r in rs if r["model_score"] is not None]) for rs in by_symbol.values() if len(rs) >= 2]
    calc_spreads = [spread([r["score"] for r in rs if r["score"] is not None]) for rs in by_symbol.values() if len(rs) >= 2]
    model_scores = [r["model_score"] for r in recs if r["model_score"] is not None]
    sets = [frozenset(r["symbol"] for r in o.recommendations) for o in orders]
    union = set().union(*sets) if sets else set()
    inter = set.intersection(*map(set, sets)) if sets else set()
    runs = [run for o in orders for run in o.advisor_runs]
    return {
        "runs": len(orders), "failed_runs": sum(1 for o in orders if o.status != "recommended"),
        "recommendations_per_run": round(len(recs) / len(orders), 1) if orders else 0,
        "companies_in_every_run": sorted(inter), "companies_in_any_run": len(union),
        "signals_claimed": len(signals), "signals_verified": sum(s["verified"] for s in signals),
        "signal_failures": dict(Counter(failure_kind(s["reason"]) for s in signals if not s["verified"]).most_common()),
        "recommendations": len(recs), "with_calculated_score": sum(r["supported"] for r in recs),
        "model_score_mean": round(statistics.mean(model_scores), 1) if model_scores else None,
        "model_score_70_or_more": sum(m >= 70 for m in model_scores), "model_scores": len(model_scores),
        "model_score_spread_median": statistics.median([s for s in model_spreads if s is not None]) if any(s is not None for s in model_spreads) else None,
        "calculated_score_spread_median": statistics.median([s for s in calc_spreads if s is not None]) if any(s is not None for s in calc_spreads) else None,
        "stance_agrees_with_score": f"{sum(r['consistent'] is True for r in recs)} of {sum(r['consistent'] is not None for r in recs)}",
        "tokens_in": sum(r["tokens_in"] for r in runs), "tokens_out": sum(r["tokens_out"] for r in runs),
        "advisor_errors": [o.errors for o in orders if o.errors],
    }


def order() -> WorkOrder:
    return new_work_order(QUERY, top_k=10, criteria={"sentiment": "bullish or bearish"})


def one_run(index: int, advisor: Advisor, clerk: DataClerk) -> tuple[WorkOrder, WorkOrder, WorkOrder]:
    bare = advisor.advise(order(), use_evidence=False, bare=True)
    no_sources = advisor.advise(order(), use_evidence=False)
    with_ = order()
    clerk.request_data(with_)
    return bare, no_sources, advisor.advise(with_, use_evidence=True)


def show(order: WorkOrder, label: str) -> None:
    print(f"----- {label}: status {order.status}, {len(order.recommendations)} recommendations" + (f", errors {order.errors}" if order.errors else ""))
    for r in order.recommendations:
        score = r["score"] if r["supported"] else "none (no verified signal)"
        print(f"  {r['symbol']:<6} stance {r['stance'] or '?':<8} model_score {r['model_score']!s:<4} calculated score {score!s:<26} signals verified {r['signals_verified']} of {r['signals_claimed']}")
        for s in r["signals"]:
            mark = "OK  " if s["verified"] else "FAIL"
            print(f"         {mark} [{s['source']}] {s['direction']}: {s['quote'][:90]!r}" + ("" if s["verified"] else f"   <- {s['reason']}")
                  + (f"   chunk {s['chunk_id'][:30]}" if s["verified"] else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", help="an OpenRouter model id for the Advisor instead of the advisor role's model")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    try:
        advisor = Advisor(model_id=args.model)
        clerk = DataClerk(toolbox=Unavailable())  # the Data Clerk's own model (DEFAULT_MODEL); fetching is off
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    model = args.model or advisor.model.model_name
    print(f"Advisor model: {model} | Data Clerk model: {clerk.model.model_name} | {args.runs} runs of each of the three conditions | fetching off\nwork order: {QUERY!r}\n")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        triples = list(pool.map(lambda i: one_run(i, advisor, clerk), range(args.runs)))
    bare, no_sources, with_ = [t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples]
    show(bare[0], "A  PROMPT ONLY, run 1")
    print()
    show(no_sources[0], "B  ADVISOR, NO SOURCES, run 1")
    print()
    show(with_[0], "C  ADVISOR WITH RETRIEVAL, run 1")
    if with_[0].data:
        print(f"      (the Data Clerk found {len(with_[0].data.sources)} chunks for this run, cited {len(with_[0].data.references)} in its own answer)")
    a, b, c = summarize(bare), summarize(no_sources), summarize(with_)
    ratio = lambda s: f"{s['with_calculated_score']}/{s['recommendations']}"
    rows = [("runs that produced recommendations", *[f"{s['runs'] - s['failed_runs']}/{s['runs']}" for s in (a, b, c)]),
            ("companies recommended per run", *[s["recommendations_per_run"] for s in (a, b, c)]),
            ("companies different over all runs", *[s["companies_in_any_run"] for s in (a, b, c)]),
            ("companies in every run", *[", ".join(s["companies_in_every_run"]) or "none" for s in (a, b, c)]),
            ("signals the model proposed", *[s["signals_claimed"] for s in (a, b, c)]),
            ("signals that passed the checks", *[s["signals_verified"] for s in (a, b, c)]),
            ("recommendations with a calculated score", *[ratio(s) for s in (a, b, c)]),
            ("model's own score: mean", *[s["model_score_mean"] for s in (a, b, c)]),
            ("model's own score: 70 or more", *[f"{s['model_score_70_or_more']}/{s['model_scores']}" for s in (a, b, c)]),
            ("model's own score: median spread over runs", *[s["model_score_spread_median"] for s in (a, b, c)]),
            ("calculated score: median spread over runs", *[s["calculated_score_spread_median"] for s in (a, b, c)]),
            ("stance agrees with the calculated score", *[s["stance_agrees_with_score"] for s in (a, b, c)]),
            ("tokens in / out (all runs)", *[f"{s['tokens_in']:,} / {s['tokens_out']:,}" for s in (a, b, c)])]
    print(f"\n{'=' * 118}\n{'':<46}{'A prompt only':>24}{'B no sources':>24}{'C with retrieval':>24}")
    for label, x, y, z in rows:
        print(f"{label:<46}{x!s:>24}{y!s:>24}{z!s:>24}")
    for name, s in (("A", a), ("B", b), ("C", c)):
        if s["signal_failures"]:
            print(f"\nwhy signals failed, {name}: {s['signal_failures']}")
        if s["advisor_errors"]:
            print(f"\nerrors, {name}: {s['advisor_errors']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"query": QUERY, "advisor_model": model, **{name: {"summary": summary, "orders": [o.model_dump() for o in orders]} for name, summary, orders in
                               (("a_prompt_only", a, bare), ("b_no_sources", b, no_sources), ("c_with_retrieval", c, with_))}}, indent=1, default=str), encoding="utf-8")
    print(f"\ntranscripts (orders with data, sources, recommendations, signals): {OUT.relative_to(OUT.parents[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
