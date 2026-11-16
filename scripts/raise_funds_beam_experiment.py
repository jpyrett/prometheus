"""LIVE experiment: does a wider beam find the best plan of the complex scenario? (design/week4_tot_guide.md section 9j)

    python scripts/raise_funds_beam_experiment.py                       # beam 3, 6 and 10 (branching 4): PAID, a few cents, a few minutes
    python scripts/raise_funds_beam_experiment.py --settings 3:4,10:6   # beam:branching pairs

For each setting it runs the three purchases of the complex scenario with the real gpt-oss-120b (reasoning effort low, the plans of a level in parallel, green-first, lean, depth 5; the evaluator is strict: only holdings in profit, one share of each kept), then compares
each purchase's score with the brute-force best (scripts/raise_funds_oracle.py) for the state that run itself had reached. Needs Jacob's yes: it calls Finnhub and OpenRouter.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from raise_funds_oracle import SCENARIO, TODAY, apply, oracle  # noqa: E402
from prometheus.common.money import price as price_door  # noqa: E402

from prometheus.agents.account_manager.account_manager import AccountManager  # noqa: E402
from prometheus.common import portfolio as pf  # noqa: E402
from prometheus.reasoning import raise_funds as rf  # noqa: E402

PURCHASES = [("LLY", 3), ("SNDK", 1), ("MU", 3)]


def run_setting(beam: int, branching: int, max_depth: int = 5, model: str | None = None) -> dict:
    manager = AccountManager(model_id=model, effort="low" if not model or "gpt-oss" in model else None)   # the low reasoning effort is for gpt-oss: it made qwen3-30b slower and worse (entry 46)
    start = pf.load_portfolio(SCENARIO)
    started = time.perf_counter()
    result = manager.raise_funds_many(start, PURCHASES,
                                      branching=branching, beam_width=beam, **({"max_depth": max_depth, "max_calls": 20 + 8 * beam * max(1, max_depth - 3)} if max_depth else {}))   # max_depth 0: chosen from each purchase (entry 49)
    seconds = time.perf_counter() - started
    prices = {s: price_door(q["price"]) for s, q in result["quotes"].items()}      # a quote is the Data Clerk's float: through the price door once (design choices entry 112)
    current, rows = start, []
    for p in result["purchases"]:
        if p["status"] != "plan":
            rows.append((p["target"], p["status"], None, None, p["stats"], rf.success(p)))
            continue
        sells = [(x["symbol"], x["shares"]) for x in p["plan"]["sells"]]
        best, _ = oracle(current, p["target"], p["shares"], 7, prices)
        rows.append((p["target"], " ".join(f"{s}{k}" for s, k in sells), p["plan"]["score"], best, p["stats"], rf.success(p)))
        current = apply(current, p["target"], p["shares"], sells, prices)
    return {"beam": beam, "branching": branching, "seconds": seconds, "run": result["run"], "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Wider beam, same scenario, live.")
    parser.add_argument("--settings", default="3:4,6:4,10:4", help="beam:branching pairs, comma separated")
    parser.add_argument("--max-depth", type=int, default=0, help="the most stocks in a plan (default 0: chosen from each purchase)")
    parser.add_argument("--model", help="an OpenRouter model id instead of the account_manager role's (for example qwen/qwen3-30b-a3b)")
    args = parser.parse_args(argv)
    for pair in args.settings.split(","):
        beam, branching = (int(x) for x in pair.split(":"))
        r = run_setting(beam, branching, args.max_depth, args.model)
        print(f"\nBEAM {beam}, BRANCHING {branching}, DEPTH {args.max_depth}: {r['seconds']:.0f} s, {r['run']['calls']} calls, {r['run']['tokens_in']} tokens in, {r['run']['tokens_out']} out")
        for target, answer, score, best, stats, ok in r["rows"]:
            if score is None:
                print(f"  {target}: {answer}   success: {ok['ok']}")
            else:
                print(f"  {target}: {'SUCCESS' if ok['ok'] else 'NOT MET'} (profitable {ok['profitable']}, kept {ok['kept']}, bought {ok['bought']} of {ok['asked']})  {score:5.1f} of best {best[0]:5.1f} = {100 * score / best[0]:5.1f}%   nodes {stats['nodes']}, levels {stats['levels']}, completed {stats['completed_plans']}, calls {stats['llm_calls']}"
                      f"   answer {answer}   best {' '.join(f'{s}{k}' for s, k in best[1])}")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
