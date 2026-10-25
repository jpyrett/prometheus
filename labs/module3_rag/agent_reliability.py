"""How reliable is the Data Clerk? Each question asked several times, success counted (Part 7 of the RAG guide).

Run from the repo root with the `cmu` conda env active:

    python labs/module3_rag/agent_reliability.py                 # 4 questions x 5 runs
    python labs/module3_rag/agent_reliability.py --runs 8
    python labs/module3_rag/agent_reliability.py --only meta-week --runs 10

A model is not deterministic at the provider even at temperature 0, so one run proves little: in Part 7 the same question
about Meta gave a cited answer once and a wrong "there is no Meta news" the next time. This asks every question several
times with the real Data Clerk (the real model, the real index, fetching OFF) and prints how often each succeeded.

SUCCESS is fixed here, before any run:
    meta-week   "What is happening with Meta this week?"   the answer cites at least one real source (the index holds 20 Meta
                                                            articles, so "there is no news" is wrong). A short question can score just
                                                            under the relevance bar, so this tests whether the agent rewords and retries.
    mkbhd       "What did MKBHD say about Apple's foldable iPhone?"   it cites the MKBHD article
    boeing      "Tell me about Nvidia's partnership with Boeing."     it says the sources do not cover it and cites nothing (Boeing is
                                                            in none of the articles)
    hello       "Hello! What can you do?"                   it answers without calling a tool
An answer with a problem (an invalid citation, empty) never counts as a success. Cost: about a tenth of a cent per run.
"""
import argparse
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so "import prometheus" works from any folder

from prometheus.agents.data_clerk.agent import DataClerk  # noqa: E402

ADMITS = re.compile(r"(?i)(do(es)? not|don't|doesn't|didn't|cannot|can't|couldn't|could not|unable|no (supporting|information|mention|article|news|source)|nothing|none of|not (mention|cover|contain|include|find|found))")


def admits(answer: str) -> bool:
    """The answer says the sources do not cover it. The model writes curly apostrophes (couldn't), so they are straightened first."""
    return bool(ADMITS.search(answer.replace("\u2019", "'")))
QUESTIONS = {
    "meta-week": ("What is happening with Meta this week?", lambda r: bool(r.references)),
    "mkbhd": ("What did MKBHD say about Apple's foldable iPhone?", lambda r: any("MKBHD" in x["headline"] for x in r.references)),
    "boeing": ("Tell me about Nvidia's partnership with Boeing.", lambda r: not r.references and admits(r.answer)),
    "hello": ("Hello! What can you do?", lambda r: not r.tool_calls and bool(r.answer)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=5, help="runs per question (default 5)")
    parser.add_argument("--only", default="", help="comma-separated question ids")
    parser.add_argument("--model", help="an OpenRouter model id instead of the data_clerk role's model")
    parser.add_argument("--workers", type=int, default=3, help="runs at a time (default 3)")
    args = parser.parse_args()
    ids = [i for i in QUESTIONS if not args.only or i in args.only.split(",")]
    try:
        clerk = DataClerk(model_id=args.model)  # fetching stays off: nothing is fetched or written
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    jobs = [(i, n) for i in ids for n in range(args.runs)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: (job[0], clerk.ask(QUESTIONS[job[0]][0])), jobs))
    print(f"model: {args.model or 'the data_clerk role'} | {args.runs} runs per question | fetching off\n")
    print(f"{'question':<11}{'success':>9}{'with a problem':>16}{'turns':>7}{'tokens in':>11}{'searches':>10}{'blocked fetches':>17}")
    for i in ids:
        rs = [r for k, r in results if k == i]
        ok = sum(1 for r in rs if r.ok and QUESTIONS[i][1](r))
        searches = statistics.mean(sum(c["name"] == "retrieve_news" for c in r.tool_calls) for r in rs)
        fetches = sum(sum(c["name"].startswith("fetch_") and c["outcome"] == "limit" for c in r.tool_calls) for r in rs)
        print(f"{i:<11}{f'{ok}/{len(rs)}':>9}{sum(not r.ok for r in rs):>16}{statistics.mean(r.turns for r in rs):>7.1f}"
              f"{int(statistics.mean(r.tokens_in for r in rs)):>11,}{searches:>10.1f}{fetches:>17}")
    failed = [(i, r) for i, r in results if not (r.ok and QUESTIONS[i][1](r))]
    if failed:
        print("\nWhat the failures said (first of each question):")
        for i in dict.fromkeys(i for i, _ in failed):
            r = next(r for k, r in failed if k == i)
            print(f"  {i}: {r.answer[:170]!r}  | tools: {[(c['name'], c['outcome']) for c in r.tool_calls]}  | problems: {r.problems}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
