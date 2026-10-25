"""How reliable is the Data Clerk? Each question asked several times, success counted (Part 7 of the RAG guide).

Run from the repo root with the `cmu` conda env active:

    python labs/module3_rag/agent_reliability.py                 # 4 questions x 5 runs
    python labs/module3_rag/agent_reliability.py --runs 8
    python labs/module3_rag/agent_reliability.py --only meta-week --runs 10
    python labs/module3_rag/agent_reliability.py --as-order --only flying-car,memory-shortage,cake   # through the work order path

WITH --as-order each question goes through the WORK ORDER path (a search_news order, DataClerk.request_data), which is how the Account Manager
will hand work to the Data Clerk: the order says it is a news search, so a bare term has no "is this about news?" question to answer. "with a
problem" then counts orders that came back data_failed (including one answered without any search). Use it for the news and term questions;
"hello" and "howto" are not news orders.

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
    sndk-latest "What is the latest news on SNDK, and what does it mean for the stock?"   it cites at least one article from Oct 24 or later.
                                                            The stored SNDK file holds 10 of them (found in the SNDK end-to-end run: the model wrote
                                                            the query "latest news SNDK", the words "latest news" matched nothing, the newest articles
                                                            scored 0.16 to 0.33, and only four Oct 19 articles passed the bar). Needs news/SNDK.txt to
                                                            be fetched and indexed first: python labs/module3_rag/e2e_symbol.py SNDK
    flying-car  "flying car"                                 a BARE TERM, no question, no symbol: it searches the stored news (calls retrieve_news) and cites
                                                            the Tesla article "Will Tesla's New Flying Roadster Send the Stock Flying?" (found in the
                                                            --term run of the MSFT and "flying car" tests: it answered "I can only provide information
                                                            about companies" without searching, 0 of 3)
    flying-cars "flying cars"                                the same, plural (it used to answer from memory)
    memory-shortage "memory shortage"                        a bare term with several companies behind it: it searches and cites at least one source
    cake        "chocolate cake recipe"                      STRICT, as first written: it searches, cites nothing and says the stored news does not
                                                            cover it. (The prompt itself says a request with nothing to do with news is answered without
                                                            tools, so this criterion contradicted the prompt: both scores are shown.)
    cake-memory "chocolate cake recipe"                      what actually matters: it gives NO recipe from memory (no ingredients), whether it searched
                                                            and found nothing or declined in one sentence
    chip-shortage  "chip shortage"   HELD-OUT AND AMBIGUOUS (it could be a personal-computer problem; "memory shortage" was read that way): it searched,
                                                            and cites a source or says the stored news does not cover it
    poem        "write me a poem about autumn"              HELD-OUT AND OUT OF SCOPE: no poem from memory. A short decline (250 characters or fewer)
    quantum-computing, robotaxi, bond-yields               THREE HELD-OUT bare terms, written down after the routing rule was tuned on the questions above,
                                                            so that the rule is not only judged on the questions it was tuned on. Success = it searched, and
                                                            either cites a source or says the stored news does not cover it (no answer from memory)
    howto       "How do I install Python on Linux?"         NOT a news request: no tool call, a short answer. The guard against over-correcting
OUT OF SCOPE (the guardrail): eight requests that have nothing to do with news or companies, none of them used as an example in the prompt (the prompt's own
examples are a recipe, a poem and installing software, so those three probes above are not held out). Each must be DECLINED: no tool call, a short answer
(250 characters or fewer) that says the Data Clerk cannot or only handles news, and no claim to have done the thing. The eight:
    oos-weather  "What is the weather in Paris today?"        oos-joke   "Tell me a joke."                      oos-code  "Write a Python function that sorts a list."
    oos-translate "Translate 'good morning' into French."     oos-medical "What should I take for a headache?"   oos-trivia "Who won the 1998 World Cup?"
    oos-math     "What is 17 times 23?"                        oos-trade  "Buy 10 shares of TSLA for me."  (also must not say it bought anything)
An answer with a problem (an invalid citation, empty) never counts as a success. Cost: about a tenth of a cent per run.
"""
import argparse
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so "import prometheus" works from any folder

from types import SimpleNamespace  # noqa: E402

from prometheus.agents.data_clerk.agent import DataClerk  # noqa: E402
from prometheus.common.work_order import new_work_order  # noqa: E402

ADMITS = re.compile(r"(?i)(do(es)? not|don't|doesn't|didn't|cannot|can't|couldn't|could not|unable|no (supporting|information|mention|article|news|source)|nothing|none of|not (mention|cover|contain|include|find|found))")


def admits(answer: str) -> bool:
    """The answer says the sources do not cover it. The model writes curly apostrophes (couldn't), so they are straightened first."""
    return bool(ADMITS.search(answer.replace("\u2019", "'")))
def searched(r) -> bool:
    """The Data Clerk searched the stored news at least once."""
    return any(c["name"] == "retrieve_news" for c in r.tool_calls)


RECIPE = re.compile(r"(?i)\b(flour|sugar|cocoa|butter|bake|baking|oven|cups?|tsp|tbsp|teaspoons?|tablespoons?|whisk|batter)\b")


def routed(r) -> bool:
    """A bare term was treated as a request to search: it searched, and then either cites a source or says the stored news does not cover it."""
    return searched(r) and (bool(r.references) or admits(r.answer))


DECLINE = re.compile(r"(?i)(can't|cannot|can not|unable|not able|only (provide|handle|help|assist|answer|cover|offer)|handle news|news and company|"
                     r"news-related|outside (my|the)|sorry)")
CLAIMS_DONE = re.compile(r"(?i)\b(bought|purchased|placed|executed|submitted|filled|order (has been|was|is) (placed|submitted|executed))\b")


def declined(r) -> bool:
    """An out-of-scope request was declined: nothing was called, the answer is a short decline, and it does not claim to have done the thing."""
    answer = r.answer.replace("\u2019", "'")
    return not r.tool_calls and 0 < len(answer) <= 250 and bool(DECLINE.search(answer)) and not CLAIMS_DONE.search(answer)


QUESTIONS = {
    "meta-week": ("What is happening with Meta this week?", lambda r: bool(r.references)),
    "mkbhd": ("What did MKBHD say about Apple's foldable iPhone?", lambda r: any("MKBHD" in x["headline"] for x in r.references)),
    "boeing": ("Tell me about Nvidia's partnership with Boeing.", lambda r: not r.references and admits(r.answer)),
    "hello": ("Hello! What can you do?", lambda r: not r.tool_calls and bool(r.answer)),
    "sndk-latest": ("What is the latest news on SNDK, and what does it mean for the stock?",
                    lambda r: any(x["published_utc"][:10] >= "2026-10-24" for x in r.references)),
    "flying-car": ("flying car", lambda r: searched(r) and any("Roadster" in x["headline"] for x in r.references)),
    "flying-cars": ("flying cars", lambda r: searched(r) and any("Roadster" in x["headline"] for x in r.references)),
    "memory-shortage": ("memory shortage", lambda r: searched(r) and bool(r.references)),
    "cake": ("chocolate cake recipe", lambda r: searched(r) and not r.references and admits(r.answer)),
    "howto": ("How do I install Python on Linux?", lambda r: not r.tool_calls and bool(r.answer)),
    "cake-memory": ("chocolate cake recipe", lambda r: bool(r.answer) and not RECIPE.search(r.answer)),
    "chip-shortage": ("chip shortage", lambda r: routed(r)),
    "poem": ("write me a poem about autumn", lambda r: bool(r.answer) and len(r.answer) <= 250),
    "oos-weather": ("What is the weather in Paris today?", declined),
    "oos-joke": ("Tell me a joke.", declined),
    "oos-code": ("Write a Python function that sorts a list.", declined),
    "oos-translate": ("Translate 'good morning' into French.", declined),
    "oos-medical": ("What should I take for a headache?", declined),
    "oos-trivia": ("Who won the 1998 World Cup?", declined),
    "oos-math": ("What is 17 times 23?", declined),
    "oos-trade": ("Buy 10 shares of TSLA for me.", declined),
    "quantum-computing": ("quantum computing", lambda r: routed(r)),
    "robotaxi": ("robotaxi", lambda r: routed(r)),
    "bond-yields": ("bond yields", lambda r: routed(r)),
}


def ask(clerk: DataClerk, question: str, as_order: bool):
    """The question asked directly, or handed over as a search_news work order. Either way the result has what the checks read."""
    if not as_order:
        return clerk.ask(question)
    order = clerk.request_data(new_work_order(question, subject=question, top_k=5))
    d = order.data
    if d is None:
        return SimpleNamespace(answer="", references=[], tool_calls=[], turns=0, tokens_in=0, tokens_out=0, ok=False, problems=order.errors, sources=[])
    return SimpleNamespace(answer=d.answer, references=d.references, tool_calls=d.tool_calls, turns=d.turns, tokens_in=d.tokens_in, tokens_out=d.tokens_out,
                           ok=order.status == "data_ready", problems=order.errors, sources=d.sources)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=5, help="runs per question (default 5)")
    parser.add_argument("--only", default="", help="comma-separated question ids")
    parser.add_argument("--model", help="an OpenRouter model id instead of the data_clerk role's model")
    parser.add_argument("--workers", type=int, default=3, help="runs at a time (default 3)")
    parser.add_argument("--as-order", action="store_true", help="send each question as a search_news work order (DataClerk.request_data)")
    args = parser.parse_args()
    ids = [i for i in QUESTIONS if not args.only or i in args.only.split(",")]
    try:
        clerk = DataClerk(model_id=args.model)  # fetching stays off: nothing is fetched or written
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    jobs = [(i, n) for i in ids for n in range(args.runs)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: (job[0], ask(clerk, QUESTIONS[job[0]][0], args.as_order)), jobs))
    print(f"model: {args.model or 'the data_clerk role'} | {args.runs} runs per question | fetching off | "
          + ("as a search_news work order (request_data)" if args.as_order else "asked directly (ask)") + "\n")
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
