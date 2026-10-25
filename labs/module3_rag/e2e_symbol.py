"""The whole desk for real, for one symbol or for one term: fetch, store, index, search, answer, recommend (Part 8 of the RAG guide).

Run from the repo root with the `cmu` conda env active:

    python labs/module3_rag/e2e_symbol.py SNDK                  # a symbol: last 7 days, FETCHES (paid APIs) and WRITES news/SNDK.txt
    python labs/module3_rag/e2e_symbol.py MSFT --days 14        # a symbol: last 14 days
    python labs/module3_rag/e2e_symbol.py --term "flying car"   # no symbol, a term: searches what is already stored, fetches nothing
    python labs/module3_rag/e2e_symbol.py SNDK --no-advisor     # stop after the Data Clerk

A SYMBOL RUN USES THE PAID APIs AND WRITES FILES. A TERM RUN DOES NOT: there is no ticker to fetch, so fetching stays off and the Data Clerk
searches the stored news (and may look for tickers in the stored market-wide news with discover_tickers). This is the only lab script that can
fetch. Every other lab script keeps fetching off.

What a symbol run does, in order, and what each step writes:
  1. a LIVE work order for the symbol (live=True says fresh news is required, and its window, --days, becomes the from_date of the fetch)
  2. DataClerk(allow_fetch=True).request_data(order): the model calls fetch_news for the symbol, then searches
       fetch_news  ->  Finnhub and Polygon  ->  news/<SYMBOL>.txt     appended, articles already there (same url) are skipped
                   ->  ingest_symbol  ->  the Chroma index in data/processed/vectorstore     only the new or changed chunks
       retrieve_news  ->  reads the index (writes nothing)
  3. the order carries the data: answer, cited chunks, every chunk the model was shown (with text), the status
  4. Advisor.advise(order): reads those chunks, writes recommendations with a calculated score onto the order. A story counts once per
     direction, so the same article arriving through two feeds is not counted twice (OK = counted, SAME = verified but not counted)
  5. the finished work order is saved as JSON in data/processed/e2e_<SYMBOL>.json  (e2e_term_<term>.json for a term)

A term run does steps 2 to 5 without the fetch: the order has the term as its subject and no ticker.

Every run prints how long each phase took. The news file is the DataAgent's own output (`NEWS_OUTPUT_DIR`, default `news/`): ONE FILE PER SYMBOL,
appended to, never one combined file for everything. A symbol that is only mentioned inside other companies' articles (MU, for instance) has no
file of its own: its articles live in the files of the symbols they were fetched under, and are found through the `tickers` tag in the index.
"""
import time

PROCESS_STARTED = time.perf_counter()  # taken before the heavy imports, so "start-up" includes them

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so "import prometheus" works from any folder

from prometheus.agents.advisor.advisor import Advisor  # noqa: E402
from prometheus.agents.data_clerk.agent import DataClerk  # noqa: E402
from prometheus.common.work_order import new_work_order, verify_action  # noqa: E402
from prometheus.rag.config import news_dir, vector_store_path  # noqa: E402
from prometheus.rag.loaders import check_symbol, load_news  # noqa: E402
from prometheus.rag.store import count_chunks, get_vector_store  # noqa: E402


def snapshot(symbol: str) -> dict:
    """What exists now for this symbol: its news file, the chunks in the index, and how many carry its ticker."""
    path = news_dir() / f"{symbol}.txt"
    articles = load_news(path) if path.exists() else []
    tagged = get_vector_store().get(where={"tickers": {"$contains": symbol}}, include=[])["ids"] if count_chunks() else []
    return {"file": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0, "articles": len(articles),
            "newest": sorted(((a.metadata["published_utc"], a.metadata["headline"]) for a in articles), reverse=True)[:3],
            "chunks_total": count_chunks(), "chunks_tagged": len(tagged)}


def show_snapshot(s: dict) -> None:
    file = f"{s['articles']} articles, {s['bytes']:,} bytes" if s["exists"] else "does not exist"
    print(f"  {s['file']}: {file}\n  index: {s['chunks_total']} chunks in all, {s['chunks_tagged']} tagged with this symbol")


def main() -> int:
    started = PROCESS_STARTED
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbol", nargs="?", help="a ticker symbol, for example SNDK (leave it out and use --term for a search with no symbol)")
    parser.add_argument("--term", help='a term instead of a symbol, for example "flying car": searches the stored news, fetches nothing')
    parser.add_argument("--days", type=int, help="how many days of news (a symbol: default 7; a term: no window unless given)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-advisor", action="store_true", help="stop after the Data Clerk")
    parser.add_argument("--model", help="an OpenRouter model id for the Advisor instead of ADVISOR_MODEL")
    args = parser.parse_args()
    if bool(args.symbol) == bool(args.term):
        print("ERROR: give a symbol, or --term with no symbol, not both and not neither", file=sys.stderr)
        return 1
    try:
        symbol = check_symbol(args.symbol.strip().lstrip("$")) if args.symbol else None
        term = args.term.strip() if args.term else None
        if term is not None and not term:
            raise ValueError("the term is empty")
        days = args.days if args.days is not None else (7 if symbol else None)
        clerk = DataClerk(allow_fetch=bool(symbol))  # the REAL toolbox and the REAL index; fetching is on only when there is a symbol to fetch
        advisor = None if args.no_advisor else Advisor(model_id=args.model)
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    ready = time.perf_counter()
    label = symbol or f"term_{re.sub(r'[^a-z0-9]+', '_', term.lower()).strip('_')}"

    print(f"WHERE THINGS ARE WRITTEN\n  news files    : {news_dir()}/<SYMBOL>.txt   (one file per symbol, appended; a term run writes none)\n"
          f"  the index     : {vector_store_path()}   (Chroma)\n  this run's order: data/processed/e2e_{label}.json\n")
    before = after = None
    if symbol:
        print("BEFORE")
        before = snapshot(symbol)
        show_snapshot(before)
        order = new_work_order(f"What is the latest news on {symbol}, and what does it mean for the stock? Is it a bullish or bearish opportunity?",
                               subject=[symbol], since_days=days, top_k=args.top_k, live=True, criteria={"sentiment": "bullish or bearish"})
        print(f"\nWORK ORDER {order.request_id}: live, subject {symbol}, last {days} days, top_k {args.top_k}, no keywords, fetching ON")
    else:
        print(f"BEFORE\n  index: {count_chunks()} chunks in all. There is no symbol, so nothing can be fetched and fetching is OFF.")
        order = new_work_order(term, subject=term, since_days=days, top_k=args.top_k, live=False, criteria={"sentiment": "bullish or bearish"})
        print(f"\nWORK ORDER {order.request_id}: subject {term!r} (a term, no ticker), {'last ' + str(days) + ' days' if days else 'no time window'}, top_k {args.top_k}, fetching OFF")
    print("running the Data Clerk ...")
    t0 = time.perf_counter()
    clerk.request_data(order)
    clerk_seconds = time.perf_counter() - t0

    d = order.data
    if symbol:
        after = snapshot(symbol)
        print("\n\nAFTER THE DATA CLERK")
        show_snapshot(after)
        print(f"  written : {after['articles'] - before['articles']} new articles in the file, {after['chunks_total'] - before['chunks_total']} new chunks in the index")
        for date, headline in after["newest"]:
            print(f"  newest in the file: {date[:10]}  {headline[:80]}")
    else:
        print(f"\n\nAFTER THE DATA CLERK\n  index: {count_chunks()} chunks in all (nothing is written by a term run)")

    print(f"\n  status: {order.status} | verify_action: {verify_action(order) or 'ok'} | errors: {order.errors or 'none'}")
    print("  tool calls: " + (", ".join(f"{c['name']}({c['outcome']})" for c in d.tool_calls) if d else "none"))
    for c in (d.tool_calls if d else []):
        print(f"     {c['name']} arguments: " + ", ".join(f"{k}={v!r}" for k, v in c["args"].items()))
    if d:
        print(f"  turns {d.turns} | tokens {d.tokens_in} in, {d.tokens_out} out | sources shown {len(d.sources)} | cited {len(d.references)}"
              + (" | NO SUPPORTING TEXT" if d.no_supporting_text else ""))
        print(f"\nDATA CLERK ANSWER\n{d.answer}")
        for r in d.references:
            print(f"  [{r['number']}] {r['publisher']}, {r['published_utc'][:10]}: {r['headline'][:80]}  (score {r['score']:.2f}, chunk {r['chunk_id'][:36]})")
        if d.sources:
            tickers = {}
            for s in d.sources:
                for t in s["tickers"][:4]:
                    tickers[t] = tickers.get(t, 0) + 1
            print("  tickers on the sources shown: " + ", ".join(f"{t} x{n}" for t, n in sorted(tickers.items(), key=lambda kv: -kv[1])[:8]))
        if d.references and symbol and after["newest"]:
            cited = sorted(r["published_utc"][:10] for r in d.references)
            print(f"  RECENCY: the cited articles are from {cited[0]} to {cited[-1]}; the newest article in the file is from {after['newest'][0][0][:10]}")
        for note in d.notes:
            print(f"  note: {note}")
        for problem in d.problems:
            print(f"  PROBLEM: {problem}")

    advisor_seconds = 0.0
    if advisor and order.status == "data_ready":
        print("\nrunning the Advisor ...")
        t1 = time.perf_counter()
        advisor.advise(order)
        advisor_seconds = time.perf_counter() - t1
        run = order.advisor_runs[-1] if order.advisor_runs else {}
        print(f"\nADVISOR ({run.get('model')}): status {order.status}, {len(order.recommendations)} recommendations, {run.get('calls', 0)} call(s), "
              f"tokens {run.get('tokens_in', 0)} in, {run.get('tokens_out', 0)} out" + (f", error: {run['error']}" if run.get("error") else ""))
        for r in order.recommendations:
            score = r["score"] if r["supported"] else "none (no verified signal)"
            print(f"  {r['symbol']:<6} {r['company'] or '':<22} stance {r['stance'] or '?':<8} model_score {r['model_score']!s:<4} calculated score {score!s:<26} "
                  f"signals verified {r['signals_verified']} of {r['signals_claimed']}")
            for s in r["signals"]:
                mark = "FAIL" if not s["verified"] else "OK  " if s.get("counted") else "SAME"
                print(f"         {mark} [{s['source']}] {s['direction']}: {s['quote'][:100]!r}"
                      + ("" if s["verified"] else f"   <- {s['reason']}") + (f"   <- {s['note']}" if s.get("note") else ""))
            print(f"         {r['signals_verified']} verified, {r['signals_counted']} counted (a story counts once per direction): score {r['score']}")
    elif advisor:
        print(f"\nthe Advisor was not run: the order's status is {order.status}")

    total = time.perf_counter() - started
    print(f"\nTIMING (seconds)\n  start-up (imports, the embedding model, the keys) {ready - started:7.1f}\n  Data Clerk: {'fetch, index, search, answer' if symbol else 'search, answer':<39}"
          f"{clerk_seconds:7.1f}\n  Advisor{'':<44}{advisor_seconds:7.1f}\n  total inside the script{'':<28}{total:7.1f}")

    out = Path(__file__).resolve().parents[2] / "data" / "processed" / f"e2e_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"before": before, "after": after, "seconds": {"startup": round(ready - started, 1), "data_clerk": round(clerk_seconds, 1),
                                                                             "advisor": round(advisor_seconds, 1), "total": round(total, 1)},
                               "order": order.model_dump()}, indent=1, default=str), encoding="utf-8")
    print(f"\nsaved: {out.relative_to(out.parents[2])}")
    return 0 if order.status in ("data_ready", "recommended") else 1


if __name__ == "__main__":
    sys.exit(main())
