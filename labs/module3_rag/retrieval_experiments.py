"""Retrieval experiments on the real news (Part 5 of the RAG guide; design choices entry 20).

Run from the repo root with the `cmu` conda env active:

    python labs/module3_rag/retrieval_experiments.py questions    # what counts as an answer to each question
    python labs/module3_rag/retrieval_experiments.py scores       # how high do answerable and unanswerable questions score?
    python labs/module3_rag/retrieval_experiments.py k            # how many results are enough? one chunk or one article per slot?
    python labs/module3_rag/retrieval_experiments.py diverse      # near-duplicate stories: does MMR help, and what does it cost?
    python labs/module3_rag/retrieval_experiments.py keywords     # does adding the API keywords to the text help?
    python labs/module3_rag/retrieval_experiments.py filters      # tickers and date windows, and what the ticker filter cannot see
    python labs/module3_rag/retrieval_experiments.py recall       # is the index's approximate search as good as an exact one?
    python labs/module3_rag/retrieval_experiments.py terse        # broad questions worded the way a model words them: short
    python labs/module3_rag/retrieval_experiments.py embedders    # the same questions on the same chunks with other embedding models
                                                                  # (downloads two models on the first run, about 500 MB)

Every experiment reads the real index through the real `search_news`. Only `keywords` builds something: two
throw-away indexes in a temporary folder, deleted at the end. The questions are written by hand about real
articles, and there are two ways to count a success:

    source article   an article whose headline matches the rule next to the question is among the hits
                     (strict: another article may answer the question too)
    answer in text   the text of a retrieved chunk contains the answer (what the LLM would actually read)

`questions` prints the ground truth, so it can be checked.
"""
import argparse
import re
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so "import prometheus" works from any folder

from prometheus.rag.config import news_dir  # noqa: E402
from prometheus.rag.ingest import ingest_documents  # noqa: E402
from prometheus.rag.loaders import load_news  # noqa: E402
from prometheus.rag.retriever import MIN_SCORE, search_news  # noqa: E402
from prometheus.rag.store import HNSW, count_too_long, get_embeddings, get_vector_store  # noqa: E402

# (question, rule on the lower-case headline for the SOURCE article, pattern the retrieved chunk's TEXT must contain)
ANSWERABLE = [
    ("Which company did Apple pick as a new US manufacturing partner for iPhone materials?", lambda h: "qnity" in h, r"qnity"),
    ("What was the verdict of the YouTuber MKBHD on Apple's folding phone?", lambda h: "mkbhd" in h, r"mkbhd"),
    ("Which analyst thinks the foldable iPhone could sell about 20 million units?", lambda h: "munster" in h and ("iphone" in h or "apple" in h), r"munster.*20 million|20 million.*munster|20m units"),
    ("Which firm cut its rating on Apple because the iPhone shipment cuts will not recover?", lambda h: "loop capital" in h, r"loop capital"),
    ("How much has Apple's market value grown while Tim Cook has been CEO?", lambda h: "tim cook" in h and "market value" in h, r"\$5 trillion"),
    ("Why is a UK regulator targeting Apple's contactless payments?", lambda h: ("cma" in h.split() or "uk regulator" in h) and ("apple" in h or "iphone" in h), r"\bcma\b|competition and markets authority"),
    ("What event did Apple confirm for the end of October?", lambda h: "october 28" in h, r"october 28"),
    ("Why did Meta create a new class of non-voting shares?", lambda h: "non-voting" in h, r"non-voting"),
    ("Which politician says Zuckerberg's California deal undercuts warnings of a billionaire exodus?", lambda h: "khanna" in h, r"khanna"),
    ("What is Meta's AI assistant Muse mainly used for?", lambda h: "muse" in h, r"(?=.*muse)(?=.*shopping)"),
    ("Why is Meta's free cash flow negative, and should investors worry?", lambda h: "meta fcf" in h, r"(?=.*free cash flow)(?=.*negative)"),
    ("Which AI video startup did Nvidia invest in?", lambda h: "higgsfield" in h, r"higgsfield"),
    ("How long can Nvidia GPUs last, according to Jensen Huang?", lambda h: "gpus could last" in h, r"(?=.*(8|eight) years)(?=.*gpu)"),
    ("Which Chinese tech companies reportedly ordered Nvidia chips worth $35 billion?", lambda h: "bytedance" in h or ("35 billion" in h and "chinese" in h), r"\$35 ?b"),
    ("Which retailer now sells Nvidia RTX Spark laptops?", lambda h: "newegg" in h, r"newegg"),
    ("Why did IBM's stock fall after strong quarterly results?", lambda h: "ibm" in h.split() or "ibm's" in h.split(), r"(?=.*\bibm\b)(?=.*(tank|fell|fall|drop|slump|slid|plung|declin))"),
    ("Which Fed chair could pop CoreWeave's bubble?", lambda h: "warsh" in h, r"warsh"),
    ("How large is OpenAI's valuation getting?", lambda h: "openai nears" in h, r"1\.25 trillion"),
    ("How big was Nvidia's historic share buyback?", lambda h: "buyback" in h, r"235 billion"),
    ("Why did Tesla abandon plans for the Cybertruck in Australia?", lambda h: "cybertruck" in h, r"(?=.*cybertruck)(?=.*australia)"),
    ("What is Tesla's new flying roadster?", lambda h: "flying roadster" in h, r"flying roadster"),
    ("What did Tesla management say on its third-quarter earnings call?", lambda h: "q3 earnings call" in h, r"(?=.*tesla)(?=.*earnings call)"),
    ("Why might a merger of Tesla and SpaceX make strategic sense?", lambda h: "merger" in h and ("tesla" in h or "spacex" in h), r"(?=.*merg)(?=.*tesla)(?=.*spacex)"),
]
# Questions the corpus cannot answer (checked: these words appear 0 times in the four news files).
NO_ANSWER = [
    "chocolate cake recipe",
    "who won the football game on Sunday",
    "Boeing 737 production delays",
    "Starbucks same-store sales growth",
    "Bitcoin ETF inflows this week",
    "Walmart quarterly earnings",
    "Pfizer vaccine trial results",
]
# The right company but a topic with no news (checked: 0 mentions). Asked with the ticker and last-7-days filters, as the Data Clerk would.
SUBJECT_NO_ANSWER = [("Apple's electric car project", ["AAPL"]), ("Apple's pizza delivery service", ["AAPL"]),
                     ("Tesla's Bitcoin holdings", ["TSLA"]), ("Tesla vehicle recall", ["TSLA"]),
                     ("Nvidia's partnership with Boeing", ["NVDA"]), ("Nvidia antitrust lawsuit ruling", ["NVDA"]),
                     ("Meta's plans to build a theme park", ["META"])]
# Broad questions: the corpus has plenty of news on them, but no single article answers them.
VAGUE = [("What is happening with Apple?", ["AAPL"]), ("What is happening with Meta?", ["META"]),
         ("What is happening with Nvidia?", ["NVDA"]), ("What is happening with Tesla?", ["TSLA"]),
         ("Apple news", None), ("latest news on Nvidia", None), ("how is Tesla doing", None), ("tell me about Meta", None)]
RANKED = 10  # how deep to look when asking "at what rank is the answer?"


# ------------------------------------------------------------------ helpers
def source_rank(hits, rule):
    """1-based rank of the first hit whose article headline matches the rule, or None."""
    return next((i for i, h in enumerate(hits, 1) if rule(h.headline.lower())), None)


def text_rank(hits, pattern):
    """1-based rank of the first hit whose chunk text contains the answer, or None."""
    return next((i for i, h in enumerate(hits, 1) if re.search(pattern, h.text, re.I | re.S)), None)


def evaluate(store, **options):
    """One row per answerable question: (question, best score, source rank, answer-in-text rank, score of that hit)."""
    rows = []
    for question, rule, pattern in ANSWERABLE:
        hits = search_news(question, k=RANKED, min_score=None, store=store, **options)
        by_text = text_rank(hits, pattern)
        rows.append((question, hits[0].score, source_rank(hits, rule), by_text, hits[by_text - 1].score if by_text else None))
    return rows


def recall_line(ranks):
    n = len(ranks)
    at = {k: sum(1 for r in ranks if r and r <= k) for k in (1, 3, 5, 10)}
    mrr = sum(1 / r for r in ranks if r) / n
    return "  ".join(f"top {k}: {at[k]:>2}/{n}" for k in at) + f"  | MRR {mrr:.2f}"


def both_lines(rows, indent="  "):
    return (f"{indent}source article  " + recall_line([r[2] for r in rows]) + "\n"
            f"{indent}answer in text  " + recall_line([r[3] for r in rows]))


def best_scores_for_no_answer(store, **options):
    return [(q, search_news(q, k=1, min_score=None, store=store, **options)[0]) for q in NO_ANSWER]


def distinct_articles():
    seen, out = set(), []
    for path in sorted(news_dir().glob("*.txt")):
        for doc in load_news(path):
            if doc.id not in seen:
                seen.add(doc.id)
                out.append(doc)
    return out


# ------------------------------------------------------------------ the experiments
def questions(store):
    articles = distinct_articles()
    print(f"{len(ANSWERABLE)} answerable questions. Two ways to count a success:")
    print("  source article : an article whose headline matches the rule is among the hits (strict: other articles may answer too)")
    print("  answer in text : a retrieved chunk's text contains the answer (what the LLM would actually read)\n")
    for number, (question, rule, pattern) in enumerate(ANSWERABLE, 1):
        matching = [d.metadata["headline"] for d in articles if rule(d.metadata["headline"].lower())]
        print(f"{number:>2}. {question}   [{len(matching)} source article{'s' if len(matching) != 1 else ''}]")
        for headline in matching:
            print(f"        - {headline[:100]}")
        print(f"        answer pattern: {pattern}")
    print(f"\n{len(NO_ANSWER)} questions with no answer in the corpus: " + "; ".join(NO_ANSWER))


def scores(store):
    rows = evaluate(store)
    print(f"{'question':<62}{'best':>6}{'source at':>11}{'answer at':>11}{'its score':>11}")
    for question, best, source, by_text, text_score in rows:
        print(f"{question[:60]:<62}{best:>6.2f}{(str(source) if source else '-'):>11}{(str(by_text) if by_text else '-'):>11}"
              f"{(f'{text_score:.2f}' if text_score else '-'):>11}")
    print("  (source at / answer at: the rank among 10 articles, '-' = not in the top 10)\n")
    negatives = best_scores_for_no_answer(store)
    print(f"{'question with no answer in the corpus':<62}{'best':>6}   closest article")
    for question, hit in negatives:
        print(f"{question:<62}{hit.score:>6.2f}   {hit.headline[:60]}")
    print(f"\n{'VAGUE questions (broad, but the corpus has plenty of news)':<62}{'best':>6}   filters")
    vague_best = []
    for question, symbols in VAGUE:
        hit = search_news(question, tickers=symbols, since_days=7, k=1, min_score=None, store=store)[0]
        vague_best.append(hit.score)
        print(f"{question:<62}{hit.score:>6.2f}   tickers={symbols}, last 7 days")
    print(f"\n{'RIGHT COMPANY, TOPIC NOT IN THE NEWS (with ticker and 7-day filters)':<62}{'best':>6}   closest article")
    subject_best = []
    for question, symbols in SUBJECT_NO_ANSWER:
        hit = search_news(question, tickers=symbols, since_days=7, k=1, min_score=None, store=store)[0]
        subject_best.append(hit.score)
        print(f"{question:<62}{hit.score:>6.2f}   {hit.headline[:60]}")
    yes_best = [r[1] for r in rows]
    no_best = [h.score for _, h in negatives]
    print(f"\nbest score, narrow answerable        : min {min(yes_best):.2f}  median {statistics.median(yes_best):.2f}  max {max(yes_best):.2f}")
    print(f"best score, vague answerable         : min {min(vague_best):.2f}  median {statistics.median(vague_best):.2f}  max {max(vague_best):.2f}")
    print(f"best score, unanswerable (any topic) : min {min(no_best):.2f}  median {statistics.median(no_best):.2f}  max {max(no_best):.2f}")
    print(f"best score, right company, no topic  : min {min(subject_best):.2f}  median {statistics.median(subject_best):.2f}  max {max(subject_best):.2f}")
    print(f"\n{'bar':>5} | results for answerable questions: {'narrow':>7} {'vague':>7} | results for unanswerable ones (should be 0): {'any topic':>10} {'right company':>14}")
    for bar in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
        print(f"{bar:>5.2f} | {'':<32}{sum(1 for s in yes_best if s >= bar):>3}/{len(yes_best)}  {sum(1 for s in vague_best if s >= bar):>3}/{len(vague_best)} |"
              f" {'':<44}{sum(1 for s in no_best if s >= bar):>4}/{len(no_best)}  {sum(1 for s in subject_best if s >= bar):>8}/{len(subject_best)}")


def k_and_repeats(store):
    rows = evaluate(store)
    print("One hit per ARTICLE (what search_news returns), rank counted among articles:")
    print(both_lines(rows))
    print("\nOne hit per CHUNK (a bare similarity search), rank counted among chunks:")
    source_ranks, text_ranks, repeats = [], [], []
    for question, rule, pattern in ANSWERABLE:
        scored = store.similarity_search_with_relevance_scores(question, k=RANKED)
        source_ranks.append(next((i for i, (d, _) in enumerate(scored, 1) if rule(d.metadata["headline"].lower())), None))
        text_ranks.append(next((i for i, (d, _) in enumerate(scored, 1) if re.search(pattern, d.page_content, re.I | re.S)), None))
        repeats.append(5 - len({d.metadata["article_id"] for d, _ in scored[:5]}))
    print("  source article  " + recall_line(source_ranks))
    print("  answer in text  " + recall_line(text_ranks))
    print(f"  slots in the top 5 wasted on a second chunk of an article already there: {sum(repeats)} of {5 * len(repeats)}"
          f" ({sum(1 for r in repeats if r)} of {len(repeats)} questions lose at least one slot)")
    print("\nRanks (among articles) of the first article whose text contains the answer:", sorted(r[3] for r in rows if r[3]),
          f"| missing from the top {RANKED}: {sum(1 for r in rows if not r[3])}")


def vectors_of(store, hits):
    got = store.get(ids=[h.chunk_id for h in hits], include=["embeddings"])
    by_id = dict(zip(got["ids"], got["embeddings"]))
    return np.array([by_id[h.chunk_id] for h in hits])


def pairwise_similarity(store, hits):
    v = vectors_of(store, hits)
    sims = v @ v.T
    n = len(hits)
    return (sims.sum() - n) / (n * (n - 1))  # mean of the off-diagonal cosine similarities (vectors have length 1)


def diverse(store):
    settings = [("plain", {}), ("MMR 0.7", dict(diverse=True, lambda_mult=0.7)),
                ("MMR 0.5", dict(diverse=True, lambda_mult=0.5)), ("MMR 0.3", dict(diverse=True, lambda_mult=0.3))]
    print("BROAD questions, top 5 articles, filter on the company's ticker and the last 7 days.")
    print("relevance = mean score of the 5 hits; overlap = mean cosine similarity between the 5 hits (lower = more varied)\n")
    print(f"{'question':<34}{'setting':<10}{'relevance':>10}{'overlap':>9}{'publishers':>12}{'days':>6}")
    shown = {}
    for question, symbols in VAGUE[:4]:
        for name, options in settings:
            hits = search_news(question, tickers=symbols, since_days=7, k=5, min_score=None, store=store, **options)
            shown[(question, name)] = hits
            print(f"{question:<34}{name:<10}{statistics.mean(h.score for h in hits):>10.2f}{pairwise_similarity(store, hits):>9.2f}"
                  f"{len({h.publisher for h in hits}):>12}{len({h.published_utc[:10] for h in hits}):>6}")
        print()
    for question, symbols in VAGUE[:2]:
        with_bar = search_news(question, tickers=symbols, since_days=7, k=5, min_score=MIN_SCORE, diverse=True,
                               lambda_mult=0.5, store=store)
        for name, hits in (("plain", shown[(question, "plain")]), ("MMR 0.5", shown[(question, "MMR 0.5")]),
                           (f"MMR 0.5, bar {MIN_SCORE} applied first", with_bar)):
            print(f"{question}  [{name}]")
            for h in hits:
                print(f"   {h.score:.2f}  {h.publisher[:10]:<10} {h.published_utc[5:10]}  {h.headline[:84]}")
            print()
    print("NARROW questions (the 23 answerable ones), ranks among 10 articles: what does diversity cost?")
    for name, options in settings:
        print(f"  {name}\n" + both_lines(evaluate(store, **options), "     "))


def build_scratch(path, with_keywords, embeddings=None):
    keywords, current = {}, None
    for file in sorted(news_dir().glob("*.txt")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.startswith("id: "):
                current = line[4:].strip()
            elif line.startswith("keywords: ") and current and line[10:].strip():
                keywords.setdefault(current, line[10:].strip())
            elif line == "<<<END>>>":
                current = None
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    scratch = Chroma(client=client, collection_name="scratch", embedding_function=embeddings or get_embeddings(),
                     collection_configuration={"hnsw": HNSW})  # the same search settings as the real index
    docs = []
    for file in sorted(news_dir().glob("*.txt")):
        for doc in load_news(file):
            text = doc.page_content
            if with_keywords and doc.id in keywords:
                text += f"\n\nKeywords: {keywords[doc.id]}"
            docs.append(Document(id=doc.id, page_content=text, metadata=doc.metadata))
    stats = ingest_documents(docs, scratch, too_long=(lambda texts: count_too_long(texts, embeddings)) if embeddings else None)
    return scratch, stats, len(keywords)


def keyword_test(_real_store):
    with tempfile.TemporaryDirectory() as tmp:
        results = {}
        for name, flag in (("without keywords (what we store now)", False), ("with keywords added to the text", True)):
            scratch, stats, with_kw = build_scratch(Path(tmp) / ("keywords" if flag else "plain"), flag)  # two folders: never build one index on top of the other
            rows = evaluate(scratch)
            negatives = [h.score for _, h in best_scores_for_no_answer(scratch)]
            vague = [search_news(q, tickers=s, since_days=7, k=1, min_score=None, store=scratch)[0].score for q, s in VAGUE]
            results[name] = rows
            print(f"{name}: {stats['articles']} articles -> {stats['chunks']} chunks ({with_kw} articles have keywords)")
            print(both_lines(rows, "   "))
            yes = [r[1] for r in rows]
            print(f"   best score: narrow answerable median {statistics.median(yes):.2f} (min {min(yes):.2f})"
                  f" | vague min {min(vague):.2f} | unanswerable median {statistics.median(negatives):.2f} (max {max(negatives):.2f})"
                  f" | gap, lowest answerable minus highest unanswerable: {min(yes) - max(negatives):+.2f}\n")
        base, with_kw_rows = results.values()
        for label, index in (("source article", 2), ("answer in text", 3)):
            changed = [(b[0], b[index], w[index]) for b, w in zip(base, with_kw_rows) if b[index] != w[index]]
            print(f"{label}: {len(changed)} of {len(base)} questions changed rank")
            for question, before, after in changed:
                print(f"   rank {before or 'none'} -> {after or 'none'}   {question[:70]}")


def recall(store):
    """Is the index's approximate search as good as an exact one? Compare with brute force over every stored vector."""
    got = store.get(include=["embeddings"])
    ids = np.array(got["ids"])
    matrix = np.array(got["embeddings"], dtype=np.float32)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    asked = [q for q, _, _ in ANSWERABLE] + NO_ANSWER + [q for q, _ in VAGUE]
    depth, overlap, identical = 100, 0.0, 0
    for question in asked:
        query = np.array(store.embeddings.embed_query(question), dtype=np.float32)
        exact = ids[np.argsort(-(matrix @ (query / np.linalg.norm(query))))][:depth]
        found = [d.id for d, _ in store.similarity_search_with_score(question, k=depth)]
        overlap += len(set(exact) & set(found)) / depth
        identical += list(exact[:10]) == found[:10]
    print(f"{len(ids)} chunks, {len(asked)} questions")
    print(f"the search finds {overlap / len(asked):.1%} of the true top {depth} chunks")
    print(f"the top 10 is exactly the true top 10 for {identical} of {len(asked)} questions")
    print("(after `ingest --rebuild --all` with the settings in store.py this is about 100% and all of the questions)")


# The model we use, then two others that need no query prefix. The questions are the same, the chunks are the same
# (800 characters, none over any of the three limits), and so are the search settings. Only the vectors differ.
EMBEDDERS = ["sentence-transformers/all-MiniLM-L6-v2",       # ours: 22M parameters, 384 numbers, reads 256 tokens
             "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # the same size, trained on question -> passage pairs
             "sentence-transformers/all-mpnet-base-v2"]      # the LangChain tutorial's model: 110M parameters, 768 numbers


def embedders(_real_store):
    rows = []
    for name in EMBEDDERS:
        model = HuggingFaceEmbeddings(model_name=name, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
        limit, dims = model._client.max_seq_length, len(model.embed_query("x"))
        with tempfile.TemporaryDirectory() as tmp:
            started = time.monotonic()
            scratch, stats, _ = build_scratch(Path(tmp) / "x", False, embeddings=model)
            build_seconds = time.monotonic() - started
            evaluated = evaluate(scratch)
            no_answer = [h.score for _, h in best_scores_for_no_answer(scratch)]
            company = [search_news(q, tickers=s, since_days=7, k=1, min_score=None, store=scratch)[0].score for q, s in SUBJECT_NO_ANSWER]
            vague = [search_news(q, tickers=s, since_days=7, k=1, min_score=None, store=scratch)[0].score for q, s in VAGUE]
        yes = [r[1] for r in evaluated]
        rows.append((name, dims, limit, stats, build_seconds, evaluated, yes, vague, no_answer, company))
        print(f"{name}\n   {dims} numbers per vector, reads {limit} tokens, {stats['chunks']} chunks ({stats['over_limit']} over the limit), "
              f"built in {build_seconds:.1f} s on the CPU")
        print(both_lines(evaluated, "   "))
        print(f"   best score  narrow answerable {min(yes):.2f} to {max(yes):.2f} | vague {min(vague):.2f} to {max(vague):.2f}"
              f" | not in the news {min(no_answer):.2f} to {max(no_answer):.2f} | right company, no topic {min(company):.2f} to {max(company):.2f}")
        # the best bar this model could have: the bar that keeps the most answerable questions while stopping the most unanswerable ones
        best = None
        for bar in np.arange(0.05, 0.95, 0.01):
            kept = sum(1 for s in yes + vague if s >= bar)
            stopped_any = sum(1 for s in no_answer if s < bar)
            stopped_company = sum(1 for s in company if s < bar)
            score = (kept == len(yes + vague), stopped_any + stopped_company, bar)
            if best is None or score > best[0]:
                best = (score, bar, kept, stopped_any, stopped_company)
        _, bar, kept, stopped_any, stopped_company = best
        print(f"   best bar that keeps all {len(yes + vague)} answerable questions: {bar:.2f}, which stops {stopped_any}/{len(no_answer)} not-in-the-news"
              f" and {stopped_company}/{len(company)} right-company questions\n")
    base = rows[0]
    print("summary (answer in the text of the top 5 / MRR / best bar and what it stops, scores are NOT comparable between models):")
    for name, dims, limit, stats, secs, evaluated, *_ in rows:
        ranks = [r[3] for r in evaluated]
        print(f"   {name.split('/')[1]:<28} top 1 {sum(1 for r in ranks if r and r <= 1)}/23  top 3 {sum(1 for r in ranks if r and r <= 3)}/23  "
              f"top 5 {sum(1 for r in ranks if r and r <= 5)}/23  MRR {sum(1 / r for r in ranks if r) / 23:.3f}  build {secs:.1f} s")


# The way the chat model words a broad question when it writes the tool call (found in the Part 6 round trip): short,
# often with "this week" or "latest". Asked with the company's ticker and the last 7 days, as the model asks.
TERSE = [(f"{company} {tail}", [ticker]) for company, ticker in
         (("Meta", "META"), ("Apple", "AAPL"), ("Nvidia", "NVDA"), ("Tesla", "TSLA"))
         for tail in ("this week", "news", "latest developments")] + \
        [(f"recent {company} news", [ticker]) for company, ticker in
         (("Meta", "META"), ("Apple", "AAPL"), ("Nvidia", "NVDA"), ("Tesla", "TSLA"))]


def terse(store):
    best = []
    print(f"{'terse question (ticker filter, last 7 days)':<46}{'best score':>11}")
    for question, symbols in TERSE:
        score = search_news(question, tickers=symbols, since_days=7, k=1, min_score=None, store=store)[0].score
        best.append(score)
        print(f"{question + ' ' + str(symbols):<46}{score:>11.2f}{f'   <- below the bar of {MIN_SCORE}' if score < MIN_SCORE else ''}")
    print(f"\nmin {min(best):.2f}  median {statistics.median(best):.2f}  max {max(best):.2f} | below {MIN_SCORE}: "
          f"{sum(s < MIN_SCORE for s in best)} of {len(best)} | below 0.35: {sum(s < 0.35 for s in best)} | below 0.30: {sum(s < 0.30 for s in best)}")
    ties = [(q, h.score) for q, h in best_scores_for_no_answer(store) if abs(h.score - min(best)) < 0.015]
    if ties:
        print("not in the news, yet scoring about as low as the lowest question above: " + "; ".join(f"{q!r} {sc:.2f}" for q, sc in ties))
        print("so no bar separates them: the score is a first guard, not a judge")


def filters(store):
    now = datetime.now(timezone.utc)
    print(f"now = {now:%Y-%m-%d %H:%M} UTC\n")
    question = "foldable iPhone sales"
    print(f"1. The ticker filter is part of the search. Question: {question!r}, k=5, no relevance bar")
    for symbols in (None, ["NVDA"], ["META"]):
        hits = search_news(question, tickers=symbols, k=5, min_score=None, store=store)
        print(f"   tickers={symbols}: {len(hits)} hits, best {hits[0].score:.2f}, e.g. {hits[0].headline[:60]}")
    unfiltered = search_news(question, k=5, min_score=None, store=store)
    after = [h for h in unfiltered if "NVDA" in h.tickers]
    print(f"   filtering AFTER the search would have kept only {len(after)} of the 5 unfiltered hits for NVDA\n")
    print("2. The date window (rolling, counted back from now). Articles that pass, from the index itself:")
    for days in (1, 2, 3, 5, 7, 10):
        got = search_news("news", since_days=days, k=20, min_score=None, store=store, now=now)
        every = store.get(where={"published_ts": {"$gte": int(now.timestamp()) - days * 86400}}, include=["metadatas"])
        print(f"   last {days:>2} days: {len({m['article_id'] for m in every['metadatas']}):>3} articles stored (search_news asked for 20 got {len(got)})")
    print("\n3. What the ticker filter cannot see: articles about a company that do not carry its ticker.")
    print("   (for a Finnhub article `tickers` is only the symbol it was fetched under, design choices entry 16)")
    rules = {"AAPL": ("apple", "iphone"), "META": ("meta ", "zuckerberg"), "NVDA": ("nvidia", "huang"), "TSLA": ("tesla", "musk")}
    print(f"   {'ticker':<7}{'headline names the company':>28}{'... and has the ticker':>24}{'... but not the ticker':>24}")
    articles = distinct_articles()
    for symbol, words in rules.items():
        about = [d for d in articles if any(w in (d.metadata["headline"].lower() + " ") for w in words)]
        tagged = [d for d in about if symbol in d.metadata["tickers"]]
        untagged = [d for d in about if symbol not in d.metadata["tickers"]]
        named = [d for d in untagged if re.search(rf"\({symbol}\)", d.page_content)]
        print(f"   {symbol:<7}{len(about):>28}{len(tagged):>24}{len(untagged):>24}   ({len(named)} of them write '({symbol})' in their text)")
    q = "Tesla Semi truck production and AI components"
    for symbols in (None, ["TSLA"]):
        hits = search_news(q, tickers=symbols, k=3, min_score=None, store=store)
        print(f"   {q!r} tickers={symbols}:")
        for h in hits:
            print(f"      {h.score:.2f}  {h.headline[:70]}  {h.tickers[:3]}")


SECTIONS = {"questions": questions, "scores": scores, "k": k_and_repeats, "diverse": diverse,
            "keywords": keyword_test, "filters": filters, "recall": recall,
            "embedders": embedders, "terse": terse}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("section", choices=SECTIONS)
    args = parser.parse_args()
    SECTIONS[args.section](get_vector_store())
    return 0


if __name__ == "__main__":
    sys.exit(main())
