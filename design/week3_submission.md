# Prometheus Capstone: Week 3 Submission (Retrieval)

Jacob Pyrett, CMU Agentic AI, Oct 2026

> Draft. Items marked **[TODO]** need real results from the implementation before submitting.

## 1. Is retrieval required? Yes

Prometheus needs retrieval for three reasons.

- **The model does not know current text.** It has no knowledge of today's news, press releases, or SEC filings. That text has to be fetched from external sources.
- **Case 2 is a meaning question, not a ticker lookup.** A request like "AI memory shortage" will not match by keyword, because relevant articles use different words. That needs semantic search.
- **The text is too large for one prompt.** News, releases, and filings for a group like the S&P 500 will not fit in a context window, and they go stale. Retrieval returns only the few relevant chunks, which keeps token cost down.

Retrieval is for text only. Prices and metrics stay in Parquet or Postgres, accounts and trades stay in Postgres with transaction locking, and quote caching stays in Redis. Those need exact answers, which embeddings cannot give.

## 2. How retrieval is integrated

Text sources (news, press releases, EDGAR filings, paid text files) are ingested in a batch step, outside any agent loop. The step loads each document, splits it into chunks, embeds the chunks, and stores them in a persistent Chroma index. One FastMCP tool searches that index.

The Data Clerk agent calls the tool when a work order needs text, and writes the returned chunks and their metadata onto the work order. The Advisor reads those chunks to score the recommendation, and the Critic can inspect the same chunks. The retriever returns evidence. It does not answer. The Advisor still makes the decision.

The stack is LangChain for loading, splitting, embedding, and retrieving, with LangGraph reserved for the trading desk. The embedder is sentence-transformers (`all-MiniLM-L6-v2`), which I already depend on. LangGraph decides when a work order needs text. It does not build the index.

**[TODO]** Add evidence that this runs: the ingest command, the index size, and one real search result from the external data.

## 3. Retrieval changes the output **[TODO]**

I will run the same work order twice: Case 2, "AI memory shortage."

- **Without retrieval:** the Advisor sees only the work order. Record its recommendation and score.
- **With retrieval:** the Advisor also receives the retrieved chunks. Record the chunks (ticker, source, date, chunk id, similarity score), then the recommendation and score.

The submission will show the two outputs side by side, and point to which retrieved chunk the Advisor's rationale cites. The trace of retrieved chunk ids on the work order is the evidence that the answer is grounded.

**[TODO]** Paste the real comparison here. I will not fill this in with an invented example.

## 4. Key design choices

| Choice | Decision | Reason |
|---|---|---|
| Data source | News, press releases, EDGAR filings. Start with a small sample: one ticker, one source, a short date range | Matches my week 2 design; similarity search is for text, not numbers |
| Chunking | Several hundred to about 1000 characters, with overlap | A retrieved passage must still contain the whole statement. The lab's 100-word chunks are too small for filings |
| Embedder | sentence-transformers (`all-MiniLM-L6-v2`), loaded through LangChain's `HuggingFaceEmbeddings` | Already installed, runs locally on CPU with no API cost, and keeps the data on my machine. Can be swapped later without changing the rest of the pipeline. Confirmed on real chunks: on the same 538 chunks and 23 real questions, the answer was in the text of the top 5 for 23 of 23 with this model, against 22 of 23 for a same-size model trained for question search and 22 of 23 for the larger `all-mpnet-base-v2` (768 numbers, 6 times slower to build). The answer's rank differed on only 5 questions, with as many wins as losses, so I kept the small model. A relevance bar is tuned to one model, so changing it means choosing the bar again |
| Metadata | Ticker, source, document date, document id on every chunk | Lets the work order's subject, source, and time period filter the search |
| Results retrieved | The work order's top-k, defaulting to 5, counted in articles (one hit per article, its best chunk) | Tested on 23 real questions about the stored news: the answer was in the text of the top 5 for 23 of 23 (top 3: 22 of 23, top 1: 19 of 23). A plain chunk search spent 16% of the top-5 slots on a second chunk of an article already listed |
| Duplicates | MMR is an option for broad questions, off by default (`lambda_mult` 0.5, applied after the relevance bar). Narrow questions use plain similarity | Real results do show near-duplicate stories, so I tested MMR instead of guessing. It helps broad questions and costs narrow ones (below) |

**MMR evidence (real news, 281 articles, 538 chunks, 2026-10-25).**

*Broad questions* ("What is happening with Apple / Meta / Nvidia / Tesla?", filtered to the ticker and the last 7 days, top 5). "Overlap" is the mean cosine similarity between the five hits, so lower means more varied. "Relevance" is their mean score.

| Question | Plain: relevance / overlap | MMR 0.5: relevance / overlap |
|---|---|---|
| Apple | 0.54 / 0.53 | 0.51 / 0.43 |
| Meta | 0.49 / 0.50 | 0.43 / 0.39 |
| Nvidia | 0.57 / 0.60 | 0.54 / 0.51 |
| Tesla | 0.54 / 0.56 | 0.49 / 0.44 |

MMR cut the overlap by 0.09 to 0.12 for 0.03 to 0.06 of relevance. For Apple it brought in the biggest story, the iPhone Duo launch, which the plain top 5 missed. It cannot create variety that is not there: Meta's news is mostly one story, so MMR swapped a relevant near-duplicate for a loosely related article, and without the relevance bar it picked one that scored 0.19. So the bar is applied before the spreading.

*Narrow questions* (23 questions with a known answer, answer found in the text of the top 5):

| Setting | Top 1 | Top 3 | Top 5 | MRR |
|---|---|---|---|---|
| Plain | 19/23 | 22/23 | 23/23 | 0.89 |
| MMR 0.7 | 19/23 | 21/23 | 23/23 | 0.88 |
| MMR 0.5 | 19/23 | 20/23 | 22/23 | 0.86 |
| MMR 0.3 | 19/23 | 19/23 | 19/23 | 0.83 |

Some questions have two articles that both answer them, and MMR drops the second one on purpose. At 0.3 it loses four answers. So the rule from the course lab held on real data: use MMR for a broad question over redundant news, and skip it for a narrow one. At 0.5 the spreading is real and the cost is small, 0.7 changes little, and 0.3 loses answers.

Limits: 27 questions that I wrote myself, on one week of news for four tickers. The questions, the ground truth and the script that reproduces every number are in `labs/module3_rag/retrieval_experiments.py`, and the full write-up is in `docs/design_choices.md` entry 20.

## 5. Failure mode: wrong or stale context

**Risk.** A semantic search for "AI memory shortage" with no constraints can return chunks about a different company or an older year. The Advisor would then score a stock on evidence that is not about it.

**How the design manages it.**

- The Data Clerk applies the work order's subject, source, and time period as a metadata filter on the search itself, not only in the query text.
- Chunk ids and metadata are written onto the work order, so the Critic and the audit log can check what a score was based on.
- If the filtered search finds nothing relevant, the Data Clerk reports "no supporting text" on the work order. The Advisor does not guess.

## Reference: who owns what

| Layer | Toolkit | Job |
|---|---|---|
| Ingest and retrieve | LangChain | Load files, split them, embed them, persist them, return top-k chunks |
| Trading desk | LangGraph | Account Manager routes the work order; the Data Clerk calls the retriever when the order asks for text |
| Serving the retriever | FastMCP (in `prometheus/mcp`) | One search tool for filings or news. The graph calls the tool; the tool does not call another agent |
| Embeddings | sentence-transformers | `all-MiniLM-L6-v2` on the laptop; a larger local model can wait for the DGX |
| Vector store | Chroma | Named in the week 2 design. Postgres stays for accounts, trades, and full-text search; Redis for quote and API cache |
