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
| Embedder | sentence-transformers (`all-MiniLM-L6-v2`), loaded through LangChain's `HuggingFaceEmbeddings` | Already installed, runs locally on CPU with no API cost, and keeps the data on my machine. Can be swapped later without changing the rest of the pipeline **[TODO: confirm after testing on real chunks]** |
| Metadata | Ticker, source, document date, document id on every chunk | Lets the work order's subject, source, and time period filter the search |
| Results retrieved | The work order's top-k, defaulting to 5 **[TODO: confirm on real queries]** | Enough evidence for the Advisor without wasting tokens |
| Duplicates | Add MMR only if real results show near-duplicate chunks | Avoids tuning before I have seen the data |

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
