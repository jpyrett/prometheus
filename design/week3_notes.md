# Week 3 working notes (not part of the submission)

Moved out of `week3_submission.md` so the submission stays short.

## Versions (checked 2026-10-22, conda env `cmu`, Python 3.13.16)

| Package | Latest found | Notes |
|---|---|---|
| `langchain` | 1.4.7 | Only needed for `create_agent`. Retrieval path does not need it |
| `langchain-core` | 1.7.0 (released Oct 22) | Changes raw streamed tool-call events (`tool-call-delta`). If anything breaks, fall back to 1.6.13 |
| `langgraph` | 1.2.16 | 1.2 added per-node timeouts, node error handlers, beta `DeltaChannel` |
| `langchain-text-splitters` | 1.1.3 | Chunking |
| `langchain-huggingface` | 1.2.3 | Wraps sentence-transformers. Verify it works with installed `sentence-transformers` 6.1.0 |
| `langchain-chroma` | 1.1.0 | `as_retriever(search_type="mmr")` supported |
| `chromadb` | 1.5.9 | Needed by `langchain-chroma` |
| `langgraph-checkpoint-postgres` | 3.2.0 | Later: Account Manager thread memory |
| `langchain-mcp-adapters` | 0.3.2 | Later, only if wrapping MCP tools as LangChain tools |

Already installed: `fastmcp` 4.1.1, `mcp` 2.3.0, `sentence-transformers` 6.1.0. No LangChain / LangGraph / Chroma packages yet.

Use the LangChain 1.x line. Older tutorials (0.1 to 0.3 imports, `RetrievalQA`, `langchain_community.vectorstores`) are the wrong shape. Tutor caveat: 1.4 / 1.2 are newer than the tutor's solid knowledge, so check docs.langchain.com when unsure.

## Working rules

> **Update 2026-10-24:** the AI now writes the code into the repo and explains each piece in `labs/module3_rag/hpw_to_do_langchain_rag.md` (Jacob runs and reviews each step), and `langgraph` is installed. The order is now: plain retrieval by hand (guide Parts 3 to 5), then tools, then the LangGraph (Parts 6 and 7), then the FastMCP tool (Part 8). Whether grading and query rewriting stay is decided on real results in Part 7. The other rules below still hold.

- Jacob writes all code. Tutor explains and reviews only.
- Order: small sample in `data/raw/` -> batch ingest into `data/processed/` -> hand-query -> check results -> expose one Data Clerk MCP tool -> only then LangGraph.
- Keep `langgraph` commented in `requirements.txt` until an agent calls the tool.
- Skip `RetrievalQA` (it retrieves and answers, merging Data Clerk and Advisor).
- No second agent inside retrieval (query rewriting, multi-hop) until plain filtered top-k fails on a real work order.

## Checklist

- [ ] Copy sample historical text files to this computer (one ticker, one source, short date range)
- [ ] Add retrieval packages to `requirements.txt`
- [ ] Batch ingest into Chroma
- [ ] Hand-query the index; save sample results
- [ ] Expose one FastMCP search tool for the Data Clerk
- [ ] Run the Case 2 comparison and paste evidence into the submission
- [ ] Add a pytest for the search tool
- [ ] Edit the submission so it reads like my own writing
