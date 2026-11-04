# Week 3 end to end: how the Data Clerk works with RAG

Jacob Pyrett, CMU Agentic AI, Oct 2026. Companion to `design/week3_submission.md` (the answers, 600 to 900 words) and `design/week3_evidence.md` (the full results) and `labs/module3_rag/how_to_do_langchain_rag.md` (the walk-through).

Every box below is a real file, class or function in this repository. Nothing is a plan.

**Updated in Week 4 (press releases, design choices entry 34; then filings, entry 35, and the text of a 10-Q from SEC EDGAR, entry 36; notes in `design/week4.md`).** The diagrams keep what Week 3 built and add press releases as a second source and SEC filings as a third. **Everything new is drawn in yellow** (a filings box is yellow too, and says filings), so the Week 3 picture is still there and the Week 4 additions can be seen at a glance. Counts that changed: the Data Clerk has **ten** tools (it had eight, then nine) and its MCP server offers **seven** by default (it offered five, then six). The picture `design/week3_e2e_diagrams.png` was drawn from the Week 3 version of these diagrams and does not show the yellow boxes until it is rendered again.

**How to read the colors**

| Color | Meaning |
|---|---|
| blue | a LangChain class or function (`langchain_core`, `langchain_chroma`, `langchain_huggingface`, `langchain_text_splitters`, `langchain_openai`) |
| green | a LangGraph class or function (`langgraph.graph`, `langgraph.prebuilt`) |
| grey | our code: plain Python in `prometheus/` |
| orange | something outside this machine (paid APIs, OpenRouter) |
| purple | an LLM |
| yellow | **added in Week 4** for press releases or filings (a new box, or a box whose text changed because of them) |

**What is and is not a "chain".** The loop is a LangGraph `StateGraph`. The pieces around it are LangChain components. The code does **not** use `RetrievalQA` or an LCEL `prompt | model | parser` chain, on purpose: `RetrievalQA` retrieves *and* answers, which would merge the Data Clerk and the Advisor (`design/week3_notes.md`), and here the LLM decides whether to retrieve at all (the "agentic RAG" style from the class slides).

---

## 1. The system map: who calls whom

Numbers show the order of one work order, from the request to the Advisor's recommendation. The batch ingest (top left) runs before any of it, and again whenever new news is fetched.

```mermaid
flowchart TB
    classDef lc fill:#dbeafe,stroke:#2563eb,color:#111
    classDef lg fill:#dcfce7,stroke:#16a34a,color:#111
    classDef ours fill:#f3f4f6,stroke:#6b7280,color:#111
    classDef ext fill:#ffedd5,stroke:#ea580c,color:#111
    classDef llm fill:#f3e8ff,stroke:#9333ea,color:#111
    classDef pr fill:#fef9c3,stroke:#ca8a04,color:#111

    subgraph SRC["Outside this machine"]
        FH["Finnhub and Polygon<br/>paid news, press releases, filings, financials"]:::ext
        FHS["Finnhub symbol search /search<br/>the directory a symbol or company name is looked up in<br/>a query of at most 20 characters"]:::pr
        EDGAR["SEC EDGAR sec.gov/Archives<br/>the complete submission of a filing (a 10-Q: 5 MB, 60 documents)<br/>free; a User-Agent with a contact (SEC_USER_AGENT)"]:::pr
        OR["OpenRouter<br/>gpt-oss-120b (Data Clerk)<br/>nemotron-3-super-120b (Advisor)"]:::ext
    end

    subgraph TOOLBOX["agents/data_clerk/data_toolbox.py  DataToolbox: plain Python, no LLM, no loop"]
        DA["fetch_news, fetch_press_releases, fetch_filings<br/>search, discover, financials.get, performance_summary"]:::ours
        DAT["fetch_filing_text (the 10-Q's text)<br/>download the stored url, edgar_text.filing_sections,<br/>APPEND a second record of the same accession"]:::pr
    end

    FILES[("news/SYMBOL.txt<br/>the text files")]:::ours
    PFILES[("data/press_releases/SYMBOL_PRs.txt<br/>press releases: their own file per symbol, appended,<br/>a release with a known id is skipped")]:::pr
    FFILES[("data/filings/SYMBOL.txt<br/>SEC filings: one file per symbol, appended,<br/>a filing with a known accession number is skipped")]:::pr

    subgraph INGEST["rag/  batch ingest, run before any question"]
        LOAD["loaders.load_news<br/>text file to list of Document"]:::lc
        PLOAD["loaders.load_press_releases<br/>a second parser, the same Document metadata keys<br/>ids pr:..., publisher Press release"]:::pr
        CHUNK["ingest.to_chunks<br/>RecursiveCharacterTextSplitter 800 / 100<br/>headline put back in front of each chunk"]:::lc
        ING["ingest.ingest_documents<br/>merge tickers, diff against the store, delete stale chunks"]:::ours
        STORE[("store.get_vector_store<br/>Chroma, cosine HNSW<br/>HuggingFaceEmbeddings all-MiniLM-L6-v2")]:::lc
        PSTORE[("store.get_vector_store(press_releases)<br/>its OWN collection in the same Chroma folder:<br/>news search never sees it")]:::pr
        FLOAD["loaders.load_filings<br/>a third parser: one Document per SECTION of a filing with text,<br/>one short Document for a filing with no text<br/>the last record WITH text of an accession wins<br/>ids sec:..., publisher SEC filing"]:::pr
        FSTORE[("store.get_vector_store(filings)<br/>its OWN collection: filings-...")]:::pr
    end

    FH -->|"fetch"| DA
    EDGAR -->|"10-Q submission"| DAT
    DAT -->|"appends"| FFILES
    DA -->|"inside fetch_filings, step 2"| DAT
    DA -->|"writes"| FILES
    DA -->|"writes"| PFILES
    DA -->|"writes"| FFILES
    FILES --> LOAD --> CHUNK --> ING -->|"add_documents with ids: embed then upsert"| STORE
    PFILES --> PLOAD --> CHUNK
    ING -->|"kind press_releases"| PSTORE
    FFILES --> FLOAD --> CHUNK
    ING -->|"kind filings"| FSTORE

    subgraph SEARCH["rag/retriever.py"]
        SN["search_news(query, tickers, since_days, k, diverse)<br/>similarity_search_with_score(filter)<br/>score = 1 minus distance, bar 0.40<br/>one hit per article, optional MMR"]:::lc
        SP["search_press_releases(query, ...)<br/>search_news on the press release store"]:::pr
        SF["search_filings(query, ...)<br/>search_news on the filings store<br/>one hit per filing section"]:::pr
        HIT["Hit: chunk id, headline, publisher, date, url, tickers, text, score"]:::ours
    end
    STORE <-->|"filter by ticker and date"| SN --> HIT
    PSTORE <-->|"filter by ticker and date"| SP --> HIT
    FSTORE <-->|"filter by ticker and date"| SF --> HIT

    subgraph TOOLS["tools/data_clerk.py  DataClerkTools: one implementation, two doors"]
        DCT["retrieve_news, retrieve_press_releases, retrieve_filings, fetch_news, fetch_press_releases, fetch_filings, fetch_filings_recommendation<br/>keyword_search, discover_tickers, get_financials, get_price_summary<br/>eleven tools; argument checks, DATA_CLERK_TOOLS allow-list<br/>fetch_press_releases and fetch_filings now also index what they fetch"]:::pr
        LCT["langchain_tools(book)<br/>StructuredTool, the three retrieve tools<br/>as content_and_artifact, ONE shared SourceBook"]:::pr
        ERR["describe_tool_error"]:::ours
        FSIG["agents/data_clerk/filing_signals.py<br/>fetch_filings_recommendation: reads the STORED filings (no network):<br/>balance sheet for both periods, capital actions, exact quotes on dilution,<br/>executive compensation, buybacks, dividends, and the gaps. Scores nothing"]:::pr
    end
    HIT --> DCT
    FLOAD -->|"the stored filings"| FSIG --> DCT
    DCT -->|"fetch tools"| DA
    DCT --> LCT

    subgraph MCP["mcp/data_clerk_server.py"]
        FMCP["FastMCP server, stdio<br/>8 tools (7 before the evidence tool, 6 before filings, 5 before press releases), fetch tools only with --allow-fetch"]:::pr
    end
    DCT -->|"same bound methods"| FMCP

    subgraph WO["common/work_order.py"]
        WORD["WorkOrder<br/>verify_request, attach_result, verify_action<br/>data (sources, results per corpus, filing_evidence), recommendations, status, errors<br/>source news, press_releases or filings; company; resolved<br/>action search_news, or search_filing (filings only, checked both ways)"]:::pr
    end

    subgraph SYM["tools/symbols.py  the exact lookup for press releases and filings (no LLM)"]
        RT["resolve_target(symbol, company)<br/>the symbol first; if not found, the company name<br/>find_symbol, find_company: exact, same words, punctuation ignored<br/>TODO: fuzzy company-name matching<br/>nothing found: Symbol: X is not found / Company Name not found."]:::pr
    end

    subgraph AGENT["agents/data_clerk/data_clerk_agent.py  DataClerk: the LangGraph agent"]
        DC["DataClerk.request_data(order)<br/>DataClerk.ask(question)"]:::ours
        RES0["_resolve_target(order): press release and filings orders only<br/>runs in code BEFORE the graph and the model:<br/>a refused order costs nothing, nothing is fetched"]:::pr
        BUD["Budget and budgeted tools<br/>3 searches of each kind, fetch off, none after web text<br/>press release and filing text counts as web text too"]:::pr
        GRAPH["StateGraph(State)<br/>nodes agent, tools, verify"]:::lg
        TN["ToolNode(tools, handle_tool_errors)"]:::lg
        SB["rag/citations.py  SourceBook<br/>numbers the sources, check(answer)"]:::ours
        PR1["prompts/data_clerk.py<br/>system prompt with today's date"]:::ours
        RES["Result: answer, references, sources,<br/>problems, notes, tool_calls, tokens"]:::ours
    end

    CM["models/chat.py  get_chat_model<br/>ChatOpenAI via OpenRouter, bind_tools"]:::lc
    LLM1(["LLM: Data Clerk model"]):::llm
    LLM2(["LLM: Advisor model"]):::llm

    subgraph ADV["agents/advisor/advisor.py  Advisor v0"]
        AD["Advisor.advise(order)"]:::ours
        VS["verify_signal: the quote is in the source and names the company<br/>calculate_score from verified signals only"]:::ours
        PR2["prompts/advisor.py"]:::ours
    end

    USER(["Caller: Account Manager, not built yet,<br/>or a lab script"]):::ours

    USER -->|"1 new_work_order, verify_request"| WORD
    WORD -->|"2 request_data(order)"| DC
    DC -->|"2b press release or filings order only"| RES0
    RES0 --> RT
    RT <-->|"look the symbol up, then the name"| FHS
    RES0 -.->|"not found or lookup failed:<br/>data_failed, no model call, no fetch"| WORD
    DC -->|"3 builds per question"| GRAPH
    DC --> SB
    DC --> BUD
    LCT --> BUD --> TN
    GRAPH --- TN
    PR1 --> GRAPH
    GRAPH <-->|"4 messages"| CM <--> LLM1
    CM --- OR
    TN -->|"5 tool calls"| LCT
    ERR -.-> TN
    SB -.->|"numbers the hits"| LCT
    GRAPH -->|"6 final state"| RES
    RES -->|"7 attach_result, verify_action"| WORD
    WORD -->|"8 advise(order)"| AD
    AD --> PR2
    AD <-->|"9 JSON signals"| LLM2
    LLM2 --- OR
    AD --> VS
    VS -->|"10 recommendations with calculated score"| WORD
```

---

## 2. The LangGraph: what runs for one question

This is `DataClerk._graph()` in `data_clerk_agent.py`. It is built fresh for every question, so each question has its own `SourceBook` (so `[1]` means one chunk in that answer) and its own `Budget`.

```mermaid
flowchart LR
    classDef lg fill:#dcfce7,stroke:#16a34a,color:#111
    classDef lc fill:#dbeafe,stroke:#2563eb,color:#111
    classDef ours fill:#f3f4f6,stroke:#6b7280,color:#111
    classDef llm fill:#f3e8ff,stroke:#9333ea,color:#111
    classDef pr fill:#fef9c3,stroke:#ca8a04,color:#111

    S(["START"]):::lg
    E(["END"]):::lg

    subgraph STATE["State = TypedDict"]
        ST["messages: Annotated(list, add_messages)<br/>turns: int<br/>repaired: bool<br/>forced: bool"]:::lg
    end

    A["node agent<br/>SystemMessage(prompt) + messages<br/>model.bind_tools(tools).invoke(...)<br/>last turn: model.invoke without tools<br/>plus HumanMessage(LAST_TURN)"]:::ours
    M(["ChatOpenAI via OpenRouter"]):::llm
    T["node tools<br/>ToolNode(tools, handle_tool_errors=describe_tool_error)<br/>runs every tool_call, returns ToolMessage"]:::lg
    V["node verify<br/>SourceBook.check(answer)"]:::ours

    S --> A
    A <--> M
    A -->|"conditional after_agent:<br/>tool_calls and turns left"| T
    A -->|"conditional after_agent:<br/>an answer, or no turns left"| V
    T -->|"edge"| A
    V -->|"conditional after_verify:<br/>invalid citation and not repaired yet<br/>adds HumanMessage(repair)"| A
    V -->|"conditional after_verify:<br/>valid, or still wrong after one correction"| E

    subgraph TOOLSET["the tools inside the ToolNode (every one wrapped by Budget)"]
        direction TB
        TR["retrieve_news<br/>StructuredTool, content_and_artifact<br/>content = numbered sources, artifact = hits"]:::lc
        TR2["retrieve_press_releases<br/>the same wrapper and the SAME SourceBook:<br/>[3] is one chunk, whichever tool found it"]:::pr
        TR3["retrieve_filings<br/>the same wrapper and the same SourceBook<br/>a source with no text says so"]:::pr
        TO["keyword_search, discover_tickers,<br/>get_financials, get_price_summary"]:::lc
        TF["fetch_news, fetch_press_releases, fetch_filings<br/>OFF unless allow_fetch,<br/>refused once web text was read<br/>(news, press release and filing text all count)"]:::lc
    end
    T --- TOOLSET

    PRE["before the graph, press release and filings orders only:<br/>_resolve_target in code: symbol first, then company name<br/>nothing found: the order is refused, the graph is never built"]:::pr
    PRE -.-> S
```

| Piece | Real code | What it does |
|---|---|---|
| `StateGraph(State)` | `langgraph.graph` | the loop: `add_node`, `add_edge`, `add_conditional_edges`, `compile()` |
| `add_messages` | `langgraph.graph.message` | the reducer that appends each new message to `state["messages"]` |
| `ToolNode` | `langgraph.prebuilt` | runs the tool calls of the last `AIMessage`; with `handle_tool_errors=describe_tool_error` a failing or refused tool becomes a one-line `ToolMessage` and the loop goes on |
| `bind_tools` | `ChatOpenAI` (LangChain) | gives the model the eleven tool schemas (ten before the evidence tool, nine before filings, eight before press releases) |
| `StructuredTool` | `langchain_core.tools` | each tool, built from a method's type hints and docstring (`tool(method, parse_docstring=True)`), and the three retrieval tools, `retrieve_news`, `retrieve_press_releases` and `retrieve_filings`, with `response_format="content_and_artifact"` and one shared `SourceBook` |
| `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage` | `langchain_core.messages` | the conversation the graph passes around |

---

## 3. One question, step by step (sequence)

"What did MKBHD say about Apple's foldable iPhone?" through `DataClerk.request_data`, with the real tool calls and messages.

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant DC as DataClerk (data_clerk_agent.py)
    participant G as StateGraph (LangGraph)
    participant M as ChatOpenAI + bind_tools
    participant TN as ToolNode (LangGraph)
    participant B as Budget
    participant T as DataClerkTools.retrieve_news
    participant R as search_news (retriever.py)
    participant CH as Chroma + MiniLM
    participant SB as SourceBook
    participant WO as WorkOrder

    C->>DC: request_data(order)
    DC->>DC: verify_request(order), build the question from the order
    Note over DC: Week 4, press release and filings orders only: _resolve_target(order) runs here, in code, before any model call.<br/>The symbol first, else the company name (symbols.resolve_target). Not found: data_failed, the rest of this diagram never happens.
    DC->>DC: new SourceBook and Budget, wrap the 11 tools, bind_tools
    DC->>G: invoke(messages=[HumanMessage], turns=0)

    G->>M: agent node: SystemMessage(prompt, today) + messages
    M-->>G: AIMessage(tool_calls=[retrieve_news(query, tickers=AAPL, k=5)])
    G->>TN: after_agent routes to tools
    TN->>B: check("retrieve_news", args): count the call, fetch rules
    TN->>T: retrieve_news(**args), under the shared search lock
    T->>T: argument checks: ticker, days, k
    T->>R: search_news(query, tickers, since_days, k)
    R->>CH: similarity_search_with_score(query, filter=ticker and date)
    CH-->>R: (Document, cosine distance) for the closest chunks
    R-->>T: list of Hit: score 1 minus distance at least 0.40, one per article
    T->>SB: add(hits): number them from 1, defuse "[3]" inside text
    SB-->>T: the numbered text
    T-->>TN: content = numbered sources, artifact = hits with numbers
    TN->>B: observe(output): web text has now been read
    TN-->>G: ToolMessage(content, artifact)

    G->>M: agent node again (turn 2) with the sources in the messages
    M-->>G: AIMessage("MKBHD gave it a blunt verdict ... [2]")
    G->>G: after_agent routes to verify
    G->>SB: verify node: check(answer)
    alt a cited number matches no source, first time
        G->>M: HumanMessage(repair): cite only 1 to n, then loop to agent
    else valid, or already corrected once
        SB-->>G: CitationCheck: cited chunks, invalid numbers
    end
    G-->>DC: final state: messages, turns, repaired, forced

    DC->>DC: _fill: Result with answer, references, sources, problems, notes
    DC->>WO: attach_result(order, result), verify_action(order)
    WO-->>C: the same order: data, status data_ready or data_failed
```

---

## 4. The ingest pipeline (LangChain components)

The part that makes retrieval possible. It runs before any question, and `fetch_news` runs the tail of it (`ingest_symbol`) so new articles are searchable at once. Week 4 added a second source (yellow) and then a third, SEC filings (also yellow, bottom of the diagram: `fetch_filings` writes its own file, `load_filings` makes one `Document` per section of a filing that has text and one short `Document` for each filing that has none, the same chunker and the same diff-and-upsert code run on them, and `ingest_filings` stores the result in the `filings-...` collection). The 10-Q has no text in the paid API, so `fetch_filing_text` takes it from SEC EDGAR and appends a second record of the same accession; `load_filings` reads the last record with text (entry 36, the two boxes next to the filings file). The press release part is: `fetch_press_releases` writes its own file, `load_press_releases` makes the same kind of `Document`, the **same** chunker and the same diff-and-upsert code run on it, and `ingest_press_releases` stores the result in its own collection, so a news search never sees a press release and the news collection is unchanged.

```mermaid
flowchart LR
    classDef lc fill:#dbeafe,stroke:#2563eb,color:#111
    classDef ours fill:#f3f4f6,stroke:#6b7280,color:#111
    classDef ext fill:#ffedd5,stroke:#ea580c,color:#111
    classDef pr fill:#fef9c3,stroke:#ca8a04,color:#111

    API["Finnhub, Polygon"]:::ext
    DAF["DataToolbox.fetch_news<br/>skips duplicates"]:::ours
    TXT[("news/SYMBOL.txt")]:::ours
    LN["load_news(path)<br/>check_symbol, clean text, headline,<br/>publisher, tickers, published time"]:::lc

    PAPI["Finnhub /press-releases"]:::ext
    PDAF["DataToolbox.fetch_press_releases<br/>skips a release whose id is already in the file"]:::pr
    PTXT[("data/press_releases/SYMBOL_PRs.txt")]:::pr
    PLN["load_press_releases(path)<br/>a separate parser: the record is PRESS_RELEASE, TEXT, END<br/>ids pr:..., publisher Press release, tickers [SYMBOL]"]:::pr
    DOC["Document<br/>id = article id<br/>metadata: headline, tickers, publisher,<br/>published_utc, published_ts, url, api"]:::lc
    MERGE["same article in two ticker files:<br/>keep one text, merge the tickers"]:::ours
    SPLIT["RecursiveCharacterTextSplitter<br/>chunk_size 800, overlap 100, add_start_index"]:::lc
    CH["chunk Document<br/>id = article id # n<br/>headline put back in front if missing"]:::lc
    DIFF["store.get(where article_id in ...)<br/>unchanged, updated, added, stale deleted"]:::ours
    EMB["HuggingFaceEmbeddings<br/>all-MiniLM-L6-v2, 384 numbers, CPU"]:::lc
    CHR[("Chroma collection<br/>news-sentence-transformers-all-MiniLM-L6-v2<br/>cosine, HNSW ef_construction 400,<br/>ef_search 300, max_neighbors 32")]:::lc
    PCHR[("Chroma collection<br/>press_releases-sentence-transformers-all-MiniLM-L6-v2<br/>the same settings, its own set of chunks")]:::pr

    FAPI["Massive (Polygon) filings endpoints<br/>/vX/index, /10-K/vX/sections, /8-K/vX/text"]:::ext
    FDAF["DataToolbox.fetch_filings<br/>skips a filing whose accession number is already in the file"]:::pr
    EAPI["SEC EDGAR: the complete submission of a 10-Q<br/>(the url stored in the record)"]:::ext
    EDAF["DataToolbox.fetch_filing_text<br/>edgar_text: the 10-Q document, HTML to lines,<br/>split into 11 Items; appends a SECOND record, text_status ok"]:::pr
    FTXT[("data/filings/SYMBOL.txt")]:::pr
    FLN["load_filings(path)<br/>a third parser: the record is FILING, TEXT, END<br/>one Document per == section == (10-K: business, risk_factors; 8-K: items)<br/>one short Document that says so for a filing with no text<br/>ids sec:..., publisher SEC filing, filed date at 00:00:00Z"]:::pr
    FCHR[("Chroma collection<br/>filings-sentence-transformers-all-MiniLM-L6-v2<br/>the same settings, its own set of chunks")]:::pr

    API --> DAF --> TXT --> LN --> DOC --> MERGE --> SPLIT --> CH --> DIFF
    DIFF -->|"add_documents(ids)"| EMB --> CHR
    DIFF -->|"delete(ids)"| CHR
    PAPI --> PDAF --> PTXT --> PLN --> DOC
    EMB -->|"ingest_press_releases: kind press_releases"| PCHR
    DIFF -->|"delete(ids), press releases"| PCHR
    FAPI --> FDAF --> FTXT --> FLN --> DOC
    FTXT -->|"10-Qs with no text"| EDAF
    EAPI --> EDAF -->|"appends"| FTXT
    EMB -->|"ingest_filings: kind filings"| FCHR
    DIFF -->|"delete(ids), filings"| FCHR
```

---

## 5. The classes and how they connect

```mermaid
classDiagram
    direction LR

    class DataClerk {
        +ask(question) Result
        +request_data(order) WorkOrder
        -_graph(book, budget, today)
        -_resolve_target(order) bool
        model
        tools
        allow_fetch
        max_turns
        symbol_search
    }
    class Budget {
        +check(name, args)
        +observe(name, output)
        limits
        allow_fetch
        read_web_text
    }
    class Result {
        answer
        references
        sources
        problems
        notes
        tool_calls
        no_supporting_text
        ok
    }
    class DataClerkTools {
        +retrieve_news(query, tickers, since_days, k, diverse)
        +retrieve_press_releases(query, tickers, since_days, k, newest_first)
        +retrieve_filings(query, tickers, since_days, k, newest_first)
        +fetch_news(symbol)
        +fetch_press_releases(symbol)
        +fetch_filings(symbol)
        +keyword_search(symbol, terms)
        +discover_tickers(phrase)
        +get_financials(symbol)
        +get_price_summary(symbol)
        +langchain_tools(book)
    }
    class SourceBook {
        +add(hits) str
        +check(answer) CitationCheck
        +number_of(hit)
    }
    class CitationCheck {
        cited
        invalid
        uncited
        references()
    }
    class Hit {
        chunk_id
        headline
        publisher
        published_utc
        url
        tickers
        text
        score
    }
    class DataToolbox {
        +fetch_news()
        +fetch_press_releases()
        +fetch_filings()
        +fetch_filing_text()
        +search()
        +discover()
        +financials
        +performance_summary()
    }
    class WorkOrder {
        request_id
        query
        action_type
        source
        subject
        company
        resolved
        status
        data
        recommendations
        advisor_runs
        errors
    }
    class DataResult {
        answer
        references
        sources
        problems
    }
    class Advisor {
        +advise(order, use_evidence, bare) WorkOrder
        model
    }
    class DataClerkServer {
        +build_server(tools, allow_fetch)
        +served_tools(allow_fetch)
    }
    class Symbols {
        <<week4>>
        +find_symbol(symbol) Resolution
        +find_company(name) Resolution
        +resolve_target(symbol, company) Resolution
    }
    class TargetNotFound {
        <<week4>>
        message
        detail
    }

    class StateGraph {
        <<LangGraph>>
        add_node()
        add_conditional_edges()
        compile()
    }
    class ToolNode {
        <<LangGraph>>
        handle_tool_errors
    }
    class StructuredTool {
        <<LangChain>>
        name
        args_schema
        response_format
    }
    class ChatOpenAI {
        <<LangChain>>
        bind_tools()
        invoke()
    }
    class Chroma {
        <<LangChain>>
        similarity_search_with_score()
        add_documents()
        get()
    }
    class HuggingFaceEmbeddings {
        <<LangChain>>
        embed_documents()
        embed_query()
    }
    class Document {
        <<LangChain>>
        id
        page_content
        metadata
    }
    class FastMCP {
        <<FastMCP>>
        tool()
        run()
    }

    DataClerk --> StateGraph : builds one per question
    DataClerk --> ChatOpenAI : model.bind_tools
    DataClerk --> Budget : one per question
    DataClerk --> SourceBook : one per question
    DataClerk --> DataClerkTools : tools
    DataClerk --> Result : returns
    DataClerk --> WorkOrder : request_data fills
    StateGraph --> ToolNode : tools node
    ToolNode --> StructuredTool : runs
    StructuredTool --> Budget : wrapped by
    StructuredTool --> DataClerkTools : calls the method
    DataClerkTools --> Hit : search_news returns
    DataClerkTools --> SourceBook : the three retrieve tools add hits
    DataClerkTools --> DataToolbox : fetch and keyword tools
    DataClerkTools --> Chroma : through search_news, search_press_releases and search_filings, three collections
    DataClerk --> Symbols : press release and filings orders, before the model
    Symbols ..> TargetNotFound : raises
    Chroma --> HuggingFaceEmbeddings : embeds
    Chroma --> Document : stores chunks
    SourceBook --> CitationCheck : check returns
    WorkOrder --> DataResult : data
    Result ..> DataResult : attach_result copies
    Advisor --> WorkOrder : reads data.sources, appends recommendations
    Advisor --> ChatOpenAI : advisor role model
    DataClerkServer --> FastMCP : build_server
    FastMCP --> DataClerkTools : serves the same methods
```

---

## 6. The Advisor and the comparison the week 3 submission asks for

`labs/module3_rag/advisor_with_without.py`. The Advisor never writes the score: the model proposes signals (exact quotes), code verifies them against the retrieved chunks, and code calculates the score from the signals that pass.

```mermaid
flowchart TB
    classDef lc fill:#dbeafe,stroke:#2563eb,color:#111
    classDef ours fill:#f3f4f6,stroke:#6b7280,color:#111
    classDef llm fill:#f3e8ff,stroke:#9333ea,color:#111
    classDef bad fill:#fee2e2,stroke:#dc2626,color:#111
    classDef good fill:#dcfce7,stroke:#16a34a,color:#111

    Q["Case 2 work order<br/>AI memory shortage. Which companies are affected?"]:::ours

    subgraph A["A  prompt only"]
        A1["Advisor.advise(order, use_evidence=False, bare=True)<br/>recommend as well as you can, score 0 to 100"]:::ours
        A2["23 recommendations over 12 companies<br/>0 signals, 0 scores that can be calculated<br/>the model's own score moves from run to run"]:::bad
    end
    subgraph B["B  designed Advisor, no sources"]
        B1["Advisor.advise(order, use_evidence=False)<br/>quotes required, never invent"]:::ours
        B2["0 recommendations in 5 of 5 runs<br/>by construction: nothing to verify"]:::ours
    end
    subgraph C["C  designed Advisor, with retrieval"]
        C1["DataClerk.request_data(order)<br/>the LangGraph of section 2"]:::ours
        C2["order.data.sources: every chunk, with text"]:::ours
        C3["Advisor.advise(order)<br/>reads the chunks, not the Data Clerk's answer"]:::ours
        C4["verify_signal per signal<br/>1 the source exists  2 direction ok<br/>3 quote is in that source  4 quote names the company"]:::ours
        C5["calculate_score<br/>round(100 x (bullish + 1) / (signals + 2))<br/>none if no signal passed"]:::ours
        C6["25 of 26 signals passed<br/>MU 67 and NVDA 60 in every run<br/>the model's own score moved 55 to 78"]:::good
    end
    LLMX(["Advisor LLM: nemotron-3-super-120b"]):::llm

    Q --> A1 --> A2
    Q --> B1 --> B2
    Q --> C1 --> C2 --> C3 --> C4 --> C5 --> C6
    A1 <--> LLMX
    B1 <--> LLMX
    C3 <--> LLMX
```

---

## 7. The desk and the main script (Part 9): the path a request takes

`./test_prometheus.sh` prompts for four parameters (and, for press releases and filings only, a company name) and hands them to the desk: a LangGraph whose state is **one work order**, passed from the Account Manager to the Data Clerk to the Advisor. The hand-drawn flow shows how the parameters become the order and where each node goes into the layers of section 1:

```mermaid
flowchart TB
    classDef lc fill:#dbeafe,stroke:#2563eb,color:#111
    classDef lg fill:#dcfce7,stroke:#16a34a,color:#111
    classDef ours fill:#f3f4f6,stroke:#6b7280,color:#111
    classDef ext fill:#ffedd5,stroke:#ea580c,color:#111
    classDef stop fill:#fee2e2,stroke:#dc2626,color:#111
    classDef pr fill:#fef9c3,stroke:#ca8a04,color:#111

    USER(["user at the terminal"]):::ours
    SH["test_prometheus.sh<br/>prompts: symbol, term, source (news, press_releases or filings),<br/>company name (press releases and filings only), from_days_ago (7, or 365)<br/>a symbol or company asks [Y/n] before the paid fetch<br/>no answer means no fetch"]:::pr
    CLI["python -m prometheus.langgraph.desk<br/>run_desk(request)"]:::ours

    subgraph G["StateGraph(DeskState): the state is the work order"]
        AM["account_manager<br/>order_from_request: symbol, term, source, company, days become the order<br/>verify_request: source wired, window 1 to 90 (news) or 730 (press releases, filings)<br/>a press release or filings symbol is not pattern-checked here: the Data Clerk looks it up"]:::lg
        DCN["data_clerk<br/>DataClerk.request_data(order)<br/>press releases and filings: resolve_target first, in code"]:::lg
        ADN["advisor<br/>Advisor.advise(order)"]:::lg
    end

    REF1["refused: errors on the request<br/>(no symbol and no term, a bad symbol, an unwired source,<br/>a press release or filings request with no symbol and no company)"]:::pr
    REF2["stopped: the order is data_failed<br/>(an invalid citation, or answered with no search,<br/>or a press release or filings symbol or company name not found:<br/>Symbol: X is not found / Company Name not found., in 0.2 s)"]:::pr
    DONE(["order status recommended<br/>report, timings, data/processed/desk_symbol_term.json"]):::ours

    USER --> SH --> CLI --> AM
    AM -->|"valid"| DCN
    AM -->|"not valid"| REF1
    DCN -->|"data_ready"| ADN
    DCN -->|"not ready"| REF2
    ADN --> DONE

    subgraph LAYERS["what the nodes use (section 1)"]
        LG1["the DataClerk's own LangGraph: agent, tools, verify<br/>retrieve_news, retrieve_press_releases, retrieve_filings,<br/>fetch_news, fetch_press_releases, fetch_filings (only for a live order)"]:::pr
        LOOK["tools/symbols.py: resolve_target<br/>Finnhub symbol search: the symbol first, then the company name, exact"]:::pr
        FETCH["DataToolbox.fetch_news(symbol, from_date)<br/>Finnhub and Polygon, writes news/SYMBOL.txt, then ingest_symbol"]:::ext
        PFETCH["DataToolbox.fetch_press_releases(symbol, from_date)<br/>Finnhub /press-releases, writes SYMBOL_PRs.txt, then ingest_press_releases"]:::pr
        CHR[("Chroma index<br/>search_news: filters, bar 0.40, newest_first")]:::lc
        PCHR[("Chroma press_releases collection<br/>search_press_releases: the same search, its own chunks")]:::pr
        FFETCH["DataToolbox.fetch_filings(symbol, from_date)<br/>Massive filings API (paid Polygon key), writes SYMBOL.txt, then ingest_filings"]:::pr
        FCHR[("Chroma filings collection<br/>search_filings: the same search, its own chunks,<br/>one hit per filing section")]:::pr
        SIG["verify_signal per quote, a story counts once per direction<br/>calculate_score"]:::ours
    end
    DCN -->|"press releases or filings"| LOOK
    DCN --> LG1
    LG1 -->|"live order only"| FETCH --> CHR
    LG1 -->|"live order only"| PFETCH --> PCHR
    LG1 -->|"live order only"| FFETCH --> FCHR
    LG1 --> CHR
    LG1 --> PCHR
    LG1 --> FCHR
    ADN --> SIG
```

| Parameter | Becomes |
|---|---|
| symbol | `subject` `[SYMBOL]`; the order is live and the Data Clerk may fetch |
| term, no symbol | `subject` and `query` are the term; not live; the stored news is searched |
| term and symbol | `query` is "term news for SYMBOL"; live; the fetch is for the whole symbol |
| source | `source`; `news` (the default), `press_releases` or `filings` pass `verify_request` (Week 4: it was `news` only) |
| from_days_ago | `since_days`: the fetch's `from_date` and the search window. News 1 to 90 (default 7); press releases and filings 1 to 730 (default 365) |
| company (Week 4, `press_releases` and `filings` only) | `company`: used when the symbol is not given or not found. The symbol is looked up first and wins; both lookups are exact; nothing found is `data_failed` with the message |

The compiled graph as LangGraph itself prints it (`python -m prometheus.langgraph.desk --show-graph`); dotted arrows are the conditional edges above. **Neither press releases nor filings changed the shape of this graph:** the same three nodes and the same edges. A refused press release or filings order simply ends after `data_clerk`, because the edge to `advisor` is only taken when the order is `data_ready`.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	account_manager(account_manager)
	data_clerk(data_clerk)
	advisor(advisor)
	__end__([<p>__end__</p>]):::last
	__start__ --> account_manager;
	account_manager -.-> __end__;
	account_manager -.-> data_clerk;
	data_clerk -.-> __end__;
	data_clerk -.-> advisor;
	advisor --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

