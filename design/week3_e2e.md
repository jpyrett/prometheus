# Week 3 end to end: how the DataAgent works with RAG

Jacob Pyrett, CMU Agentic AI, Oct 2026. Companion to `design/week3_submission.md` (the answers) and `labs/module3_rag/hpw_to_do_langchain_rag.md` (the walk-through).

Every box below is a real file, class or function in this repository. Nothing is a plan.

**How to read the colors**

| Color | Meaning |
|---|---|
| blue | a LangChain class or function (`langchain_core`, `langchain_chroma`, `langchain_huggingface`, `langchain_text_splitters`, `langchain_openai`) |
| green | a LangGraph class or function (`langgraph.graph`, `langgraph.prebuilt`) |
| grey | our code: plain Python in `prometheus/` |
| orange | something outside this machine (paid APIs, OpenRouter) |
| purple | an LLM |

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

    subgraph SRC["Outside this machine"]
        FH["Finnhub and Polygon<br/>paid news, filings, financials"]:::ext
        OR["OpenRouter<br/>gpt-oss-120b (Data Clerk)<br/>nemotron-3-super-120b (Advisor)"]:::ext
    end

    subgraph TOOLBOX["agents/data_clerk/data_agent.py  DataAgent: plain Python, no LLM, no loop"]
        DA["fetch_news, fetch_press_releases, fetch_filings<br/>search, discover, financials.get, performance_summary"]:::ours
    end

    FILES[("news/SYMBOL.txt<br/>the text files")]:::ours

    subgraph INGEST["rag/  batch ingest, run before any question"]
        LOAD["loaders.load_news<br/>text file to list of Document"]:::lc
        CHUNK["ingest.to_chunks<br/>RecursiveCharacterTextSplitter 800 / 100<br/>headline put back in front of each chunk"]:::lc
        ING["ingest.ingest_documents<br/>merge tickers, diff against the store, delete stale chunks"]:::ours
        STORE[("store.get_vector_store<br/>Chroma, cosine HNSW<br/>HuggingFaceEmbeddings all-MiniLM-L6-v2")]:::lc
    end

    FH -->|"fetch"| DA
    DA -->|"writes"| FILES
    FILES --> LOAD --> CHUNK --> ING -->|"add_documents with ids: embed then upsert"| STORE

    subgraph SEARCH["rag/retriever.py"]
        SN["search_news(query, tickers, since_days, k, diverse)<br/>similarity_search_with_score(filter)<br/>score = 1 minus distance, bar 0.40<br/>one hit per article, optional MMR"]:::lc
        HIT["Hit: chunk id, headline, publisher, date, url, tickers, text, score"]:::ours
    end
    STORE <-->|"filter by ticker and date"| SN --> HIT

    subgraph TOOLS["tools/data_clerk.py  DataClerkTools: one implementation, two doors"]
        DCT["retrieve_news, fetch_news, fetch_press_releases, fetch_filings<br/>keyword_search, discover_tickers, get_financials, get_price_summary<br/>argument checks, DATA_CLERK_TOOLS allow-list"]:::ours
        LCT["langchain_tools(book)<br/>StructuredTool, retrieve_news as content_and_artifact"]:::lc
        ERR["describe_tool_error"]:::ours
    end
    HIT --> DCT
    DCT -->|"fetch tools"| DA
    DCT --> LCT

    subgraph MCP["mcp/data_clerk_server.py"]
        FMCP["FastMCP server, stdio<br/>5 tools, fetch tools only with --allow-fetch"]:::ours
    end
    DCT -->|"same bound methods"| FMCP

    subgraph WO["common/work_order.py"]
        WORD["WorkOrder<br/>verify_request, attach_result, verify_action<br/>data, recommendations, status, errors"]:::ours
    end

    subgraph AGENT["agents/data_clerk/agent.py  DataClerk: the LangGraph agent"]
        DC["DataClerk.request_data(order)<br/>DataClerk.ask(question)"]:::ours
        BUD["Budget and budgeted tools<br/>3 searches, fetch off, none after web text"]:::ours
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

This is `DataClerk._graph()` in `agent.py`. It is built fresh for every question, so each question has its own `SourceBook` (so `[1]` means one chunk in that answer) and its own `Budget`.

```mermaid
flowchart LR
    classDef lg fill:#dcfce7,stroke:#16a34a,color:#111
    classDef lc fill:#dbeafe,stroke:#2563eb,color:#111
    classDef ours fill:#f3f4f6,stroke:#6b7280,color:#111
    classDef llm fill:#f3e8ff,stroke:#9333ea,color:#111

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
        TO["keyword_search, discover_tickers,<br/>get_financials, get_price_summary"]:::lc
        TF["fetch_news, fetch_press_releases, fetch_filings<br/>OFF unless allow_fetch,<br/>refused once web text was read"]:::lc
    end
    T --- TOOLSET
```

| Piece | Real code | What it does |
|---|---|---|
| `StateGraph(State)` | `langgraph.graph` | the loop: `add_node`, `add_edge`, `add_conditional_edges`, `compile()` |
| `add_messages` | `langgraph.graph.message` | the reducer that appends each new message to `state["messages"]` |
| `ToolNode` | `langgraph.prebuilt` | runs the tool calls of the last `AIMessage`; with `handle_tool_errors=describe_tool_error` a failing or refused tool becomes a one-line `ToolMessage` and the loop goes on |
| `bind_tools` | `ChatOpenAI` (LangChain) | gives the model the eight tool schemas |
| `StructuredTool` | `langchain_core.tools` | each tool, built from a method's type hints and docstring (`tool(method, parse_docstring=True)`), and `retrieve_news` with `response_format="content_and_artifact"` |
| `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage` | `langchain_core.messages` | the conversation the graph passes around |

---

## 3. One question, step by step (sequence)

"What did MKBHD say about Apple's foldable iPhone?" through `DataClerk.request_data`, with the real tool calls and messages.

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant DC as DataClerk (agent.py)
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
    DC->>DC: new SourceBook and Budget, wrap the 8 tools, bind_tools
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

The part that makes retrieval possible. It runs before any question, and `fetch_news` runs the tail of it (`ingest_symbol`) so new articles are searchable at once.

```mermaid
flowchart LR
    classDef lc fill:#dbeafe,stroke:#2563eb,color:#111
    classDef ours fill:#f3f4f6,stroke:#6b7280,color:#111
    classDef ext fill:#ffedd5,stroke:#ea580c,color:#111

    API["Finnhub, Polygon"]:::ext
    DAF["DataAgent.fetch_news<br/>skips duplicates"]:::ours
    TXT[("news/SYMBOL.txt")]:::ours
    LN["load_news(path)<br/>check_symbol, clean text, headline,<br/>publisher, tickers, published time"]:::lc
    DOC["Document<br/>id = article id<br/>metadata: headline, tickers, publisher,<br/>published_utc, published_ts, url, api"]:::lc
    MERGE["same article in two ticker files:<br/>keep one text, merge the tickers"]:::ours
    SPLIT["RecursiveCharacterTextSplitter<br/>chunk_size 800, overlap 100, add_start_index"]:::lc
    CH["chunk Document<br/>id = article id # n<br/>headline put back in front if missing"]:::lc
    DIFF["store.get(where article_id in ...)<br/>unchanged, updated, added, stale deleted"]:::ours
    EMB["HuggingFaceEmbeddings<br/>all-MiniLM-L6-v2, 384 numbers, CPU"]:::lc
    CHR[("Chroma collection<br/>news-sentence-transformers-all-MiniLM-L6-v2<br/>cosine, HNSW ef_construction 400,<br/>ef_search 300, max_neighbors 32")]:::lc

    API --> DAF --> TXT --> LN --> DOC --> MERGE --> SPLIT --> CH --> DIFF
    DIFF -->|"add_documents(ids)"| EMB --> CHR
    DIFF -->|"delete(ids)"| CHR
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
        model
        tools
        allow_fetch
        max_turns
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
        +fetch_news(symbol)
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
    class DataAgent {
        +fetch_news()
        +fetch_press_releases()
        +fetch_filings()
        +search()
        +discover()
        +financials
        +performance_summary()
    }
    class WorkOrder {
        request_id
        query
        action_type
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
    DataClerkTools --> SourceBook : retrieve_news adds hits
    DataClerkTools --> DataAgent : fetch and keyword tools
    DataClerkTools --> Chroma : through search_news
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
