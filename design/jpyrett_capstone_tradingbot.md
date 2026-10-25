# Capstone: Self-Learning Trading Bot ("Prometheus")

## Jacob Pyrett , Oct 2026 - CMU Agentic AI Course, Submission #2

## Type

*Forecaster/Research Agent*      

This has the combined roles:

- **forecasting:**  analyzing recommendations which stocks are good (or bad) opportunities for trading 
- **research:**  based on general trends ("AI memory shortage", "energy companies", "health / medical: weight loss pills" etc.)
- **research:**  Find "What's Happening ?" in the market and send this as an input for forecasting & ranking

## Overview / Summary

My capstone project:  self-learning stock trading bot code-named "Prometheus".  This system will be composed of several agents (multi-agent)  with different roles:

[ mimics: Stock Trading Desk]  

- **Data Clerk (retrieval / gatherer, calls tools) agent**:   Retrieves -> Filters -> Aggregates stock numeric and textual news, press releases, filings, historical data (for financial metrics). Passes info and/or notifications to Advisor, summarization & sentiment analysis for News & filings 
- **Advisor** **(calls tools, memory)** **agent:** Provides recommendations of stock trading opportunities to be critiqued
- **Critic (reasoning / evaluator / ranker)  agent**:  Checks recommendations from Advisor. Reasons which stocks are the best candidates from received recommendations. Denies/Approves recommended candidates.   
- **Account Manager (reasoning, memory, calls tools) agent:**  Manages actual trading details.  Decides which stocks are available to sell (T+1 holding period), cash available/settled management, approves trades / manages trader. Keeps track of all positions. 
- **Trader (calls tools, memory, custom rules) agent:**  Fills orders. Runs backtesting / paper trading simulation.  Future: Hookup to Alpaca API for real trading via configurable on/off switch, reports to Account Manager

## Use Cases

**Case 1: Trade with these provided funds (money)**

- **User:** Here is: X dollars (i.e. $1000) - take it and buy some stocks (simulated).

**Case 2: Trade: Searching  by specific phrases or trends (Search: News / Trends)**

- **User:** Find stock trading opportunities (recommendations) based on these provided specific phrases or trends.
  - "AI memory shortage"
  - "energy companies"
  - "health / medical: weight loss pills"
  - "EV Cars" etc.

**Case 3: Trade based on overall recommendations ("What's Happening?")** for a specific group (S&P 500, NASDAQ, ALL)

- **User:** Find stocks (trading opportunities) based on current stock activity, news, price action / performance

## Real-World Problem & Goals

Automated stock trading ("trading bot") is requested by several real users (friends, family, work colleagues) of my personal stock application.   Literally: "If I gave you $1000, can your app trade for me?"  Or my ex-manager:  "Let me know when it trades and I'll use it".   I have shied away from this sort of project addition due to fear & risk, but now with this course -  Let's be ambitious and use Agentic AI to experiment.  

**Problem:** 

```
1) Can agents research using provided data & provide top-notch stock recommendations using reasoning & be objective?

2) If so, can the agents "self-learn or self-improve" their performance as they gain experience?
```

- Mimics a "trading desk" composed of virtual representations of human roles: Data Clerk, Advisor, Critic, Manager, Trader  
- Ingests text-based data sources (via RAG) from several data feeds. Some commercial (paid) files + API-based and government public filings 
- Reason based-on dynamic data-driven inputs:  stock trading opportunities / candidate
- Act: Critique which stock recommendations are worthwhile and optimal to trade. Pass this list to be traded by AccountManager + Trader 
- Self-learning using memory (short & long-term) to improve earning performance
- Tools:  leverages external helper tools (see Environment -> Tools list below)

## Intended Users

Experimental: Non-commercial  / hobbyist   

1. Myself & a few friends/family  (learn Agentic AI, get retraining & experience, complete Capstone, make some money?) I will be Human-in-the-Loop (HITL) initially.
2. Beginning to intermediate traders after incorporating into my main existing stock application  (other people's money with use-at-own-risk)
3. Stock & AI enthusiasts: Interesting to see what unsupervised results (recommendations) it finds
4. Single user initially with the user cases: Case 1 ($1000), Case 2: "AI stocks",  Case 3:  "What's Happening?"
5. Several concurrent users:  2, 5, 10 . How does system scale?
6. Agents are users. Do they improve over time? Earning totals from back testing better?

## Why a Simple Prompt or Standalone LLM Will Fail

- Scale of data / context window: would not scale for more than a single concurrent user   
- several cross cutting concerns happening simultaneously:  could hallucinate, mix intermediate data, be too brittle  
- Cost, latency, performance would need to run very expensive frontier model.  Would cost too much to gather recommendations.  
- Too many tokens used in one pass. Too much data would need to be passed into the LLM. 
- Feedback loop would be missing / too hard  to do on a "one-shot"
- Memory : short-term and long-term would be ineffective with a "one-shot"    
- Single prompt would use too many tokens just for the prompt.  
- Single prompt calls too many tools and handle too many intermediate results    one-time prompt
- Needs checks & balances using reasoning for: quality, production stability,  trading atomicity, account balance integrity 
- Performance / parallelization:  would be too slow doing one full processing loop  at a time (gather data, produce recommendations, save to memory, rank, review, trade)

## Environment

These are the required subcomponents (as of week 1) based on current learnings:

### Data

My existing stock application has already paid data feeds: 

- Canonical model ("stock recommendation") via Structured JSON to store the stock metadata and values
- Historical data (flat-files)  
- API-based "Live - 15 minute delayed" market conditions (via data feed: FinnHub, Massive.io)
- API-based News data feed (via data feed: FinnHub, Massive.io)
- API-based Press Releases 
- Government (EDGAR, .gov) sites for financial filings

For capstone presentation, will provide sample rows for explainability. 

### Tools

These tools will use the metadata / file based description to be called.

- Ingestion data pipelines (raw data) & Aggregation
- Summarization tool
- Sentiment analysis tool
- Calculator tool 
- Memory tool : Stores & Retrieves from memory (short & long-term)
- Search / filtering tool
- Reranker tool
- MCP architecture  (FastAPI)
- Embeddings: Vector database  / RAG

### Systems

- Test harness:  using Cursor's agent harness
- Data providers: [Finnhub.io](http://Finnhub.io) ,  [massive.com](http://massive.com)   
- Government public filings
- Local AI:  I have two (2) DGX Spark AI workstations [256 GB VRAM, 40 cores, 12.2K CUDA cores, 8 TB SSD] will run final here
- OpenRouter:  used for Development with my Linux laptop + Cursor IDE.  Then, will move to Local AI for daily executions

### Models

- Different agents will need different models. Advisor & Critic will need stronger reasoning (  mid tier models: Nemotron 3 120B or 550B). The rest should work with faster/cheaper 30B models from Qwen and/or GPT-OSS-120B  or less.
- Leverage OpenRouter (cheap mid-range model) for development, local AI for daily execution

---

## (week 2): Refined agent concept & the task it is designed to complete

Each sub-agent below  is refined by:

- **Account Manager  -  Concept:   Central (primary) agent  which coordinates with all other agents   (orchestrator)**  
  **Task:   Main EntryPoint:  Receives requests from main: Prometheus app / agent harness, routes to correct agents**
    - Reasons:     What asking for? What kind of info (stock, news, etc.) needed? For when (time period)?  For Who (subject)?  
    - Acts:  ReAct  (Reasons:  Receives/Validates/Rejects (notifies user) Request, Executes  [(Updates Request Metadata -> passes to Router),  Waits for Completion],  Observes: Result, Acts:  Trades if accepted or  Returns Recommendations, Info) 
    - Memory/ remembers:  Semantic memory :  SQL Database (User account data, logs),  Chat History:  LangGraph checkpointer memory  + Thread
    - How uses tools:  Parses the user request into the shared JSON work order (case, amount, search terms, time window, who) and hands that JSON to the Data Clerk as an agent call. AccountManager waits for the structured result.

- **Data Clerk -  Concept:    Data Engineer agent / One-Stop shop "Data Garage"  for providing data to be evaluated by other agents**  
  **Task:   Fetch** `<SUBJECT>` **data from** `<SOURCE>` **for** `<TIME PERIOD>` **(no older than** `<TIME_EXPIRY>`**),** `<LIVE>` **if fresh data is required, up to** `<TOP-K>` **results.** `<ACTION_TYPE>` **is search_news, fetch_quote, search_socials, or calculate_financials — not trades.**
    - Reasons:  Does the `<WORK_ORDER>` have the fields needed? Which `<ACTION_TYPE>` is the task? Did `<WORK_ORDER>`.SOURCE return data? If the row counts check out, notify the caller.
    - Acts:  ReAct (verify_request on the `<WORK_ORDER>` → parse_action for `<ACTION_TYPE>` → MCP fetch → verify_action on row counts). Returns the same `<WORK_ORDER>` with completed metadata to the caller.
    - Memory/ remembers:  Short-Term Cache (Redis),  analytics: Files (Parquet),  VectorDB :  Similairity 
    - How uses tools:  Parses that work order, then calls its own MCP (quotes, news, filings, RAG). It returns structured JSON to the Account Manager.

- **Advisor**  **-  Concept:   Generates stock recommendations for the Critic to approve or deny**  
  **Task:   Given a** `<WORK_ORDER>`**, decide whether** `<SUBJECT>` **meets** `<CRITERIA>` **(sentiment: bearish / bullish) and produce a scored trading opportunity**
    - Reasons:  Is `<SUBJECT>` a buy under `<CRITERIA>`? Does the `<WORK_ORDER>` have the fields needed? If yes, ask the Data Clerk for the data, then calculate the recommendation score.
    - Acts:  ReAct (verify_request on the `<WORK_ORDER>` → parse_criteria into WORK_ORDER.criteria → DataClerk.request_data → calculate_score). Appends the result on WORK_ORDER.recommendations and passes that `<WORK_ORDER>` on.
    - Memory/ remembers:  The same `<WORK_ORDER>` as it is updated: criteria first, then the recommendation score in WORK_ORDER.recommendations. That JSON is what the Critic receives.
    - How uses tools:  verify_request and parse_criteria (MCP) read the `<WORK_ORDER>`. DataClerk.request_data is an agent call, not Advisor's own MCP. calculate_score is a local LangChain tool on this agent.

- **Critic  -  Concept:   Ranker who approves or denies the Advisor's recommendations**  
  **Task:   Given a** `<WORK_ORDER>` **with** `<RECOMMENDATION>`**, approve or deny it by checking** `<CRITERIA>`**: is valid, expiration_time, risk_level, is_deal, and** `<OTHER_RECOMMENDATIONS_CANDIDATES>`
    - Reasons:  Should this recommendation be approved or denied? Does the `<WORK_ORDER>` have the fields needed? Do the `<CRITERIA>` still hold against current conditions? On error, notify the user. If approved, the `<WORK_ORDER>` goes to the Trader.
    - Acts:  ReAct (verify_request → parse_criteria into WORK_ORDER.criteria → is_valid, is_expired, is_deal, in_risk_level, find_other_candidates). Each check writes WORK_ORDER.Recommendation.Status. The finished `<WORK_ORDER>` is passed to the Trader.
    - Memory/ remembers:  The same `<WORK_ORDER>` as each check updates WORK_ORDER.Recommendation.Status, plus the IDs in `<OTHER_RECOMMENDATIONS_CANDIDATES>`. That JSON is what the Trader receives.
    - How uses tools:  verify_request is an MCP call. parse_criteria, is_valid, is_expired, is_deal, in_risk_level, and find_other_candidates are local LangChain tools on this agent. The Critic does not call the Data Clerk's MCP.

- **Trader -  Concept:   Performs the actual trades**  
  **Task:   Process the** `<WORK_ORDER>` **trade request: confirm it is valid, check whether that trade is still active, and report Filled or Rejected back to the Account Manager**
    - Reasons:  Is this trade request valid? Has `<WORK_ORDER>` already been started and is it still active? If WORK_ORDER.Trades[Trade_ID] is Filled or Rejected, notify the Account Manager so the user's portfolio and available funds can be updated.
    - Acts:  ReAct (verify_request on the `<WORK_ORDER>` → check_trade_status for Trade_ID → AccountManager.notify). Status lands on WORK_ORDER.Trades[Trade_ID] as Active, Filled, or Rejected.
    - Memory/ remembers:  The same `<WORK_ORDER>` as WORK_ORDER.Trades[Trade_ID] moves from Active to Filled or Rejected. That status is what the Account Manager uses to update the portfolio.
    - How uses tools:  verify_request and check_trade_status run on this agent. AccountManager.notify is a LangGraph agent call, not the Trader's own MCP. The Trader does not call the Data Clerk.

## (week 2): Well-defined reasoning loop:  How agent reasons & acts over multiple steps

 **Here are reasoning/ action/observation loops for the various agents:**

---

**Account Manager  : Main Orchestrator - EntryPoint**

*Reasoning 1:*  "I need to create `<WORK ORDER>` given this `<QUERY>` . The list of fields I need for the `<WORK_ORDER>` is  `<TOP-K>` data  for number of results. The  `<SUBJECT>` from `<SOURCE>`  for `<TIME PERIOD>` that can only be old: `<TIME_EXPIRY>` .   The main action to do is the `<ACTION_TYPE>`.

- `<WORK ORDER>` = (text) structured JSON request with place for metadata (request_id, requester, timestamp, etc) 
- `<SUBJECT>` = (text) stock symbol  or (list of symbols),  term / trend (i.e memory stocks),  group or exchange ("S&P 500, NYSE)
- `<SOURCE>` = historical data, live data, news, filing, press releases , social media feed (x.com)  
- `<TIME_PERIOD>` = (text)parseable calendar date range or string ("December 2026 - now", "last 3 months", YTD"
- `<GROUP>` = (text) the subset of stocks or industry looking for  ("AI stocks",  "NASDAQ", etc.)
- `<LIVE>` = does need to be new uncached / fresh data
- `<TOP_K>` = how many results to return ("give me 5")
- `<ACTION_TYPE>` is:  trade, search_news, fetch_quote, search_socials, calculate_financials

*Action1:*  Call  tool:  parse_input  (`<QUERY>``<TOP-K>`,`<SUBJECT>`, `<SOURCE>`, ...) all the fields in the QUERY  

*Observation 1:*   Receives new JSON object with the mapped input to fields.   This is REQUEST  JSON with some metadata filled. 

*Reasoning 2:* Does the [WORK_ORDER] have all the fields?  Yes, then  create a new WORK_ORDER with the request.  

*Action 2:*   Call tool:  create_work_order(`<REQUEST>`)  passing the JSON in as input .  This will be a local LangChain tool call. 

*Observation2* :  Received the structured JSON result: WORK_ORDER from tool. This is the main object (Unit_of_Work).

*Reasoning 3*:  "Did the WORK_ORDER get created? I'm ready to send it to be processed."   

*Action :*   Call tool:   

*Observation3* :  Received the structured JSON result from tool.   YES/NO

Reasoning 4:  If work is done , then I shall notify the calling requestor.  I can find this in the WORK_ORDER metadata,

(processing continues:  varations for live data, and how to handle groups)  

**Data Clerk :   Fetches data** 

*Reasoning 1:*  "I need to process this `<LIVE>` `<WORK ORDER>` `<TOP-K>` data  for `<SUBJECT>` from`<SOURCE>`  for `<TIME PERIOD>` that can only be old: `<TIME_EXPIRY>` .   The action to do is the `<ACTION_TYPE>` except trades."

- `<WORK ORDER>` = (text) structured JSON request with place for metadata (request_id, requester, timestamp, etc) 
- `<SUBJECT>` = (text) stock symbol  or (list of symbols),  term / trend (i.e memory stocks),  group or exchange ("S&P 500, NYSE)
- `<SOURCE>` = historical data, live data, news, filing, press releases , social media feed (x.com)  
- `<TIME_PERIOD>` = (text)parseable calendar date range or string ("December 2026 - now", "last 3 months", YTD"
- `<GROUP>` = (text) the subset of stocks or industry looking for  ("AI stocks",  "NASDAQ", etc.)
- `<LIVE>` = does need to be new uncached / fresh data
- `<TOP_K>` = how many results to return ("give me 5")

*Action1:*  Call  tool:  verify_request([WORK_ORDER]) 

*Observation 1:*   Receives:  YES / NO / ERROR 

*Reasoning 2:* Does the [WORK_ORDER] have all the fields?  Yes, then  find the `<ACTION_TYPE>` which is the performed task.  

*Action 2:*   Call tool:  parse_action`<ACTION_TYPE>`  using regex.  

The `<ACTION_TYPE>` is:   [search_news(`<SUBJECT>`),
                           fetch_quote(`<SUBJECT>`),
                           search_socials(`<SUBJECT>`),
                           calculate_financials(`<WORK_ORDER>`)]
  
   The metadata and singular result (i.e. quote etc.)  with row counts, etc.         
   This call will be MCP tool call to the data engineering tools for the Data Clerk. 

*Observation2* :  Received the structured JSON result from tool which is the same WORK_ORDER, but with completed metadata 

*Reasoning 3*:  Did receive data from remote: `<WORK_ORDER>`.SOURCE  (news, API, historical data)

*Action 3:*  Call tool:  verify_action( WORK_ORDER). This will check row counts in metadata

*Observation3* :  Received the structured JSON result from tool.   YES/NO

Reasoning 4:  If work is done , then I shall notify the calling requestor.  I can find this in the WORK_ORDER metadata,

(processing continues:  varations for live data, and how to handle groups)  

**Advisor** : Generates recommendations   

*Reasoning 1:*  "I received this: `<WORK_ORDER>`, I need to  find out if  `<SUBJECT>`  IS `<CRITERIA>` in order to generate a profit making stock buying (trading) opportunity."

- `<WORK ORDER>` = (text) structured JSON request with place for metadata (request_id, requester, timestamp, etc) 
- `<SUBJECT>` = (text) stock symbol  or (list of symbols),  term / trend (i.e memory stocks),  group or exchange ("S&P 500, NYSE)
- `<CRITERIA>` = sentiment (positive, negative) (i,.e, good, bad, excellent, horrible)  -> bearish (bad),  bullish (good)

*Action1:*  Call  tool:  verify_request([WORK_ORDER]) 

*Observation 1:*   Receives:  YES / NO / ERROR 

*Reasoning 2:* Does the [WORK_ORDER] have all the fields?  Yes, then  find the `<ACTION_TYPE>` which is the performed task.  

*Action 2:*   Call tool using MCP:  parse_criteria(WORK_ORDER, CRITERIA)  using regex.  

Observation 2:  Received updated WORK_ORDER with WORK_ORDER.criteria JSON field populated  

*Observation2* :  Received the structured JSON result from tool which is the same WORK_ORDER, but with completed metadata 

*Reasoning 3*:  "Now, I need to determine the actual recommendation. I will ask the Data Clerk to request the data." 

*Action 3:*  Call tool:  DataClerk.request_data( WORK_ORDER).

*Observation3* :  Received the structured JSON result from DataClerk tool.   WORK_ORDER will show error or rows calculated etc.

Reasoning 4:  "If the data is ready, I can now calculate the actual recommendation for  WORK_ORDER.  I will calculate now the overall recommendation score."

Action 4:  Call tool:  calculate_score in this local agent using LangChain local tool call

Observation 4:  Recieved same WORK_ORDER, but with new recommended appeneded in WORK_ORDER.recommendations metadata. 

**Critic  :** Ranker who approves and/or denies  the recommendations from the Advisor 

*Reasoning 1:*  "I received this: WORK_ORDER with [WORK_ORDER.Recommnedation],   I need to  find out if  I should approve or deny this recommendation.  The approval `<CRITERIA>` are the decision making conditions to check . "

- `<WORK ORDER>` = (text) structured JSON request with place for list of recommendation metadata (request_id, requester, timestamp, etc)
- `<SUBJECT>` = (text) stock symbol  or (list of symbols),  term / trend (i.e memory stocks),  group or exchange ("S&P 500, NYSE)
- `<RECOMMENDATION>` = JSON fragment with an recommendation.ID, SUBJKECT and the CRITERIA details added 
- `<CRITERIA>` = is valid,  expiration_time (expiry),  risk_level,  is_deal, 
- `<OTHER_RECOMMENDATIONS_CANDIDATES>`  = IDs of other recommendations

*Action1:*  Call  tool: from MCP :  verify_request([WORK_ORDER]) 

*Observation 1:*   Receives:  YES / NO / ERROR 

*Reasoning 2:* Does the [WORK_ORDER] have all the fields?  Yes, then  find the `<ACTION_TYPE>` which is the performed task.  

*Action 2:*   Call tool:  parse_criteria(WORK_ORDER, CRITERIA)  using regex. 

Observation 2:  Received updated WORK_ORDER with WORK_ORDER.criteria JSON field populated  

*Reasoning 3*:  "Now, I need to actually compare the criteria against current conditions" 

*Action 3:*  Call tool using LangChain:  is_valid(WORK_ORDER).Recoomendation[i]

*Observation3* :  Received the structured JSON result from DataClerk tool.   WORK_ORDER will show that CRITERIA._  field

 Next Reasoning / Action/ Observations : continue calls for:  

is_expired(),  is_deal (this checks if the current price is a good buy compared to past purchases),
in_risk_level(),  find_other_candidates() 

with all of these tool calls returning completed: WORK_ORDER.Recommendation.**Status**  updated status. Nofify the user on errors.

Final:  WORK_ORDER ends up at Trader agent

**Trader : Performs actual trades**

*Reasoning 1:*  "I need to process this `<WORK ORDER>` .Action trade request. The action to do is the `<ACTION_TYPE>` except trades."

- `<WORK ORDER>` = (text) structured JSON request with place for metadata (request_id, requester, timestamp, etc)

*Action1:*  Call  tool:  verify_request([WORK_ORDER]) 

*Observation 1:*   Receives:  YES / NO / ERROR 

*Reasoning 2:*  "Has this trade `<WORK ORDER>`  been started and still active?  "

*Action2:*  Call  tool:  check_trade_status([WORK_ORDER], Trade_ID) 

*Observation 2:*   Receives:  WORKORDER.Trades[Trade_ID]  Active / Filled (completed) / Rejected (error) 

*Reasoning 3:*  "If WORK_ORDER.Trades[Trade_ID] completed, I need to notify AccountManager so the  User's portfolio and avaliable funds  can be updated."

*Action3:*  Call  tool:  AccountManager.notify([WORK_ORDER])  uses LangGraph to notify the other Agent that the trade completed/rejected/etc.

Obervation 3: Was AccountManager found and notified?

---

## (week 2) An explicit identification of memory requirements, including whether short-term memory, long-term memory, or both are needed and when they apply

Memory needed:   **Both. HybridMemory** will be used to do text-based SQL searching also ("Control-F") search 

**Short Term**:

 Applied:    

1) When looking up current  quick / just retrieved & cached data to avoid SLA violations for API calls (data providers)
2) Keeping track of the current in-flight processing (Trades, Data gatherings, LLM scoring for recommendations)
3) Similarity search for finding like Trends/Topics

  **Cache**:  Redis  - to store (cache) the computed fields (calculate_financials)  
                   quote data (stock prices) so do not need to fetch within X seconds.  
                   social media:  [x.com](http://x.com) is expensive to read tweets  

  **Vector Database**:  
       Similarity Semantic search for:  Other candidate recommendations.  Added as recommdations are processed  
       RAG Data:  Social Medaia News, Press Releases, Filings  for similarity (no historical data)  
      LangGraph - Checkpointer with Thread memory:  Keep track of current in-progress WORK_ORDERs

**Long Term**:  
         Applied:   

1) User profile & account portfolio persistence & integrity
2) During trading for record ACID locking
3) account portolio managment, 4) error loggging & audit trails. QA.
5) Full-Text search capability 

**SQL Database**:  postrgres.  
Stores:  *User Profiles + Account Info*  (risk-level,  $$$ balance, portfolio (list: symbols & number))  
         Logs (audit trail)  
         Trades (uses: ACID row-level locking)  
         RAG Data:  Social Media, News, Press Releases, Filings, Historical Data (full -text search)

---

## (week 2)  A clear explanation of how and when the agent will use external tools to address known model limitations

The main capablities Prometheus trading bot & the limitations around the usage:

- **Reasoning: Acting, Observing: *(All Agents)***  
The LLM will just act more like a  WorkOrder processing system -  verifying  &  router to tools that process numeric tasks. The  LLMs need to use external tools like Semantic Search, Full Text Search , call APIs ets. The only work the LLMs will do is a lot of parsing and routing except they will summarize. There will be more smaller helper LLM-based agents as the implementation gets to Week3. 
- **WorkOrder management:   Includes Creating Orders**  *(AccountManager Agent)*  
The external tools will need to log everything as they go along so they can continue reliability on failure or restart. The LangGraph Checkpoint external tool 
- **Trading:  Performing physical trades**  *(Trading Agent)*
The LLM does not know if the order filled, or what cash is left. Call when a trade is sent, or when you have to know if it is still open.
- **Gathering Data / Processing Data:** *(DataClerk Agent)*  
The LLM does not have today's quote, news, or filing. Call APIS and Redis when the work order needs a fact (data) that was not already in the prompt and action (search_news, search_filing, fetch_quote etc)
- **RAG-Based Data Processing:**   *(Data Clerk Agent)*  
The LLM needs to delegate the actual  historical data processing and python Requests API calls to external tool and parse there.  The Cache (Redis) and Vector Database (ChromaDB) will be needed for retriving fast live data (honor SLAs) and finding similar news . 
- **Creating  Recommendations & Scoring** *(Advisor Agent)*  
The LLM will need to delegate the mathematical scoring of the recommendations to a calculator tool  The sentiment scoring will likely be a small helper agent - but will have to see how it goes with checking the quality of the bull/bear classifcation results.
- **Judging & Critquing Recommendations**  *(Critic Agent)*  
The critic will need to verify several fields to make sure the recommendation is still timely and a "good deal" To see previous recommendation price deltas need to check the persistent database. It needs to do data anaylsys and also look at the risk level from the User's profile - to make sure the recommendation is not too risky. This reqauires DB access tools. 
- **User: Account & Portfolio Management**  *(AccountManager Agent)*  
The LLM has to  rely on external stable SQL databases (external tool) to  have data integrity and  ACID transaction when trading.  Otherwise, the LLM could go roque and trick the system and have an invalid portfolio. Worse yet, would get banned by the brokers due to trading violations.

## (week 2)  A justification for each added capability showing how reasoning loops, memory, and tools improve reliability compared to a prompt-only approach

- **Reasoning: Acting, Observing: *(All Agents)***  
A single prompt finishes in one shot and never looks at the result. The loop is needed to check that result before the next step runs - so the observation works.
- **WorkOrder management:   Includes Creating Orders**  *(AccountManager Agent)*  
It is too hard to verify all the  User requested query  -  WorkOrder fields in one pass . The saved WORK_ORDER is the same object every agent reads.  Doing all in one pass - also might lose the original query if something does not validate or is another error. This allows some fields to potentally be malformed and allows partial processing.
- **Trading:  Performing physical trades**  *(Trading Agent)*  
A prompt cannot make a trade happen. It has to be sent by a wrapped external tools calling a broker's API. Once the trade happens,  more steps are needed to alert the AccountManager and update the User's profile in the database including calculating their new available fuinds balance.   
- **Gathering Data / Processing Data:** *(DataClerk Agent)*  
A prompt only knows what was pasted in. Today's quote, news, and filings are outside that text, so they have to be fetched. We need cache (Redis) and Full-text search (SQL Database)  for not breakign external data providers SLA (very important!)
- **RAG-Based Data Processing:**   *(Data Clerk Agent)*  
A prompt cannot hold the external dataL news,filings, press releases, historical data for long periods or large groups (like S&P 500) - it runs out of Context Window and also the prompt would expire.  A single prompt answers from training memory. We cannot also search  SUBJECT (trend/keyword/stock symbol) using multiple approaches: Semantic Search and Full-text search in one prompt.
- **Creating  Recommendations & Scoring** *(Advisor Agent)*  
A prompt will write a score that hallunicantes and makes a fluent answer. The number is only reliable when it is calculated from the criteria and then validated by the Critic agent .  A single prompt will just make up a score and not be able to audit it or persist it.
- **Judging & Critquing Recommendations**  *(Critic Agent)*  
A single prompt could write the recommendation  and then grades its own work without validating it. But we need  external tool checks (is good deal, did a new News event or filing happen? Is the risk too high?). A separate check against the rules is what blocks a bad recommendation from becoming a trade - this is too much for one prompt.
- **User: Account & Portfolio Management**  *(AccountManager Agent)*  
A prompt forgets the balance when the chat ends, and two prompts can spend the same cash. The SQL transaction locks the row so the balance and the trade commit once. We need multiple trips to the database to update the normalized tables.

---

## (week 1): Actions Needed To Reach Goal

- Obtain data (historical & API -based), downloaded from .gov sites
- (future: store data as embeddings using RAG)
- Filter data based on search criteria and/or use case
- Aggregate / Summarize & perform sentiment analysis (good/bad score) for ranking
- Calculate aggregate ranking scores for other numerical stock data (stock performance, financial metrics)
- Produce recommendations (specific stocks)
- Double-check these recommendations using separate Critic verification
- Provide final approved recommendations
- Alert the AccountManager of suggested trades (approved recommendations)
- AccountManager checks to see if funds can be raised (sell some ) or has available settled funds to purchase approved trades
- AccountManager follows trading rules like Settling time period (T+1)  settled and (simulated) cash available 
- AccountManager orders the Trader agent to perform specific trades (buy & sell)
- Trading agent reports back to AccountManager  what it traded.

## (week 1): How Feedback Is Used To Guide Agents Behavior Across Multiple Steps

- Parameters (Case 1: Amount money, Case 2 Search terms, Case 3: Find All) will be passed with request.
- Recommendations (trading opportunity) canonical model (structured JSON)
  - This list will have workflow metadata to resume on failure/auditing
- Unit of work (for Reasoning):  list of recommendations will be passed along between agents
  - Chain-of-thought "thinking" enabled when needed. Saved in results
- (future:  Content ID / RAG IDs can be used). Stock tickers and/or search 
- Each agent will validate incoming data.  Only continue processing if input is valid (is Structured)
  - Check: anti hallucination tests, data validity, tokenization
- Prompts will use prompt engineering to be well defined
- JSON-RPC  will be used in the FastMCP pattern from Lab 1.2 
- Structured JSON outputs from tools will be returned to LLM
- Memory  (TBD when we get to this module)
- Agents will update the JSON - Recommendations as they go along, later RAG)

`<end - Module 2 / Week 2 Submission>`