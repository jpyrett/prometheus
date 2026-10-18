# Capstone: Self-Learning Trading Bot ("Prometheus")

## Jacob Pyrett , Oct 2026 - CMU Agentic AI Course

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

I propose this agentic system which:

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
- Single user initially with the user cases: Case 1 ($1000), Case 2: "AI stocks",  Case 3:  "What's Happening?"
- Several concurrent users:  2, 5, 10 . How does system scale?
- Agents are users. Do they improve over time? Earning totals from back testing better?

### Systems

- Test harness:  using Cursor's agent harness
- Data providers: [Finnhub.io](http://Finnhub.io) ,  [massive.com](http://massive.com)   
- Government public filings
- Local AI:  I have two (2) DGX Spark AI workstations [256 GB VRAM, 40 cores, 12.2K CUDA cores, 8 TB SSD] will run final here
- OpenRouter:  used for Development with my Linux laptop + Cursor IDE.  Then, will move to Local AI for daily executions

### Models

- Different agents will need different models. Advisor & Critic will need stronger reasoning (  mid tier models: Nemotron 3 120B or 550B). The rest should work with faster/cheaper 30B models from Qwen and/or GPT-OSS-120B  or less.
- Leverage OpenRouter (cheap mid-range model) for development, local AI for daily execution

## Actions Needed To Reach Goal

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

## How Feedback Is Used To Guide Agents Behavior Across Multiple Steps

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

<end - Module 1 / Week 1 Submission>