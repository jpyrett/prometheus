Jacob Pyrett - Week 6 Submission
Nov 18, 2026  CMU Agentic AI  

**1. What is your capstone agent designed to do, and what safety or reliability risks could arise in its intended workflow?**

Prometheus is a multi-agent stock trading desk. The Account Manager takes a user request and routes it to the other agents (Data Clerk, Advisor, Critic, Trader) to gather data, generate a recommendation, trade in simulation, or just get information. The risks are interference with each of these processes. The data is live news, press releases, historical price files and current quotes from an API. It comes from paid data providers and government filings rather than generic sites, which lowers the risk of poisoned content but does not remove it. The main risks are poisoned RAG documents and prompt injection, a malicious user hijacking an agent, leaking another user's account or work order, direct database access, ungrounded recommendations, overspending, and trading outside the simulation.

A further risk is violating the data providers' SLAs, especially the government EDGAR system, by fetching too much news or too many filings. There is no caching layer (like Redis) yet, so the current system will not scale. A user can also spam requests to break those SLAs and clog the backend, so the security layer should detect spamming and DDoS attempts from the same user.
**2. What guardrails will you use to constrain the behavior of your agent?**

The rules are security policies written in YAML. A decision router enforces them in code, and they are RAG-ingested so the agents are forced to apply them and a rule is never skipped or hallucinated away. An LLM can ignore a prompt-only rule; it cannot ignore the router.

- **User input:** every user is treated as malicious and carries the risk of using the application. The user talks only to the Account Manager, which stays on alert and treats all input as an attempt to game the system or steal data. No access to other agents, no Postgres connection, no SQL or DDL, no modifying a work order, and no view of another user's account or work order. Requests must follow SEC and FINRA rules.
- **Ingestion:** documents are scanned for poisoning and injection before indexing. The Data Clerk reaches only the registered sources: Finnhub, Massive and EDGAR filings.
- **Agents:** only prompts in the registered Prometheus prompt folder run, only financial topics are retrieved, and an agent sees only the current user's account.
- **Outputs and tools:** the Trader trades only in simulation. The Account Manager cannot export, email or transmit data outside the application. A run never spends more than its funds. The MCP server serves the internal agents only, with no third-party plug-ins.
- **Platform:** CORS and React hardening, a user-supplied OpenRouter key for local runs, and a pytest suite for development only.
- **Added:** a least-privilege database role, secrets kept out of prompts and logs, rate limits, an audit log, and failing closed.

The router is scoring-based: **dynamic runtime enforcement, where scorers route every output using a scorer LLM as the judge**. The strategy is an **LLM Risk Scorer** and a **Trust Score**.

- **Risk Score (danger):** a cheap code rule runs first, then the Qwen judge rates the output from 0 to 1 on injection, personal data and policy. The highest value is the score and the router acts on it. Examples: a news article with a hidden instruction or an answer showing another user's balance is blocked; a question that is not about finance goes to review.
- **Trust Score (reliability):** a weighted average of how grounded the answer is, its source, its confidence and its history.

The scoring below is the class example from the Module 6 Live Session; Prometheus will use it with its own rules:

```python
def trust_score(s):
    # weighted average of 0-1 signals
    return .4*s["grounded"] + .2*(s["source"] + s["confidence"] + s["history"])

def act(risk):
    if risk >= 0.8: return "BLOCK"
    if risk >= 0.5: return "REDACT_OR_REVIEW"
    if risk >= 0.2: return "ALLOW_AND_FLAG"
    return "ALLOW"
```

*Source: class example, Module 6 Live Session.*

**3. What evaluation metrics will you use to assess the performance of your agent?**

- **Task Success Count and Percentage:** how many runs of each task succeed, out of how many were tried. The tasks are:

  > <small>Accept & Parse User Request · Gather Data · Refresh Historical File-Based Data · Filter News By Date · Filter By Keyword Trend Phrase · Map Company Name -> Symbol · Create WorkOrder · User Profile -> Register · Add Funds · RaiseFunds · Generate Recommendations · Critique Recommendations · Trade · Trade Account Funds Handshake</small>

- **Risk Score vs Trust Score:** the two numbers side by side for every output: danger versus reliability. Risk rises when a rule is at stake: a hidden instruction in a news article or filing, a request for another user's account or work order, SQL or direct database access, a topic that is not financial, a prompt outside the registered prompt folder, a data source that is not Finnhub, Massive or EDGAR, exporting data outside the application, a trade outside the simulation, or spam from one user. Also counted: the outputs sent to each route (allow, allow and flag, redact or review, block).
- **Injection block rate %:** the share of injection and poisoning attempts in the red team's attack set that are blocked, with the false-positive rate on legitimate requests.
- **Red-Team/Blue-Team attack ratio %:** the share of red-team attacks that get through, and how many the blue team has closed with a fix.
- **Policy violations count:** measured by LangSmith tracing detection and router scoring.
- **Data integrity metric (account money):** the ability of the account management & trading handshake to prevent overspending when trading and when raising funds. The portfolio math must always add up for the trades made and the funds the user added or removed.
- **Judge accuracy %:** how often the LLM judge reaches the same decision as a person. I mark a sample of requests, documents and agent outputs by hand with the action each should get; the metric is the share where the judge's score led to the same action.
- **Groundedness %:** the share of claims that pass the citation check.
- **Escalation rate %:** the share of requests and outputs sent to a person for review.
- **Critic denial rate %:** the share of the Advisor's recommendations that the Critic denies.
- **Guardrail overhead:** the latency and the cost the guardrails add to each run.

**4. Under what conditions should your system require human intervention rather than allowing the agent to proceed autonomously?**

- **Ask mode, direct request:** the Account Manager parses the user's request into an "intent": get a current quote, or buy or sell a named stock, with how to split the purchased stocks (evenly by dollars, round-robin, or an exact number). When a direct order is missing a field or the request is ambiguous, a ReAct loop asks the user to clarify before anything is placed.
- **Ask mode, trades:** the user approves every trade. If a trade fails or has a funding problem, the Account Manager notifies the user.
- **Risk score:** the scorer routes every output at runtime. A risk score of 0.5 or more redacts the output or holds it for operator review, and 0.8 or more blocks and logs it. A score of 0.2 or more is allowed but flagged for a later look. Auto mode asks the user nothing, so a held output there is denied.
- **Critic:** it denies any recommendation that needs a person to look. **TODO, not built yet:** add the ability for the user to override the Critic. There is a Strict recommendations mode that partly handles this, but it needs to be expanded.
- **Disagreement:** the judge and the rules disagree.

**5. How will guardrails, evaluation, and human intervention work together as part of an overall safety strategy?**

The dynamic Scorer will use the YAML rules to enforce security. Scorers then route every output: a separate Qwen model is the scorer LLM and judge, and it gives each request, document and agent output a risk score. The router routes by that score: allow, allow and flag, redact or review, or block. LangSmith traces every step between the agents, and the metrics are used in the dynamic scoring.

The guardrails are tested with a **red team and a blue team**. The red team attacks: it tries to break in with prompt injection, poisoned documents, requests for another user's data and spam. The blue team defends: it detects and blocks those attacks and closes each hole the red team finds, for example by adding an input scan for hidden instructions, tightening a YAML rule or adjusting a score threshold. Every known attack becomes a new regression test.

**6. What trade-offs did you consider in your design, particularly in terms of autonomy, reliability, efficiency, and oversight?**

- **User input first:** make sure the user is asking for one of the specific actions: ask about something, get recommendations, place a direct order to trade, or manage their portfolio and account.
- **Then the work order:** once it is generated, protect it and enforce the integrity of the RAG data and of the trades, using the handshake while the trades happen.
- **Scoring:** the judge scorer router decides on the Risk Score versus the Trust Score, using the scoring thresholds.
- **Auto versus Ask Mode is a trade-off:** we want a smooth experience where the user is not bombarded by too many irritating questions, so Auto mode fails the ambiguous steps instead of asking. Ask mode trades speed for oversight.

**7. How does your plan support safer and more dependable deployment of your agent in a real-world context?**

Every action is constrained by a written policy, scored and traced. Only registered prompts run, and the policies are RAG-ingested so they are applied as written and not hallucinated. The LLM scorer is a large, capable Qwen model, so the judge is smart enough to catch what the code rules miss. We will use *LangSmith* for observability and traceability. The system fails closed, cannot reach a real broker, and isolates each user's data. The metrics show whether the guardrails work before cloud hosting or real trading is considered.

-- end of submission --
