# Direct orders: a brand finds its listed owner, and a list goes on without what was not found

Decided by Jacob on 2026-11-15, in his words:
- *"So, Google is now Alphabet. Is there a way to cross refernece and/or fix this in the Data clerk company name search to return GOOGL"*
- *"Missing companies in a list should keep going and ask the user if they want to ignore the missing companies and continue (ReAct pattern). If yes to continue, them drop the invalid companies from the workorder and contrinue"*
- *"A better user message is needed when a company is missing, than just stopping. Notify the user with a ReAct prompt that this is missing and OK to continue for each missing companiy not matched or found."*
- On the advisor's two open points, the same day: the model words the prompts (*"Yes fix the first one"*), and *"Auto mode will not ask any prompts. Stop and ask the user to clarify the initial prompt if there is a blocking issue. Remove missing (not found / unmatched) companies if in auto mode"*.

Written by Fable 5.2 as the design advisor. Read `AGENTS.md` first. This amends entries 87 and 88 of `docs/design_choices.md`; it gets its own new entries at the end (the next numbers after 100) and the old entries are not rewritten.

## 1. What happened

Jacob's first live request through the web page:

> I have 10000 to spend. Create a new user profile for me: name is D Pyrett. I am 40 with a moderate risk profile. Buy as much stock in: F, GM, Procter and gamble and Google. And Chick-fil-a.

The Account Manager read it as a direct order of five purchases with $10,000 to spend. It asked the Data Clerk for the names and symbols (`DataClerk.request_symbols`), which answered:

| Typed | Answer |
|---|---|
| F | Ford Motor Co (F), by its ticker |
| GM | General Motors Co (GM), by its ticker |
| Procter and gamble | The Procter & Gamble Company (PG), by its name |
| Google | `Google: Company Name not found.` |
| Chick-fil-a | `Chick-fil-a: Company Name not found.` |

Two names were not found, so `name_the_stocks` returned two problems, the desk ended at its first node and nothing was placed.

**Measured the same day:**
- Finnhub's symbol search returns **0 rows** for `google` and for `facebook`, and **one row** for `alphabet`: GOOGL, "Alphabet Inc.". The directory holds no brand and no former name, so it cannot cross-reference by itself.
- The Data Clerk already has a brand finder, used by the research path for a list item such as "Burger King" (`DataClerk.find_companies(name, count=1)`, `recommendations._list_companies`). Asked for "Google", its model proposed the owner "Alphabet Inc." and code found that name in the directory by the same words: **GOOGL**, `how: company name "Alphabet Inc.": the same words as the listing`. Asked for "Chick-fil-a", it answered no owner: `privately held, or not known to be listed`. One model call each (gpt-oss-120b, about 450 tokens in).

So the knowledge exists in the desk. The direct-order path does not use it.

## 2. Change 1: the company name search finds the listed owner of a brand

**Where:** `DataClerk.request_symbols` (the `lookup_symbol` work order), and what it needs in `data_clerk_agent.py`.

1. **First, the exact lookup, unchanged.** A ticker in capitals is the row whose symbol is it. A name is the row with the same words, else the same words without the generic company words. Most requests end here and no model is called.
2. **New, only for a NAME the exact lookup did not find** (never for a ticker, and not for a name that fits several listings: the directory already answered that one): the Data Clerk asks its model for **the listed owner of that name** (the proposal step of `find_companies`, for one name), then looks **the owner's name** up with the same exact lookup (`symbols.find_company`).
   - **The symbol is always the directory's.** The model supplies a company name and nothing else that is used. Its ticker hint is not used here, and neither is the "start of the listed name" tier of `_resolve_proposal`: those are fine for choosing companies to rank, and a trade needs the exact words.
   - No stored-news discovery for this (`discover_tickers` is for a theme, and it prints a progress bar).
3. **Found:** a row in the shape `find_companies` already gives: `{brand: "Google", parent: "Alphabet Inc.", symbol: "GOOGL", company: "Alphabet Inc.", public: True, how: "brand", note: 'the listed owner of what you typed, named by the Data Clerk\'s model; the symbol is the directory\'s'}`.
4. **Not listed:** the name gets a row too, with no symbol, and the reason in the Data Clerk's own words: `{brand: "Chick-fil-a", parent: None, symbol: None, company: None, public: False, how: None, note: "privately held, or not known to be listed"}`. The order's error for it is `Chick-fil-a: privately held, or not known to be listed` (or `no listed company named '<owner>' in Finnhub's directory` when the owner the model named is not there). A name the exact lookup refused for fitting several listings gets a row with no symbol as well, with the candidates as its note.
5. **A lookup that FAILED is never "not found"** (a rate limit, no key, the model did not answer): `<name>: the symbol lookup failed (...)`, as today, and **no row**. That is how the Account Manager tells a company that is not listed from a question that was not answered.
6. The order is still `data_failed` when any name has no symbol, with the rows on it, so every other caller behaves as today.

The user's record of a brand (`order.direct_orders`) has `typed_company: "Google"`, `company: "Alphabet Inc."`, `symbol: "GOOGL"`. The confirmation that already exists for a company the user NAMED applies to it unchanged: in Ask mode the user confirms the symbol, the company, the shares and the amount; in Auto mode the returned symbol is used. The confirmation card and the report say that the symbol is the listed owner of what was typed.

## 3. Change 2: a list goes on without the companies that were not found, when the user says so

**Where:** `AccountManager.name_the_stocks`, `agents/account_manager/direct.py`, `langgraph/desk.py` (`read_request`).

After the lookup every stock the user named is one of three:
- **found** (a row with a symbol);
- **not listed** (a row with no symbol: not found, private, or fitting several listings);
- **not answered** (no row: the lookup failed).

| Case | What the Account Manager does |
|---|---|
| All found | As today. |
| Any not answered | Stops, as today, with the reason. Nothing is offered: an outage is not a company to ignore. |
| None found | Stops, as today, with the reasons. There is nothing to go on with. |
| Some found, some not listed, **Ask mode** | **Tells the user about each one that is missing and asks whether it is OK to continue without it**, one prompt for each. |
| Some found, some not listed, **Auto mode** | **Asks nothing.** The companies that are not listed are removed from the work order, the user is told which and why, and the order goes on with the rest. |

The mode is the run's own (`request["account"]["mode"]` when given, else the trade mode of the profile the Account Manager holds), the same one `decide_direct` uses.

**Auto mode asks no prompt** (Jacob). A company that was not found or not matched is dropped without a question: a line for each in the feed and the terminal ("Auto mode: Chick-fil-a was not found (privately held, or not known to be listed) and was left out; the order goes on with F, GM, PG and GOOGL"), in `order.data.notes` ("dropped in Auto mode: ...") and in `plan["dropped"]`. When the issue is BLOCKING (nothing the user named was found, the lookup failed, two orders are for the same stock, the request could not be read), the run **stops and asks the user to clarify the initial request**: the reasons, then "Please clarify your request and send it again: name each stock by its ticker or by the listed company's name." That is the run's last message, not a prompt in the middle of a run. The rest of this section is Ask mode.

**The prompts** are a bounded loop of the Account Manager in the ReAct shape Jacob asked for, the same shape as the how-much questions of `clarify.py` (entry 85): it OBSERVES the Data Clerk's answer for a name, ACTS by telling the user and asking, OBSERVES the reply, and goes on to the next missing name or stops. One prompt for each company that was not matched or found, in the order the user typed them, never more prompts than missing names. It is asked in **Ask mode** only, when there is somebody to ask.

**The model words each prompt; code checks it; code reads the yes or no** (Jacob: the model words them, as in the how-much loop). For each missing company the Account Manager's own model (`self.model`, the `account_manager` role) is given facts made by code (the name as typed, the Data Clerk's reason, what stays on the order now, the user's request) and returns one question as JSON. The prompt is in `prometheus/prompts/` with the others. Code accepts the wording only if it names the missing company as the user typed it, is one question, and is no longer than the limit in `configs/account_manager_v1.json` (a new `missing` section: the most prompts, the most characters; every threshold stays in the config). A wording that fails the check, or a model that fails or times out, never blocks the user: **the code's own sentence is shown instead**. The answer is a yes or a no from the channel, read by code; the model does not decide anything and does not read the answer. A prompt must be a sentence a person understands at once. It says **what** is missing, **why**, and **what happens on a yes**. The code's own sentence, and the model for what the model should say:

> **Chick-fil-a was not found:** privately held, or not known to be listed. OK to continue without it? Your order would go on with F, GM, PG and GOOGL.

It goes through the desk's `approver`, the yes/no channel that already carries a card with its own `question` (the Trader's funding card). No new argument on `build_desk`. One card for each missing company:

```python
{"kind": "missing_company", "n": 1, "of": 1,
 "typed": "Chick-fil-a", "reason": "privately held, or not known to be listed",
 "text": "      NOT FOUND   Chick-fil-a: privately held, or not known to be listed\n"
         "        still on your order   F: Ford Motor Co (F)\n"
         "                              GM: General Motors Co (GM)\n"
         "                              Procter and gamble: The Procter & Gamble Company (PG)\n"
         "                              Google: Alphabet Inc. (GOOGL), the listed owner of what you typed",
 "question": "Chick-fil-a was not found (privately held, or not known to be listed). OK to continue without it? [y/N] ",
 "kept": [{"typed": "F", "symbol": "F", "company": "Ford Motor Co"}, ...]}
```

- **Yes** (`True`) to a prompt: that company is **dropped from the work order** and the next missing one is asked about. After the last yes, `order.direct_orders`, `order.subject` and the plan's intent hold only the companies that were found, so the money the user named is spread over those. What was dropped stays on the record: a line for each in `order.data.notes` ("dropped at the user's word: Chick-fil-a (privately held, or not known to be listed)") and `plan["dropped"]`. The state's `errors` are empty and the desk goes on: `account_manager_direct`, then the Trader.
- **No** (`False`) to any prompt, **or no answer** (`None`: nobody at the keyboard, no approver, the prompt timed out): the order stops, and nothing is placed. No answer is never a yes. (Ask mode with nobody to ask is NOT Auto mode: nothing is dropped on the user's behalf.)

**Never just stopping.** Whenever the order stops here (a no, no answer, nothing found, a lookup that failed), in either mode, the user is told in plain words what was missing and why, and is **asked to clarify the request and send it again**: each name with the Data Clerk's reason, then one line of help ("Please clarify your request and send it again: name each stock by its ticker or by the listed company's name, or leave it out"). The web page's last feed line already does this (`Emitter.finished`, `NOT_PLACED`); the terminal session must say the same and not only "the Data Clerk's work order is new".

Each trade that is placed is still placed as typed or not at all (entry 88). What changes is that a list is no longer refused whole for one name that is not a company.

The research path's list (`recommendations._list_companies`) is not changed: it already goes on without an item that is not listed.

**The session** (`application/session.py`): the approver it already builds asks at the terminal, and shows the card's text and question as it does for the funding card. `--yes` does not answer this question (it answers only "is this what you meant").

**The web page** (`application/web/desk_run.py`, `ui/`): a card with `kind: "missing_company"` becomes a question card in the Activity panel, in the style of the how-much questions: the badge "ACCOUNT MANAGER ASKS" (with "1 OF 2" when there are several), the sentence above as its text ("**Chick-fil-a was not found:** privately held, or not known to be listed. OK to continue without it?"), what stays on the order in the monospace block, and the buttons **OK, continue without it** and **Stop the order**. The stage is "waiting for you" while it is open. The feed then says what was answered ("You: continue without Chick-fil-a") and, from the Account Manager, what was dropped. The work order table shows the companies that were kept.

## 4. Tests

Real data only (`AGENTS.md`); no stand-in agent or model in a new test.

- **Capture once, then test offline.** Record the real replies this needs: the Finnhub searches the lookup makes for F, GM, "Procter and gamble", "Google", "Chick-fil-a" and "Alphabet Inc." (in the form `tests/fixtures/finnhub_company_search.json` already has), and the model's two owner replies for "Google" and "Chick-fil-a" (in the form `tests/fixtures/session_real/direct_order_forms.json` has for the model's replies). These few calls are allowed (Jacob's standing yes for paid runs that test the desk; nothing is fetched or written to the stored data).
- **Data Clerk:** Google gives GOOGL with `how: "brand"` and the symbol from the directory; Chick-fil-a gives a row with no symbol and its reason; a ticker and a name the directory has never reach the model; a failed lookup gives no row and is not "not found".
- **Auto mode, on Jacob's own request:** nothing is asked (a test fails if the approver or the model that words prompts is called); Chick-fil-a is dropped and noted as dropped in Auto mode; four direct orders go on. With every name not listed, and with a failed lookup, the run stops with the reasons and the request to clarify.
- **The wording:** the model's real reply for Chick-fil-a (captured once, beside the other real model replies) passes the check; the same real reply checked against another company's name does not and the code's sentence is used; with no model the code's sentence is used.
- **Ask mode, on Jacob's own request:** one prompt (Chick-fil-a; Google is found as GOOGL and is not asked about here); with a yes, four direct orders (F, GM, PG, GOOGL), Chick-fil-a dropped and noted, the path goes on to `account_manager_direct`; with a no and with nobody to ask, the order stops with the reasons and the line of help; two missing names give two prompts in the order typed, and a no to the second stops the order although the first was a yes; with every name not listed, stopped and nothing asked; with a failed lookup, stopped and nothing asked.
- **Web:** the `missing_company` card becomes the question card with its sentence, its block and its two buttons; the answer goes back through the answers route; a run that stops says what was missing in the feed's last line.
- Existing tests that pin "a name that is not found stops the order" or "the symbol lookup calls no model" are amended to the new rule, each with a line saying why; none is deleted.
- **One live session at the end** (allowed, no fetch, a throwaway account with enough cash that is deleted afterwards): Jacob's request in Ask mode with the question answered yes. Save it as a real capture (`session --save`) beside the others in `tests/fixtures/session_real/`.

## 5. Documents

- `docs/design_choices.md`: two new entries at the end (the next numbers after 100): the brand's listed owner in the Data Clerk's lookup (problem, decision, alternatives: an alias table kept by hand, Finnhub's search alone, the model's ticker; consequences: one small model call only when the exact lookup finds nothing), and the list that goes on (problem, decision, alternatives: refuse the whole list as before, drop without asking in Ask mode too, ask in Auto mode too, prompts worded by code only; consequences).
- `AGENTS.md` (the direct-order paragraph), `HowToRun.txt`, `docs/SESSION_NOTES.md`, the module texts of `direct.py`, `desk.py` and `DataClerk.request_symbols`, `design/ui_v1_spec.md` (the new question in 5.3).

## 6. Not in this change

- Creating a user profile from the request's words ("name is D Pyrett, I am 40, moderate risk"). The request box does not do it; the Profile card does. It would need new fields on the Account Manager's form.
- Asking again for a corrected name ("did you mean ...?"). A dropped company is dropped.
