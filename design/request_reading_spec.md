# The Account Manager's model reads the request

Status: NOT BUILT. Written 2026-11-16. It is the next thing to build.

What the tree holds instead (Jacob's checkpoint of 2026-11-16, "works much better. Using this as a checkpoint version"): the INTERIM fix that entry 113 of `docs/design_choices.md` describes, two word rules checked by a program in `intent.py` (`theme_is_in`: a theme of short words passes when every word of it was typed; `quoted_theme`: a phrase in quotes that holds the model's topic is the theme). Jacob's request of section 1 works with them. A worker began this spec and was stopped at once when Jacob took the checkpoint; its edits were undone, so nothing of sections 3 to 6 is in the tree. When this spec is built, its decision gets a NEW entry at the end of `docs/design_choices.md` (114 or the next free number) that says it replaces the word rules of entry 113; section 6's "entry 113 is replaced" is read that way.

## 1. What Jacob said

He sent the page: `I want to buy the best "AI stocks" as a theme. Buy $20000 worth of the top 3` and got
"I could not understand the request (the theme 'AI' shares no word with what you typed)".

- "This is a failure as the DataClerk should be passed the phrase: "AI stocks" as a trend phrase."
- After I proposed a word rule and a quoted-phrase rule checked by a program: "the AccountManager needs to use LLM intelligence to parse out the actual request instead of filtering to a strict templated regular expression. It needs to send the company names, company symbols or keyword phrase on to the DataClerk"

The cause was a program filter: the model read the theme as `AI`, and `intent.py` compares only words of three letters or more with the typed text, so a two-letter theme could never pass. `intent.py` holds a row of such filters. They refuse a correct reading whenever the user's wording is not the wording a pattern expects ("buy 50 msft", `buy 50 shares of "Microsoft"`, a selling phrase that is not in the list, an even split said with a word that is not in the list).

## 2. The rule

| # | Rule |
|---|---|
| R1 | **The model reads the request.** Everything that needs UNDERSTANDING is the model's reading, and no program filters it against the typed words: the action (buy, research, order), the stocks (tickers, company or brand names), the keyword phrase, the side of an order (buy or sell), the allocation (even or default). |
| R2 | **A program checks only what a program can check exactly.** (a) The FORM: JSON of the known shape, the known fields, a ticker has a ticker's shape, a name and a phrase are plain words of the allowed length, one of symbol or company in an order, a number is a number (a bool or a string is refused, entry 111). (b) The NUMBERS, because a number is money: the count, the budget and the shares of an order must each be a number the user wrote (`numbers_in`: digits, `$5,000`, `5k`, a number word); the shares of an order are never a number the user wrote as dollars or as a percentage; a number of shares has at most six decimals and is never rounded (entry 110). |
| R3 | **What the model named goes to the Data Clerk, and the Data Clerk is the check that it exists.** Tickers and company names are looked up in the directory as today (`request_symbols`, `classify`, `find_companies` for a list item; entries 101 and 102 for a name that is not found); a keyword phrase goes to `DataClerk.find_companies` as the model wrote it. |
| R4 | **Entry 98 stays** (Jacob's own rule: a named stock is the user's own trade unless the words ask for a recommendation): `RECOMMEND_CUE` and `as_direct_order` are unchanged in what they decide. Only the checks they call change (section 3). |
| R5 | **The user is told what was understood** (`TradeIntent.describe()`), as today. Nothing else replaces the removed filters: no second model call, no new pattern. |

## 3. What changes in `prometheus/agents/account_manager/intent.py`

**Removed** (each is a filter of the user's wording):
- "the ticker X is not in what you typed" and "the company ... is not in what you typed", on the search path and in `check_order`.
- "the theme ... shares no word with what you typed", and the two rules built on 2026-11-16 before Jacob's second message: `theme_is_in` (a short theme checked word by word) and `quoted_theme` (a quoted phrase made the theme by a program), with `QUOTED`, `CLERK_PHRASE`, `CLERK_PHRASE_CHARS` and `_token_list` when nothing else uses them.
- "an even split was not asked for in what you typed" (`EVEN_CUE` in `parse_intent`) and "buying was not asked for in what you typed" (`BUY_CUE` in `parse_intent`).
- In `check_order`: the position templates. The shares are no longer looked for "right before the stock" (`_order_phrases`), and the side is no longer taken from the last buying or selling word before it (`_cue_before`). "you typed N shares of X: put the number in shares, not null" goes with them.

**Kept or put in their place** (R2):
- The count and the budget: a number the user wrote, as today.
- The shares of an order: a number the user wrote (`numbers_in`), not one written as dollars or as a percentage, at most six decimals. One function says whether a number was written as money or as a percentage; it looks at the number's own token (`$` before it, `dollars`, `USD`, `%`, `percent` after it), not at the sentence around it.
- `RECOMMEND_CUE` in `check_order` and in `parse_intent` (R4).
- Every check of the form.
- `TradeIntent.what()`: a theme whose last word is "stocks", "companies" or "shares" is not followed by "companies" (already built).

`clarify.py` is NOT changed: its answers are numbers and its `EVEN_CUE` use is its own (entry 85). A helper that nothing uses after this change is removed; one that `clarify.py` or another module still uses stays.

## 4. The prompt (`prometheus/prompts/account_manager_request.py`)

The model is now the reader, so the prompt carries what the filters carried:
- **topic** is the keyword phrase to search for: the industry, theme or trend the person wants, in their own words. A phrase the person put in quotes is kept word for word (`"quartz stocks"` in quotes stays `quartz stocks`). Without quotes the word stocks or companies is left out, as today.
- **symbols**: the tickers the person means, in capitals whatever case they typed.
- **companies**: the company or brand names the person means, written properly (an obvious slip of spelling or spacing corrected). A company's name never becomes a ticker from the model's memory: the directory finds the symbol (entry 101).
- **orders**: the same for the stock of an order; the side is what the person means, whatever word they used for buying or selling.
- Write down only what the person asked for: never add a stock, a theme or a number they did not give. If they gave no stock and no theme, leave all three empty.
- The examples stay made-up words; no real sector, company or ticker is named in the prompt. The module text is updated: code checks the form and the numbers, not the words.

## 5. Tests

Real data only; no stand-in model in a new test; do not edit an existing capture.

- Capture the REAL model's replies with the NEW prompt (one call each, the standing yes of entry 84) into a new fixture, for:
  1. Jacob's sentence (section 1).
  2. `I want to buy the best AI stocks. Buy $20000 worth of the top 3`
  3. `Buy the 3 most recommended EV stocks with $5,000`
  4. `Look at the best 5G companies`
  5. `Put $20,000 into the 3 best AI names, equal amounts` (no buying word of the old list in the usual place, no "evenly")
  6. `buy 50 msft` (a lower-case ticker)
  7. `buy 50 shares of "Microsoft"` (the quote the old template refused)
  8. `get rid of 10 shares of Tesla`
  9. `Buy $5,000 of MSFT` (dollars are never shares)
  Record what the model really says. A reply that is not what this spec expects is a finding to report, not something to hand-write.
- The real replies go through the real `parse_intent`: 1 to 5 give the keyword phrase, the count, the budget and the allocation the person meant; 6 to 9 are direct orders with the stock, the side and the shares the person meant.
- The numbers are still checked: a real reply set against a sentence with a different number is refused (count, budget, shares); a number the user wrote as dollars is refused as shares.
- Every existing real capture of a reply still gives the same intent as before. One that now differs is a finding.
- An existing test that pins a REMOVED refusal is changed to the new behaviour or, when the refusal was all it tested, removed; list each in the report.
- One live session with Jacob's sentence under the new prompt: Auto mode, throwaway account, no fetch, `--cleanup`, saved over the capture the worker made earlier today (`tests/fixtures/session_real/auto_ai_quoted_theme.json`, its own file, not yet reviewed). Pin only what is stable: the reading, and that the run spends no more than $20,000.
- The whole Python suite is green apart from the known failure; `tests/test_no_float_in_the_account.py` stays green.

## 6. Documents

- Entry 113 of `docs/design_choices.md`: the worker wrote one today for the word rules. That text was never accepted, so it is REPLACED by this decision (problem, decision, alternatives: the word rule and the quoted-phrase rule checked by a program, refused by Jacob; a second model call to verify; consequences: a wrong reading of a stock or a theme is no longer refused by a program, the Data Clerk and the sentence shown to the user are the checks, the numbers are still exact).
- `AGENTS.md` (the paragraphs that say code checks every field against the user's words), `docs/SESSION_NOTES.md` (and its suite count), `HowToRun.txt` and `README.md` where they describe the check, the module text of `intent.py` and of the prompt.

## 7. What the first attempt found (2026-11-16; read before building)

A worker changed the prompt as section 4 says and captured the nine real replies of section 5 before it was stopped. They are kept in `tests/fixtures/session_real/request_reading_forms.json` (no test reads the file; the prompt they were made with is not in the tree). What they show:

1. **The model did not keep the quoted phrase.** For Jacob's sentence it still answered topic `AI`, so the Data Clerk would be asked for "AI", not "AI stocks". Section 4's wording has to be firmer, with a made-up example in the prompt, and checked on a real reply; if the model still drops the word, say so to Jacob instead of adding a program rule for quotes.
2. **Without the buying-word filter a request to LOOK became a purchase.** `Look at the best 5G companies` was read as action `buy` with count 5 (the old prompt gave `research` and no count; the 5 of "5G" is a digit in the text, so the number check let it through). In Auto mode that would have spent money the user did not ask to spend. So section 3 is NOT right as written for the action: before the "buying was not asked for" check is removed, the prompt must make the model reliable on look against buy (real replies to several "look at", "show me", "what are" sentences), and a count must not be read out of a name such as 5G. If the model cannot be made reliable, the check that a purchase was asked for STAYS (it guards money, like the numbers), and Jacob is told.
3. **The position checks of a direct order guarded real mistakes too**: with them removed, a number or a side swapped between two orders ("buy 50 MSFT and sell 10 TSLA"), or `"shares": null` when the user wrote a number, is no longer refused. Section 3 keeps the number as "a number the user wrote", which does not catch a swap. Decide this with Jacob before removing them: the choice is between his rule that a run never does with money what the user did not say, and his rule that the model reads the request.
4. The other replies were as hoped: `EV` and `AI` as themes with their counts and budgets; "equal amounts" read as an even split; `buy 50 msft` as MSFT 50; `buy 50 shares of "Microsoft"` as the company Microsoft, 50; `get rid of 10 shares of Tesla` as a sale of 10; `Buy $5,000 of MSFT` as MSFT with a budget of 5000 and no shares.

So build it in two parts. Part 1, safe and what Jacob's failure was about: the theme, the tickers and the company names of a SEARCH are the model's reading, with no word filter. Part 2, only after items 2 and 3 are settled with Jacob: the action, the side and the shares of an order.

## 8. Open, for Jacob

1. **The numbers are still checked by a program** (R2): a budget, a count or a share count that is not a number you wrote is refused, and you are asked to write it in digits. "twenty thousand dollars" is refused today for that reason; say if the model should read numbers alone too.
2. **Entry 98 is unchanged** (R4): a program still decides, from the words that ask for a recommendation, whether a named stock is your own order or a request to recommend.
