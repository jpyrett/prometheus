# Critic v2: a denial reason must earn its place

Status: spec for the coding worker. Written 2026-11-20 by the design advisor for Jacob. It becomes design choices entry 133 (the Critic), 134 (an exit plan for a neutral purchase) and 135 (the cooldown's replacement). It changes rules of `design/critic_v1_final_spec.md` and one consequence of entry 103.

## 1. What Jacob said

On the five findings of the Proving Ground (the Critic's verdict is unstable; no evidence it adds value; neutral purchases never exit; the cooldown holds back half; every bullish pick was denied in the two live sessions):

> *"Improve these: especially the Critic. Make it work better by finding reasons to deny the recommendations. Make it add value."*

## 2. What the outcomes say (the evidence this spec rests on)

296 recommendations that the Critic reviewed and that have an outcome, from four six-month Time Machine tests (weekly, 2026-06-08 to 2026-11-09, fast): oil 149, AI 79, restaurants 41, consumer staples 27. Excess = return against SPY over the same days. A denial for the user's RISK PROFILE (`risk_above_profile`) is suitability, not a forecast, and is left out of every comparison of returns (37 of them).

**The Critic as it is, judged by returns:**

| Theme | Approved | Denied | Gap (approved minus denied) |
|---|---|---|---|
| Oil | 90: +0.85, 77% won | 57: +0.60, 79% won | +0.25 (0.4 standard errors) |
| AI | 28: -0.68, 46% won | 21: +1.44, 86% won | -2.11 (-2.6 se): WRONG |
| Restaurants | 25: +1.71, 50% | 11: +2.50, 38% | -0.79 |
| Staples | 22: +1.07, 73% | 5: +0.88, 75% | +0.18 |
| All | 165: +0.75, 68% | 94: +1.02, 77% | -0.27 (-0.4 se) |

**Why: three reasons make most denials, and they deny winners in every theme.**

| Reason | Fired on | Of the closed ones, won | Note |
|---|---|---|---|
| `fresh_negative_news` (a warning that asks for a person, so a denial since entry 103) | 76 | 60 of 72 | "looks bad" by the model's label of a newer article |
| `blend_not_supported` (a blocker) | 42 | 37 of 41 | the Critic's model relabels the Advisor's quotes and the blend falls under 0.3; the values run from 0.06 to 0.296, not near misses |
| `company_mismatch` (asks for a person) | 15 | 11 of 15 | A BUG: "Exxon Mobil" against the directory's "Exxonmobil Holdings Corp", "APA Corporation" against "Apa Corp" |

**What does separate losers, in all four themes:** a recommendation of the TREND family for a stock that FELL over the five sessions before the decision (`ret_5d <= 0`).

| Theme | Trend and falling | Everything else | Gap |
|---|---|---|---|
| Oil (where it was noticed) | 36: -1.03, 44% won | 111: +1.33, 88% won | +2.36 (2.7 se) |
| AI (not used to find it) | 9: -0.30, 0 of 9 won | 40: +0.35, 79% | +0.65 |
| Restaurants (not used) | 11: -2.99, 27% | 25: +4.13, 58% | +7.12 (2.8 se) |
| Staples (not used) | 7: +0.82, 57% | 20: +1.11, 79% | +0.29 |
| All | 63: -1.06, 37% won | 196: +1.46, 82% won | +2.53 (3.8 se) |

It has a reason, not only a number: the trend family's premise is that a move continues, and a stock that is going down this week does not confirm it. About twenty candidate rules were looked at on the oil data; this is the one that held on the three themes it was not found on. The others did NOT hold and are NOT built: near the 52-week high, a rich price at a lower threshold, a high Advisor score, a falling market week, RSI, the z-score. `overpriced` picks losers in oil (8 of 13 won) and denies winners in AI (10 of 10 won): mixed, left as it is.

**The honest limits.** One six-month window, four themes, fast mode, overlapping weeks (the outcomes are not independent, so the standard errors flatter the result). The rule was chosen on this window; only later outcomes can confirm it. Section 6 is how they will.

## 3. The rules

| # | Rule |
|---|---|
| C1 | **A denial reason must earn its place.** Every finding code of the Critic is scored against the outcomes by a program (the report card, section 6). A blocker that denies winners is a defect to fix or demote; a new blocker needs a reason in words AND outcomes on data it was not found on. |
| C2 | **`fresh_negative_news` informs, it no longer denies.** It stays a warning and no longer asks for a person. The serious classes are untouched: `fresh_negative_event` (going concern, restatement, material weakness, guidance cut, earnings miss, dilution, dividend cut) is still a blocker. Entry 103 stands for every other finding that asks for a person. |
| C3 | **`blend_not_supported` is a warning, not a blocker.** The counterfactual blend is still computed and shown, the evidence dimension still loses its points, and the review still says so. |
| C4 | **A company is the same company whatever its spelling.** `company_mismatch` compares the names with case, spacing, punctuation and corporate words removed ("Exxon Mobil" is "Exxonmobil Holdings Corp"; "APA Corporation" is "Apa Corp"), and the symbol itself named in either counts. A name that still does not match is the same finding as today. |
| C5 | **`trend_not_confirmed` (NEW blocker).** A recommendation whose trade family is `trend` is denied when the stock's return over the last five sessions, as the Critic's OWN re-asked metrics give it (`ret_5d` of `request_metrics`), is at or under `chart.trend_confirm_ret_5d` (config, 0.0). The message says the number. When `ret_5d` cannot be computed the finding is a warning `trend_unconfirmed_unknown` (no denial). No other family is touched. |
| C6 | **The same quote gets the same label.** The Critic's model labels a claim once: a label is kept by (model id, prompt version, the claim's quote, the source text's hash, the control result) in a small store (`evidence.label_store`, config; a file under `data/critic/labels/`, gitignored; null turns it off) and read back the next time. A verdict no longer moves because the same model was asked the same thing twice. The control (the planted unrelated text) still runs on every model call. |
| C7 | **The Critic still does not size, does not see the portfolio, imports nothing of the Advisor's scoring code, and has no liquidity, stop-distance or market-cap rule** (entries 92, 93, 97, 105). C5 reads the record's family name and the Critic's own metric. No threshold is added that is not in `configs/critic_v1.json`. |

## 4. The Critic, module by module

- `critic/evidence.py`: C2 (`requires_human=False` on `fresh_negative_news`), C3 (`blend_not_supported` through `warning(...)`), C6 (the label store: `critic/labels.py`, new; read before the model is asked, written after; the claims the store answers are not sent to the model, and a call with nothing left to ask is not made). The reply's audit says how many labels came from the store.
- `critic/recheck.py`: C4. The comparison is a small pure function in the Critic's own package (`critic/names.py`, new): it does NOT call `advisor.names_company` for this any more (that function is about a quote naming a company and stays as it is for the quotes).
- `critic/chart.py` or `critic/timing.py` (where the worker finds the family and the re-asked metrics together; say which and why): C5.
- `configs/critic_v1.json`: `chart.trend_confirm_ret_5d: 0.0`, `evidence.label_store: "data/critic/labels"`, `config_version` "2". `critic/config.py` accordingly. `integrity.GATES` is unchanged.
- `critic/report.py` and the page's review view (`web/views.py` `_review_part`) need no shape change: a blocker and a warning already have their places. Check the wording of the summary for a review that is now approved WITH a `blend_not_supported` warning.
- The Advisor is not changed by this section.

## 5. Tests (real data; the purge rule holds)

- Real captured reviews re-run through the changed code with the captured model replies (how `tests/test_critic*.py` do it): a real record that was denied only for `fresh_negative_news` is approved and carries the warning; a real `fresh_negative_event` is still denied; a real record denied only for `blend_not_supported` is approved and carries the warning and the lower evidence points; the real XOM and APA names match and a real different company does not; a real trend record with a negative `ret_5d` on its decision day is denied `trend_not_confirmed`, the same record on a day with a positive one is not, a conviction record with a negative one is not.
- The label store: the same claim twice asks the model once (count the calls on the captured-reply model); another model id, another source text or a failed control asks again; with the store off nothing is kept.
- Every existing Critic test: an expected value that changes because of C2, C3 or C4 is changed and LISTED in the report (which test, which value, why). No test is deleted.

## 6. The report card (`prometheus/proving/report_card.py`)

A program that scores every finding of the Critic against the outcomes, so that C1 is a routine and not a memory.

    report = report_card.build(runs=None)        # every saved Time Machine run and the Track Record; or the runs given
    ./test_proving.sh critic-report [--run RUN_ID ...] [--save FILE]          free

- One row per finding code (blocker, warning, human reason): `n` fired, of those closed and won, mean excess of the hit group and of the rest, the gap, its standard error, `inside_chance` (the same two-standard-error band as `stats.CHANCE_BAND`; move the helper to one place and use it in both), `kind` (blocker | warning | human), `suitability` (true for `risk_above_profile`: listed, never judged by returns), and per theme the same numbers (the theme is the run's own request, by run id; the report card does not guess a sector).
- The verdict's own row: approved against denied without the suitability denials.
- `by_family` and `by_config_version` (the Critic's `config_version` on each review), so v1 and v2 reviews are never mixed in one row.
- A recommendation that is in several runs (the same record id) counts once.
- A sentence per row from a closed set of templates: "denies winners", "picks losers", "no clear difference", "too few to say" (`min_sample`).
- A route `GET /api/proving/critic-report` and a card on the Track Record tab: one row per rule, a dot for the gap with its two-standard-error bar on a common axis, zero marked, the suitability rows greyed, a table twin. The colour rule of the page holds: green only for a rule that picks losers beyond chance, red only for one that denies winners beyond chance, grey inside chance.
- The offline evaluation's `prometheus-outcomes` dataset gets the finding codes of each example, and an evaluator `critic_rule_value` (per code, the gap). `--dry-run` only.

## 7. An exit plan for a neutral purchase (entry 134)

Problem: a neutral company has no plan, so in a test its position never exits and the cash stays in it (the $10,000 of the oil test was spent by mid-July), and in the Live view it shows "no plan".

- The Advisor writes, for every NEUTRAL candidate it scores, an `exit_plan`: `{stop_price, time_stop_date, basis}`. The stop is `neutral.stop_atr_multiple` (config `configs/advisor_v1.json`, 2.0) times the 14-day ATR under the last close; the time stop is `neutral.horizon_bars` (20) trading days after the decision day. No target: nothing in the record says where a neutral company should go. When the ATR cannot be computed there is no stop and the time stop alone stands.
- It is an EXIT plan and nothing else: the Trader's tickets, the entry zone, the limit range and the order type of a neutral purchase are what they are today (a test compares the tickets before and after). The Advisor still writes only recommendations to buy and is still given no portfolio (entry 105).
- Who reads it: `proving/` (`record.plan_of`, `SimPortfolio`, `outcomes.score`: a neutral position is followed by its stop and its time stop, by the same `simulator.follow`) and the Live view (stop and time stop on the ladder; the note "bought as a company the Advisor scored neutral" stays, with "it has a stop and a time stop, no target").
- Nothing is sold in a live account by it (P11 and entry 105 are unchanged).
- The two numbers are conventions, not findings. The report says what the oil and the AI test do with them and without (the cash recycled, the neutral outcomes' mean), and the config makes them Jacob's to change.

## 8. The cooldown's replacement (entry 135)

Problem: the engine takes a cooled-down recommendation off the order and puts nothing in its place, so a weekly test of one theme trades fewer than the number asked for (the cooldown caught 46 of 91 recommendations in the oil test). The live Advisor, which ranks WITH its ledger, would have moved on to the next company.

- Phase 1 asks the Advisor for `count + time_machine.spare_recommendations` (config, 3) recommendations a day (never more than the Advisor's own limit).
- In the walk the engine takes them in rank order, skips what `ranking.cooldown_of` catches, and keeps the first `count`. The ones beyond the count get `not_traded_reason: "beyond_count"` and a hypothetical outcome, as the neutral ones beyond the number do today.
- The Critic reviews all of them in phase 1 (they are in the order), so nothing is reviewed in the walk and the walk still calls no model.
- The cache key holds the spare number. The request's own `count` is what the user typed, unchanged.

## 9. Proof (paid: the models; the standing yes covers it; never a fetch)

1. The four six-month tests again, fast, with Critic v2 and sections 7 and 8 (oil, AI, restaurants, consumer staples; the prompts and ranges of `data/proving/runs/` runs `tm-20261120-222822-1221`, `-232020-db7f`, `-232408-013c`, `-232616-e1c6`), each saved, and the report card over them, beside the report card over the four v1 runs. The table of section 2 again for v2.
2. STABILITY: the oil test twice more, cold (`--no-cache`), and the count of recommendations in common whose verdict differs, beside the 23% of v1. And the Critic alone twice on the SAME cached research of the oil test with the label store off and then on (the Critic's own noise, separated from the Data Clerk's and the Advisor's).
3. One live session on a throwaway account (`--cleanup`), Auto, the oil request, no fetch, to see a v2 review on live quotes; its capture saved to `tests/fixtures/session_real/`.
4. What the report must say plainly: whether approved beat denied in each theme and by how much against chance; that the new rule was chosen on this same window; anything that got WORSE.

## 10. Order of work, and the documents

Sections 3 to 5 first (the Critic and its tests), then 6, then 7 and 8, then 9. The design advisor writes entries 133 to 135, `AGENTS.md`, `HowToRun.txt`, the amendment of `design/critic_v1_final_spec.md` and the session notes from the worker's report and its measured numbers; the worker writes docstrings in the house style and no other document.

## 11. Open, for Jacob

1. The Advisor's TREND family is the weak one in three of four themes (60% of its closed recommendations won against 93% for conviction, with a stop about three times as far as its target). C5 removes its worst half at the Critic. Whether the family's own conditions or its plan should change is an Advisor question, not built here.
2. `overpriced` and `valuation_extreme` deny winners in the AI theme. Left alone: one theme.
3. The Advisor's highest scores did worst in the two themes that have any (75 and over: -0.9 and 0.0 against +1.4 and +1.1). A calibration question for the Advisor's blend.
