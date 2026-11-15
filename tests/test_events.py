"""Tests for prometheus.agents.advisor.events (Advisor v1 spec section 8.4, phase 3, design choices entry 51).

The table of event classes is tested against REAL sentences copied from the stored text (the file is named beside each), and against near misses that were also real and were
matching before the patterns were narrowed. A class for which the stored text has no example (the stored news is a few weeks of eight companies: it has no restatement or going
concern) is tested with one plain sentence of the pattern, marked CANONICAL; those say what the pattern means, they are not market data.
"""
import pytest

from prometheus.agents.advisor import events as ev

# (class, sentence, where it was copied from)
REAL = [
    ("guidance_raise", "Procter & Gamble (PG) lifted its full-year sales growth outlook to reflect an acquisition, while the", "news/PG.txt"),
    ("guidance_raise", "Raised free cash flow guidance while maintaining revenue growth guidance, indicating solid and reliable cash flow generation from high-margin software.", "news/TSLA.txt"),
    ("earnings_beat", "Fiscal Q4 EPS beat Zacks Consensus Estimate by 14.63%, revenue surged 371.6% year-over-year.", "news/SNDK.txt"),
    ("earnings_beat", "Tesla reported Q3 2026 earnings beating Wall Street expectations with EPS of $0.50 vs.", "news/TSLA.txt"),
    ("buyback_authorization", "General Motors has announced a $20 billion share repurchase authorization and has reduced outstanding shares by nearly 40% since early 2023.", "news/F.txt"),
    ("buyback_authorization", "Nvidia has authorized a historic $235 billion stock buyback while continuing aggressive AI ecosystem investments.", "news/NVDA.txt"),
    ("guidance_cut", "Despite beating earnings with 30% YoY revenue growth and achieving profitability, stock dropped 20% due to lowered full-year guidance, major slowdown in Germany.", "news/NVDA.txt"),
    ("earnings_miss", "Automotive manufacturer Ford (NYSE:F) missed Wall Street\u2019s revenue expectations in Q3 2026, with revenue falling 2% year on year to $49.53 billion.", "news/F.txt"),
    ("analyst_rating_change", "SanDisk Corporation has been upgraded to a Zacks Rank #1 (Strong Buy) rating, reflecting upward trends in earnings estimates over the past three months.", "news/SNDK.txt"),
    ("sector_sympathy", "As a neocloud operator, it is exposed to the predicted industry-wide consolidation and insufficient revenue generation to cover capital expenditures.", "news/MSFT.txt"),
    ("macro", "Stock futures pointed lower as oil prices climbed above $114 a barrel, but some of the hard-hit tech names were enjoying a modest rebound.", "news/SNDK.txt"),
    ("lawsuit_headline", "Ford Motor (F) is back under the legal spotlight after a federal judge certified state classes in a long running lawsuit over CP4 fuel pumps used in certain 6.7L diesel trucks.", "news/F.txt"),
]
CANONICAL = [
    ("dividend_increase", "Procter & Gamble raises its quarterly dividend 5% to $1.06 per share."),
    ("dividend_initiation", "The company announced that it will initiate a quarterly dividend of $0.10 per share."),
    ("dividend_cut", "The board cut the quarterly dividend by half to preserve cash."),
    ("dilution_offering", "The company announced the pricing of a public offering of 10,000,000 shares of its common stock."),
    ("ceo_departure_unplanned", "The company announced that its chief executive officer resigned effective immediately."),
    ("material_weakness", "Management identified a material weakness in the company's internal control over financial reporting."),
    ("restatement", "The company will restate its previously issued financial statements for fiscal 2025."),
    ("going_concern", "These conditions raise substantial doubt about the company's ability to continue as a going concern."),
    ("index_flow", "Sandisk will be added to the S&P 500 index effective before the open on Monday."),
]
NEAR_MISSES = [  # sentences that matched before the patterns were narrowed, or that a looser pattern would take
    ("earnings_beat", "The author expects the Oct. 29 earnings report to exceed expectations and drive stock appreciation.", "a prediction, not a result"),
    ("guidance_raise", "Procter & Gamble (NYSE:PG) reported a 4.0% increase in organic sales for the first quarter of fiscal 2027, led primarily by volume growth, while maintaining its full-year outlook.", "'increase' is of sales"),
    ("guidance_raise", "Company did not raise guidance due to weakness in Vans brand, and shares fell 3% in early trading.", "negated"),
    ("guidance_cut", "Ford reported third quarter earnings that topped estimates, and upped the lower end of its profit guidance.", "the LOWER END was raised"),
    ("guidance_cut", "Karooooo's shares fell 8% despite strong subscriber growth due to lower-than-expected earnings and unchanged guidance.", "'lower-than-expected' and unchanged"),
    ("earnings_miss", "Cybertruck sales fell 92% short of estimates, Optimus robot integration lagged expectations.", "a person's product forecasts, not a result"),
    ("dividend_increase", "The company has paid dividends for 136 years and raised them for 70 consecutive years, earning Dividend King status.", "a history, not a new raise"),
    ("dividend_increase", "The Company intends to increase its dividend on an annual basis, subject to declaration by the Board of Directors.", "an intention (every 10-Q has it)"),
    ("dividend_initiation", "We currently anticipate that our available funds will fund our investments in infrastructure and AI initiatives, subject to a dividend.", "'initiatives' is not 'initiate'"),
    ("dividend_cut", "We may be required to reduce or delay investments and capital expenditures, suspend dividend payments, seek additional capital, or restructure.", "a risk factor: what may happen"),
    ("dilution_offering", "SK Hynix's decision to scrap plans for a U.S. initial public offering of its Solidigm business could benefit Micron Technology.", "an IPO that was cancelled, of another company"),
    ("restatement", "Apple Inc. Non-Employee Director Stock Plan, as Amended and Restated was approved.", "a plan, not the financial statements"),
    ("buyback_authorization", "As of the end of the quarter the Company has an authorized share repurchase program, which does not obligate the Company to acquire any amount.", "no amount, an existing programme"),
    ("sector_sympathy", "While the dividend yield of 0.6% is modest compared to competitors, the article argues that Costco has outperformed.", "a comparison, not a move with a peer"),
]


@pytest.mark.parametrize("name, sentence, where", REAL)
def test_every_class_with_a_real_example_matches_it_and_the_direction_is_the_tables(name, sentence, where):
    found = {m["class"]: m for m in ev.match_events(sentence)}
    assert name in found, f"{name} did not match a real sentence of {where}"
    assert found[name]["direction"] == ev.EVENT_CLASSES[name].direction


@pytest.mark.parametrize("name, sentence", CANONICAL)
def test_the_classes_the_stored_text_has_no_example_of_match_their_canonical_sentence(name, sentence):
    assert name in {m["class"] for m in ev.match_events(sentence)}


@pytest.mark.parametrize("name, sentence, why", NEAR_MISSES)
def test_a_near_miss_does_not_match_its_class(name, sentence, why):
    assert name not in {m["class"] for m in ev.match_events(sentence)}, why


def test_every_text_class_is_covered_by_an_example_and_the_directions_are_fixed():
    covered = {n for n, _, _ in REAL} | {n for n, _ in CANONICAL}
    assert covered == set(ev.TEXT_CLASSES)
    assert set(ev.BULLISH) == {"guidance_raise", "earnings_beat", "buyback_authorization", "dividend_increase", "dividend_initiation", "activist_stake_13d"}
    assert set(ev.BEARISH) == {"guidance_cut", "earnings_miss", "dividend_cut", "dilution_offering", "ceo_departure_unplanned", "material_weakness", "restatement", "going_concern", "form_144_sale_notice"}
    assert set(ev.NEUTRAL) == {"analyst_rating_change", "sector_sympathy", "macro", "lawsuit_headline", "index_flow"}
    assert all(ev.EVENT_CLASSES[n].horizon == 40 for n in (*ev.BULLISH, *ev.BEARISH)) and all(ev.EVENT_CLASSES[n].horizon is None for n in ev.NEUTRAL)


# ---------------------------------------------------------------- verify_event: the model's claim, checked

SOURCES = {1: {"number": 1, "text": "Procter & Gamble (PG) lifted its full-year sales growth outlook to reflect an acquisition, while the company reported Q1 earnings that exceeded analyst expectations."},
           2: {"number": 2, "text": "Micron lifted its outlook. Sandisk shares fell 4% on Tuesday."}}
GOOD = {"class": "guidance_raise", "source": 1, "quote": "Procter & Gamble (PG) lifted its full-year sales growth outlook to reflect an acquisition"}


def check(event, **kw):
    return ev.verify_event(event, kw.get("symbol", "PG"), kw.get("company", "Procter & Gamble"), kw.get("sources", SOURCES))


def test_a_true_event_is_accepted():
    assert check(GOOD) == (True, None)
    assert check({**GOOD, "source": "1", "direction": "bullish"}) == (True, None)           # the source as text, the direction as the class has it
    assert check({"class": "earnings_beat", "source": 1, "quote": "reported Q1 earnings that exceeded analyst expectations"}, company="Procter & Gamble")[0] is False  # no company name in the quote


@pytest.mark.parametrize("change, message", [
    ({"class": "moon_landing"}, "is not an event class"),
    ({"class": "form_144_sale_notice"}, "comes from the filing index"),
    ({"source": 9}, "is not one of the sources shown"),
    ({"source": True}, "is not one of the sources shown"),
    ({"quote": "lifted its"}, "too short"),
    ({"quote": "Procter & Gamble (PG) cut its full-year sales growth outlook to reflect a recession"}, "is not in source 1"),
    ({"direction": "bearish"}, "contradicts the class"),
    ({"class": "guidance_cut"}, "does not show guidance_cut"),
])
def test_a_claim_that_fails_one_check_is_refused_with_its_reason(change, message):
    ok, reason = check({**GOOD, **change})
    assert ok is False and message in reason


# ---------------------------------------------------------------- the filing index, amounts, earnings dates, timing

def test_form_144_and_schedule_13d_come_from_the_filing_index_without_any_quote_from_a_model():
    sources = [
        {"number": 4, "tickers": ["PG"], "published_utc": "2026-10-20T00:00:00Z", "text": "Form 144 (notice of a planned sale of restricted shares by an insider (insider selling)) filed 2026-10-20. The text of this filing is not stored: only its form, date and link are known."},
        {"number": 5, "tickers": ["PG"], "published_utc": "2026-10-01T00:00:00Z", "text": "Form SCHEDULE 13D (report by an investor who holds more than 5% of the shares and may seek influence) filed 2026-10-01. The text of this filing is not stored."},
        {"number": 6, "tickers": ["PG"], "published_utc": "2026-10-02T00:00:00Z", "text": "Form SCHEDULE 13D/A (amended) filed 2026-10-02."},
        {"number": 7, "tickers": ["MU"], "published_utc": "2026-10-03T00:00:00Z", "text": "Form 144 filed 2026-10-03."},
        {"number": 8, "tickers": ["PG"], "published_utc": "2026-10-04T00:00:00Z", "text": "Procter & Gamble filed a Form 144 shelf. Not a record start."},
    ]
    found = ev.filing_index_events("PG", sources)
    assert [(e["class"], e["source"], e["direction"], e["origin"]) for e in found] == [("form_144_sale_notice", 4, "bearish", "filing_index"), ("activist_stake_13d", 5, "bullish", "filing_index")]
    assert found[0]["published_utc"] == "2026-10-20T00:00:00Z" and found[0]["quote"].startswith("Form 144")


def test_the_dollar_amount_of_a_quote():
    assert ev.amount_of("announced a $20 billion share repurchase authorization") == 20e9
    assert ev.amount_of("a $1,500 million buyback") == 1.5e9 and ev.amount_of("a $2.5 billion programme") == 2.5e9
    assert ev.amount_of("no amount here") is None and ev.amount_of(None) is None


def test_the_last_earnings_release_is_the_newest_results_press_release_of_the_symbol():
    sources = [{"tickers": ["PG"], "headline": "Procter & Gamble Announces Quarterly Results", "text": "", "published_utc": "2026-07-29T11:00:00Z"},
               {"tickers": ["PG"], "headline": "Procter & Gamble Announces Fiscal Year Results", "text": "", "published_utc": "2026-10-24T11:00:00Z"},
               {"tickers": ["PG"], "headline": "P&G launches a new detergent", "text": "", "published_utc": "2026-10-25T11:00:00Z"},
               {"tickers": ["MU"], "headline": "Micron Reports Results", "text": "", "published_utc": "2026-10-26T11:00:00Z"}]
    assert ev.last_earnings_release("PG", sources) == "2026-10-24" and ev.last_earnings_release("SNDK", sources) is None


RECENT = [["2026-10-19", 100.0, 1000.0], ["2026-10-20", 100.0, 1000.0], ["2026-10-21", 100.0, 1000.0], ["2026-10-22", 100.0, 1000.0],
          ["2026-10-23", 110.0, 3000.0], ["2026-10-26", 115.0, 1500.0], ["2026-10-27", 120.0, 1000.0]]


def test_timing_dates_an_event_against_the_bars_and_measures_its_reaction():
    t = ev.timing("2026-10-23T09:00:00Z", RECENT)
    assert t["age_bars"] == 2 and t["bar_date"] == "2026-10-23"                                  # the event bar is 10-23: two bars came after it
    assert t["reaction"] == pytest.approx(120 / 100 - 1)                                         # the last close over the close BEFORE that day (10-22)
    assert t["event_volume_ratio"] == pytest.approx(3000 / 1000)                                 # the event bar's volume over the mean of the bars before it
    saturday = ev.timing("2026-10-24T09:00:00Z", RECENT)                                         # a weekend news item: the first bar on or after is Monday's
    assert saturday["bar_date"] == "2026-10-26" and saturday["age_bars"] == 1 and saturday["reaction"] == pytest.approx(120 / 110 - 1)  # event bar Monday: one bar after it
    after = ev.timing("2026-10-28T09:00:00Z", RECENT)                                            # published after the last bar: no reaction yet
    assert after["age_bars"] == 0 and after["bar_date"] is None and after["reaction"] == pytest.approx(120 / 120 - 1)
    assert ev.timing(None, RECENT) is None and ev.timing("2026-10-23", RECENT[:1]) is None


def test_timing_has_no_reaction_when_the_event_is_older_than_the_bars():
    t = ev.timing("2026-09-01T00:00:00Z", RECENT)
    assert t["reaction"] is None and t["age_bars"] == len(RECENT)
