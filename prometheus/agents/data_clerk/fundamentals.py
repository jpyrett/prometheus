"""Ratios worked out from the Finnhub payload and from the filing evidence (Advisor v1 spec section 5.4 and 5.5, design choices entry 51).

Plain Python: numbers in, numbers out. No file, no network, no model. `filing_signals.py` stores FIGURES and QUOTES and scores nothing; the derived numbers live here, each one a
function of the evidence that returns `(value, inputs_summary)` or raises `Missing(reason)`. Missing is not a failure: the metrics catalogue turns it into `value: None` with the
reason as the `note`. A number that cannot be computed is never replaced by a guess.

The evidence is the result of `filing_signals.recommendation_evidence` for one company. Three things to know about its figures:
  * a 10-Q's cash-flow rows are YEAR TO DATE and the evidence says for how many months (`period_months`); a 10-Q's income rows have a three-month and a year-to-date column
    (four values) or only one column (two values): `ytd` picks the year-to-date pair;
  * every row carries the `units` line of its statement ("(In millions, except ...)"): `multiplier` turns it into 1, thousand, million or billion;
  * a cash outflow is negative as the filing shows it (repurchases, dividends paid); proceeds from issuing stock are positive.
"""
import re
from datetime import date, timedelta

from prometheus.agents.data_clerk.filing_signals import change

DILUTION_WINDOW_DAYS = 120   # an 8-K that discusses dilution counts as current for this long (spec 5.5)


class Missing(Exception):
    """A derived metric cannot be computed; the message says what is missing, in words."""


def multiplier(units: str | None) -> float:
    """The factor a statement's units line gives: "(In millions, ...)" 1e6, "(In thousands ...)" 1e3, "(In billions ...)" 1e9, else 1 (dollars)."""
    found = re.search(r"\bin (millions|thousands|billions)\b", units or "", re.I)
    return {"millions": 1e6, "thousands": 1e3, "billions": 1e9}[found.group(1).lower()] if found else 1.0


def ytd(values: list[float]) -> tuple[float, float] | None:
    """(this year to date, last year to date) from an income row: four values (three months and year to date, this year and last) -> the last pair; two values (a first quarter, or a
    10-K) -> the pair. Anything else -> None."""
    if len(values) == 4:
        return values[2], values[3]
    if len(values) == 2:
        return values[0], values[1]
    return None


def _action(evidence: dict, item: str) -> dict | None:
    return next((a for a in evidence.get("capital_actions", []) if a["item"] == item and a.get("current") is not None), None)


def _income(evidence: dict, item: str) -> tuple[float, float]:
    """(this year to date, last year to date) of an income headline, in DOLLARS. Raises Missing."""
    row = next((i for i in evidence.get("income", []) if i["item"] == item), None)
    pair = ytd(row["values"]) if row else None
    if pair is None:
        raise Missing(f"no {item} figure for the year to date in the stored filings")
    factor = multiplier(row.get("units"))
    return pair[0] * factor, pair[1] * factor


def _dollars(action: dict) -> float:
    return action["current"] * multiplier(action.get("units"))


def _balance(evidence: dict, key: str) -> dict:
    sheet = evidence.get("balance_sheet")
    item = sheet and sheet["items"].get(key)
    if not item or item.get("current") is None:
        raise Missing(f"no {key.replace('_', ' ')} on the stored balance sheet")
    return item


# ---------------------------------------------------------------- from the Finnhub payload

def earnings_yield(pe_ratio: float | None) -> tuple[float, dict]:
    """`1 / pe_ratio` when the P/E is positive."""
    if pe_ratio is None:
        raise Missing("Finnhub gave no P/E ratio")
    if pe_ratio <= 0:
        raise Missing(f"the P/E ratio is {pe_ratio:g} (not positive): the company has no earnings yield")
    return 1 / pe_ratio, {"pe_ratio": pe_ratio}


# ---------------------------------------------------------------- from the filing evidence

def net_payout_yield_ttm(evidence: dict, market_cap: float | None) -> tuple[float, dict]:
    """`(abs(repurchases) + abs(dividends paid) - proceeds from issuing common stock) / market_cap`, the year-to-date cash flows annualized by `12 / months`.
    At least one of repurchases and dividends must be a row of the statement; a row that is absent counts as zero and is named in `absent` (the statement leaves out a line that is
    nothing). Missing: no cash-flow rows, the months of the period unknown, or no market capitalization."""
    months = evidence.get("period_months")
    repurchases, dividends, proceeds = _action(evidence, "repurchases of common stock"), _action(evidence, "dividends paid"), _action(evidence, "proceeds from issuing common stock")
    if repurchases is None and dividends is None:
        raise Missing("no repurchases or dividends row in the stored cash-flow statement")
    if not months:
        raise Missing("the number of months the year-to-date cash flow covers is not stated")
    if not market_cap or market_cap <= 0:
        raise Missing("no market capitalization from Finnhub")
    bought, paid, issued = (abs(_dollars(repurchases)) if repurchases else 0.0, abs(_dollars(dividends)) if dividends else 0.0, _dollars(proceeds) if proceeds else 0.0)
    value = (bought + paid - issued) * 12 / months / market_cap
    return value, {"repurchases": bought, "dividends": paid, "issued": issued, "months": months, "market_cap": market_cap,
                   "absent": [n for n, a in (("repurchases", repurchases), ("dividends", dividends), ("proceeds from issuing common stock", proceeds)) if a is None]}


def share_count_change(evidence: dict) -> tuple[float, dict]:
    """The change of the total shares issued and outstanding between the two balance sheets, as a fraction (`change_pct / 100`): -0.01 is one percent fewer shares."""
    sheet = evidence.get("balance_sheet")
    shares = sheet and sheet.get("shares")
    if not shares or shares.get("change_pct") is None:
        raise Missing("no shares issued and outstanding on the stored balance sheet")
    return shares["change_pct"] / 100, {"current": shares["current"], "prior": shares["prior"], "unit": shares["unit"], "columns": sheet["columns"]}


def payout_coverage(evidence: dict) -> tuple[float, dict]:
    """`abs(dividends paid) / net income`, both for the year to date. Missing when net income is not positive (the ratio would say nothing)."""
    dividends = _action(evidence, "dividends paid")
    if dividends is None:
        raise Missing("no dividends row in the stored cash-flow statement")
    income = _income(evidence, "net income")[0]
    if income <= 0:
        raise Missing("net income for the year to date is not positive")
    paid = abs(_dollars(dividends))
    return paid / income, {"dividends": paid, "net_income": income}


def sbc_to_revenue(evidence: dict) -> tuple[float, dict]:
    """`share-based compensation expense / revenue`, both for the year to date."""
    sbc = _action(evidence, "share-based compensation expense")
    if sbc is None:
        raise Missing("no share-based compensation row in the stored cash-flow statement")
    revenue = _income(evidence, "revenue")[0]
    if revenue <= 0:
        raise Missing("revenue for the year to date is not positive")
    return _dollars(sbc) / revenue, {"sbc": _dollars(sbc), "revenue": revenue}


def _net_income_and_cfo(evidence: dict) -> tuple[float, float]:
    cfo = _action(evidence, "cash from operations")
    if cfo is None:
        raise Missing("no cash from operations row in the stored cash-flow statement")
    return _income(evidence, "net income")[0], _dollars(cfo)


def accrual_ratio(evidence: dict) -> tuple[float, dict]:
    """`(net income - cash from operations) / total assets`, the income and cash flow for the year to date, the assets of the latest balance sheet."""
    income, cfo = _net_income_and_cfo(evidence)
    assets = _balance(evidence, "total_assets")
    total = assets["current"] * multiplier(evidence["balance_sheet"].get("units"))
    if total <= 0:
        raise Missing("total assets are not positive")
    return (income - cfo) / total, {"net_income": income, "cash_from_operations": cfo, "total_assets": total}


def cfo_negative_with_profit(evidence: dict) -> tuple[bool, dict]:
    """True when cash from operations is negative while net income is positive (the year to date)."""
    income, cfo = _net_income_and_cfo(evidence)
    return cfo < 0 < income, {"net_income": income, "cash_from_operations": cfo}


def negative_equity(evidence: dict) -> tuple[bool, dict]:
    """True when the total stockholders' equity of the latest stored balance sheet is below zero (a stockholders' DEFICIT: Domino's, Papa John's, Yum, McDonald's after years of buybacks). Debt
    divided by such an equity is not a leverage ratio, so the Advisor does not apply its debt-to-equity rules to the company (design choices entry 54). The sign is all that is read, so the units do not matter."""
    item = _balance(evidence, "total_equity")
    return item["current"] < 0, {"total_equity_as_reported": item["current"], "label": item.get("label"), "units": evidence["balance_sheet"].get("units")}


def _against_sales(evidence: dict, key: str) -> tuple[float, dict]:
    """The percentage change of a balance-sheet item (latest column against the earlier one) minus the percentage change of year-to-date revenue against last year's, in points."""
    item = _balance(evidence, key)
    if item.get("change_pct") is None:
        raise Missing(f"the {key.replace('_', ' ')} of the earlier balance sheet is zero or missing")
    revenue, prior = _income(evidence, "revenue")
    if prior <= 0:
        raise Missing("last year's revenue for the same period is not positive")
    sales_pct = (revenue - prior) / prior * 100
    return item["change_pct"] - sales_pct, {f"{key}_change_pct": item["change_pct"], "revenue_change_pct": round(sales_pct, 2)}


def receivables_vs_sales(evidence: dict) -> tuple[float, dict]:
    return _against_sales(evidence, "accounts_receivable")


def inventory_vs_sales(evidence: dict) -> tuple[float, dict]:
    return _against_sales(evidence, "inventories")


def cash_balance(evidence: dict) -> tuple[float, dict]:
    """The cash and cash equivalents of the latest stored balance sheet, in dollars (not the marketable securities): what a buyback authorization is compared with."""
    item = _balance(evidence, "cash")
    return item["current"] * multiplier(evidence["balance_sheet"].get("units")), {"label": item.get("label"), "as_reported": item["current"], "units": evidence["balance_sheet"].get("units")}


def dilution_current(evidence: dict, today: date) -> tuple[bool, dict]:
    """True when the company is issuing stock NOW, from what the stored filings show: (a) the total shares outstanding rose between the two balance sheets, (b) a cash-flow row of
    proceeds from an ISSUANCE of common or class A/B stock is not zero (not the exercise of employee options: every company has those), or (c) an 8-K filed within 120 days of
    `today` has a passage about dilution (an offering, an at-the-market programme, convertible notes). A dilution passage in a 10-Q does not count: every 10-Q has one (the
    anti-dilutive securities note), so it would mark every company. Spec 5.5 says "an 8-K or a 10-Q"; see design choices entry 51 for why 10-Q passages are left out.
    Missing when there is no evidence at all to decide from."""
    reasons = []
    sheet = evidence.get("balance_sheet")
    shares = sheet and sheet.get("shares")
    if shares and shares.get("change") is not None and shares["change"] > 0:
        reasons.append(f"shares outstanding rose {shares['change_pct']:+.2f}%")
    for action in evidence.get("capital_actions", []):
        if action["item"] == "proceeds from issuing common stock" and action.get("current") and re.search(r"issuance of (?:common|class [ab])", action.get("label", ""), re.I):
            reasons.append(f"proceeds from issuing stock: {action['label']}")
    cutoff = (today - timedelta(days=DILUTION_WINDOW_DAYS)).isoformat()
    for item in evidence.get("evidence", []):
        source = item["source"]
        if item["topic"] == "dilution" and not item.get("derived") and source["form"] == "8-K" and source["filed"] >= cutoff:
            reasons.append(f"8-K filed {source['filed']} mentions '{item.get('matched', '')}' [{item['id']}]")
    if not reasons and not shares and not evidence.get("evidence") and not evidence.get("capital_actions"):
        raise Missing("no stored filing evidence to decide from")
    return bool(reasons), {"reasons": reasons[:5]}
