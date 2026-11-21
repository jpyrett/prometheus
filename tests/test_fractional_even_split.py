"""The EVEN split in fractional shares: "split $30,000 evenly over three stocks" buys $10,000.00 of each (design choices entry 110; design/fractional_shares_spec.md section 4).

REAL DATA ONLY: the prices are the real prices of the real sessions in `tests/fixtures/session_real/` (what each decision was sized at), and the real oil session (`auto_oil_no_second_mark.json`) is decided again by the real
Account Manager and traded by the real Trader (the stored closes of Friday 2026-11-13, `handshake_helpers.py`). The whole-share split of entry 108 (Part G) is the fallback and its own tests stay as they were: here is
the proof that the switch off gives exactly it. The connected-book tests use the real local Postgres on throwaway accounts (skipped when it is not reachable).
"""
import glob
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest
from handshake_helpers import FRIDAY, MOMENT, close_of, friday_close, trader  # noqa: F401  (friday_close is a fixture)
from test_neutral_recommendation import OIL_CASH, OIL_USER, account, allowed, oil_order  # noqa: F401  (allowed is a fixture)
from test_trader_handshake import YEAR, db, store  # noqa: F401  (db and store are fixtures)

from prometheus.agents.account_manager import sizing as sz
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.money import money, price

pytestmark = pytest.mark.usefixtures("friday_close")

CAPTURES = Path(__file__).parent / "fixtures" / "session_real"


def real_prices() -> dict[str, Decimal]:
    """{symbol: the price a real decision was sized at}, from every real session capture."""
    found = {}
    for name in sorted(glob.glob(str(CAPTURES / "*.json"))):
        for d in (json.loads(Path(name).read_text(encoding="utf-8")).get("order") or {}).get("trade_decisions", []):
            if d.get("shares") and d.get("est_cost"):
                found[d["symbol"]] = price(d["price"]) if d.get("price") else price((money(d["est_cost"]) / sh.quantity(d["shares"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))      # the capture's floats go through the doors of money and shares
    return found


PRICES = real_prices()


def purchases(*symbols: str) -> list[sz.Purchase]:
    return [sz.Purchase(record_id=f"rec-{s}", symbol=s, action="open", price=PRICES[s], rank=n) for n, s in enumerate(symbols, 1)]


def size(symbols, pot, **options):
    return sz.size_purchases(purchases(*symbols), pot, allocation="even", fractional=True, **options)


# ---------------------------------------------------------------- the account book decides whether a fraction may be bought

def test_an_account_book_that_cannot_hold_a_fraction_gets_the_whole_share_split_and_the_user_is_told_why(db, monkeypatch, allowed):
    account_id = db.account("15000.00", name=OIL_USER.user)
    told = []
    holds = db.store.holds_fractions()
    am = account(profile=OIL_USER, screen=told.append, fractional_even_split=None, store=db.store, account_id=account_id)
    if holds:                                                                                        # a migrated book: fractions, saved exactly
        out = am.decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", "even", None, count=3)
        assert [d["shares"] for d in out.trade_decisions if d["will_trade"] == "yes_auto"] == [Decimal("31.581606"), Decimal("22.738642"), Decimal("100.502512")] and not [t for t in told if "whole shares" in t]
        assert db.store.number_approved_recommendations(account_id, *YEAR) == 3
    monkeypatch.setattr(db.store, "holds_fractions", lambda: False)                                   # the book as it was before `bash scripts/setup_postgres.sh`: the real method, answered the other way
    other = db.account("15000.00", name=OIL_USER.user)
    told.clear()
    old = account(profile=OIL_USER, screen=told.append, fractional_even_split=None, store=db.store, account_id=other)
    out = old.decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", "even", None, count=3)
    assert [(d["shares"], d["est_cost"]) for d in out.trade_decisions if d["will_trade"] == "yes_auto"] == [(31, Decimal("4907.92")), (23, Decimal("5057.47")), (101, Decimal("5024.75"))]
    assert len([t for t in told if "the split is in whole shares" in t and "setup_postgres.sh" in t]) == 1 and out.errors == []
