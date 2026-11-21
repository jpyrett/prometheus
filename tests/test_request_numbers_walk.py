"""The Proving Ground replays a request as typed, so a price limit is honoured by the Time Machine with no code of its own (design choices entry 138, L6): the walk of phase 2 hands the limit the cached research was made under to the Account Manager's
decision. On the REAL research of nine weekly decision days of a real fast Time Machine run (`tests/fixtures/critic_v2_real/`, the oil request, cash 10000; FREE, no model is called): the same research walked with no limit and with a limit of $160 a share.
(The research itself was made with no limit: what the Advisor leaves out for its price is tested on the Advisor's real captures in `tests/test_request_numbers.py`.)"""
from decimal import Decimal

from test_proving_spare import no_model, put_research, walk  # noqa: F401  (no_model is a fixture)


def traded(run) -> dict:
    return {(o.decision_day, o.symbol): o for o in run.outcomes if o.traded}      # by the day and the company (the number of a neutral purchase's key follows the order of the day's purchases)


def test_the_walk_hands_the_limit_to_the_account_managers_decision_and_nothing_is_bought_above_it(tmp_path, no_model):
    free = walk(put_research(tmp_path / "free"))
    limited = walk(put_research(tmp_path / "limited", mutate=lambda entry: entry["plan"].__setitem__("max_price", 160)))      # the plan of each cached day carries the limit, as a live run's does
    a, b = traded(free), traded(limited)
    assert free.status == limited.status == "finished"
    assert any(o.entry.price > 160 for o in a.values()) and a and b                                  # with no limit the walk bought companies above $160 a share ...
    assert all(o.entry.price <= Decimal(160) for o in b.values()) and {k for k, o in a.items() if o.entry.price > 160}.isdisjoint(b)      # ... with the limit it bought none (a neutral company under it may take the place of one over it)
