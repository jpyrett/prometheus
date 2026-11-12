"""The user profile and the view of it the Critic may see (Critic v1 spec section 4.1)."""
import json

import pytest
from pydantic import ValidationError

from prometheus.common.portfolio import load_portfolio
from prometheus.common.user_profile import TOLERANCES, RiskView, UserProfile, load_profile
from prometheus.rag.config import REPO_ROOT

SAMPLE = REPO_ROOT / "configs" / "profiles" / "jacob_pyrett.json"


def profile(**kw) -> UserProfile:
    return UserProfile(**{"user": "T", "risk_tolerance": "moderate", **kw})


def test_the_sample_profile_loads_and_matches_the_sample_portfolio_user():
    p = load_profile(SAMPLE)
    assert (p.risk_tolerance, p.trade_mode, p.age, p.investment_horizon_years) == ("moderate", "ask", None, None)
    assert (p.max_position_pct, p.max_loss_per_trade_pct, p.max_new_exposure_pct, p.min_price) == (0.10, 0.01, 0.30, 5.0)
    assert p.user == load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json").user


def test_the_tolerances_run_from_the_least_risk_to_the_most():
    assert TOLERANCES == ("conservative", "moderate", "aggressive")


@pytest.mark.parametrize("field, bad", [
    ("age", 17), ("age", 121), ("investment_horizon_years", 0), ("auto_min_score", -1), ("auto_min_score", 101), ("max_position_pct", 0), ("max_position_pct", 1.01),
    ("max_loss_per_trade_pct", 0), ("max_new_exposure_pct", 1.5), ("max_order_dollars", 0), ("min_price", -1), ("risk_tolerance", "reckless"), ("trade_mode", "manual"), ("user", "")])
def test_each_bound_is_checked(field, bad):
    with pytest.raises(ValidationError):
        profile(**{field: bad})


@pytest.mark.parametrize("field, ok", [("age", 18), ("age", 120), ("max_position_pct", 1), ("max_loss_per_trade_pct", 1), ("min_price", 0), ("auto_min_score", 0), ("auto_min_score", 100)])
def test_each_bound_is_inclusive_where_the_spec_says_so(field, ok):
    assert getattr(profile(**{field: ok}), field) == ok


def test_excluded_symbols_are_cleaned_and_a_bad_one_is_refused():
    assert profile(excluded_symbols=["$tsla".upper(), "GM", "GM"]).excluded_symbols == ["TSLA", "GM"]
    with pytest.raises(ValidationError):
        profile(excluded_symbols=["../etc"])


def test_excluded_sectors_must_be_sector_etfs():
    assert profile(excluded_sectors=["xle", "XLK"]).excluded_sectors == ["XLE", "XLK"]
    with pytest.raises(ValidationError, match="unknown sector ETF"):
        profile(excluded_sectors=["ZZZ"])


def test_an_unknown_field_is_an_error_not_ignored():
    with pytest.raises(ValidationError):
        profile(risk_toleranse="high")


def test_the_risk_view_has_exactly_the_limits_and_neither_the_name_nor_the_mode():
    view = profile(age=65, investment_horizon_years=2.5, trade_mode="auto", auto_min_score=90, max_order_dollars=5000, excluded_symbols=["GM"], excluded_sectors=["XLE"]).risk_view()
    assert isinstance(view, RiskView)
    assert set(RiskView.model_fields) == {"risk_tolerance", "age", "investment_horizon_years", "max_position_pct", "max_loss_per_trade_pct", "max_new_exposure_pct", "max_order_dollars",
                                          "min_price", "excluded_symbols", "excluded_sectors"}
    dumped = json.dumps(view.model_dump())
    assert "trade_mode" not in dumped and "auto_min_score" not in dumped and '"user"' not in dumped
    assert (view.age, view.investment_horizon_years, view.max_order_dollars, view.excluded_symbols, view.excluded_sectors) == (65, 2.5, 5000, ("GM",), ("XLE",))


def test_the_risk_view_cannot_be_changed():
    with pytest.raises(ValidationError):
        profile().risk_view().max_position_pct = 1.0


def test_a_bad_file_is_reported_with_its_name(tmp_path):
    for name, text, message in [("none.json", None, "does not exist"), ("bad.json", "{", "not valid JSON"), ("wrong.json", '{"user": "T"}', "risk_tolerance")]:
        path = tmp_path / name
        if text is not None:
            path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message) as error:
            load_profile(path)
        assert name in str(error.value)
