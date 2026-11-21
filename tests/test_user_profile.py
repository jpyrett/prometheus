"""The user profile and the view of it the Critic may see (Critic v1 spec section 4.1)."""
import json

import pytest
from pydantic import ValidationError

from prometheus.common.user_profile import TOLERANCES, UserProfile, load_profile


def profile(**kw) -> UserProfile:
    return UserProfile(**{"user": "T", "risk_tolerance": "moderate", **kw})


def test_the_tolerances_run_from_the_least_risk_to_the_most():
    assert TOLERANCES == ("conservative", "moderate", "aggressive")


@pytest.mark.parametrize("field, bad", [
    ("age", 17), ("age", 121), ("investment_horizon_years", 0), ("min_price", -1), ("risk_tolerance", "reckless"), ("trade_mode", "manual"), ("user", "")])
def test_each_bound_is_checked(field, bad):
    with pytest.raises(ValidationError):
        profile(**{field: bad})


@pytest.mark.parametrize("field, ok", [("age", 18), ("age", 120), ("min_price", 0)])
def test_each_bound_is_inclusive_where_the_spec_says_so(field, ok):
    assert getattr(profile(**{field: ok}), field) == ok


def test_excluded_sectors_must_be_sector_etfs():
    assert profile(excluded_sectors=["xle", "XLK"]).excluded_sectors == ["XLE", "XLK"]
    with pytest.raises(ValidationError, match="unknown sector ETF"):
        profile(excluded_sectors=["ZZZ"])


def test_an_unknown_field_is_an_error_not_ignored():
    with pytest.raises(ValidationError):
        profile(risk_toleranse="high")
    with pytest.raises(ValidationError):
        profile(auto_min_score=80)      # entry 104: the automatic-trading mark is gone from the profile, as the size limits went in entry 91


def test_a_bad_file_is_reported_with_its_name(tmp_path):
    for name, text, message in [("none.json", None, "does not exist"), ("bad.json", "{", "not valid JSON"), ("wrong.json", '{"user": "T"}', "risk_tolerance")]:
        path = tmp_path / name
        if text is not None:
            path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message) as error:
            load_profile(path)
        assert name in str(error.value)


@pytest.mark.parametrize("field", ["max_position_pct", "max_loss_per_trade_pct", "max_new_exposure_pct", "max_order_dollars"])
def test_the_profile_has_no_size_limit_and_one_in_a_file_is_refused_not_ignored(field):
    """Design choices entry 91 (Jacob, 2026-11-15: "I do not want any of these limits"): the four size limits are not part of the profile. A profile file that still names one is an error, so no old file silently keeps a limit that nothing applies."""
    assert field not in UserProfile.model_fields and not hasattr(profile(), "sizing_limits")
    with pytest.raises(ValidationError, match=field):
        profile(**{field: 0.1})
