from __future__ import annotations

import pandas as pd
import pytest

from rollender_stein.locf import forward_fill_to_calendar


@pytest.fixture
def macro_release() -> pd.DataFrame:
    """Synthetic monthly series with a publication lag.

    Reference period = month covered. Release date = first business day of the next month.
    Mirrors how AHETPI / M2 / etc. behave: the value describes January but is published
    in early February.
    """
    return pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
            "release_date": pd.to_datetime(["2024-02-02", "2024-03-01", "2024-04-01"]),
            "ahetpi": [29.50, 29.65, 29.80],
        }
    )


def test_value_appears_only_on_or_after_release(macro_release: pd.DataFrame) -> None:
    cal = pd.bdate_range("2024-01-15", "2024-04-05")
    out = forward_fill_to_calendar(macro_release, cal)

    assert pd.isna(out.loc["2024-02-01", "ahetpi"]), "leaked: visible before release"
    assert out.loc["2024-02-02", "ahetpi"] == 29.50
    assert out.loc["2024-02-29", "ahetpi"] == 29.50
    assert out.loc["2024-03-01", "ahetpi"] == 29.65
    assert out.loc["2024-04-01", "ahetpi"] == 29.80


def test_january_rows_have_no_january_reference_value(macro_release: pd.DataFrame) -> None:
    """Anti-look-ahead guard: the January reference period is published in Feb,
    so January calendar rows must NOT see the 29.50 value even though its
    reference_date is 2024-01-31.
    """
    cal = pd.bdate_range("2024-01-15", "2024-04-05")
    out = forward_fill_to_calendar(macro_release, cal)

    jan_slice = out.loc["2024-01-15":"2024-01-31"]
    assert jan_slice["ahetpi"].isna().all(), (
        "look-ahead leak: a release_date >= 2024-02-02 became visible in January"
    )


def test_duplicate_release_dates_rejected() -> None:
    bad = pd.DataFrame(
        {
            "release_date": pd.to_datetime(["2024-02-02", "2024-02-02"]),
            "value": [29.50, 29.55],
        }
    )
    cal = pd.bdate_range("2024-02-01", "2024-02-05")
    with pytest.raises(ValueError, match="duplicate"):
        forward_fill_to_calendar(bad, cal)


def test_missing_release_column_rejected() -> None:
    bad = pd.DataFrame({"some_value": [1.0]})
    cal = pd.bdate_range("2024-02-01", "2024-02-05")
    with pytest.raises(KeyError):
        forward_fill_to_calendar(bad, cal)


def test_release_column_must_be_datetime() -> None:
    bad = pd.DataFrame({"release_date": ["2024-02-02"], "value": [1.0]})
    cal = pd.bdate_range("2024-02-01", "2024-02-05")
    with pytest.raises(TypeError, match="datetime"):
        forward_fill_to_calendar(bad, cal)


def test_explicit_value_cols_subset() -> None:
    macro = pd.DataFrame(
        {
            "release_date": pd.to_datetime(["2024-02-02", "2024-03-01"]),
            "wanted": [1.0, 2.0],
            "ignored": [99.0, 99.0],
        }
    )
    cal = pd.bdate_range("2024-02-01", "2024-03-05")
    out = forward_fill_to_calendar(macro, cal, value_cols=["wanted"])
    assert "wanted" in out.columns
    assert "ignored" not in out.columns
