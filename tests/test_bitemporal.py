from __future__ import annotations

import pandas as pd
import pytest

from rollender_stein.bitemporal import (
    insert_macro_releases,
    latest_release_stream,
    open_db,
)


@pytest.fixture
def con():
    with open_db(":memory:") as c:
        yield c


def _three_monthly_releases() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
            "release_date": pd.to_datetime(["2024-02-02", "2024-03-01", "2024-04-01"]),
            "value": [29.50, 29.65, 29.80],
        }
    )


def test_schema_creates_three_tables(con) -> None:
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchdf()
    assert set(tables["table_name"]) >= {"macro_release", "asset_price", "fx_close"}


def test_insert_then_stream_returns_one_row_per_release(con) -> None:
    rows = _three_monthly_releases()
    n = insert_macro_releases(con, "AHETPI", rows, source="FRED_ALFRED", vintage="2024-04-15")
    assert n == 3

    stream = latest_release_stream(con, "AHETPI")
    assert len(stream) == 3
    assert list(stream["value"]) == [29.50, 29.65, 29.80]
    assert list(stream["release_date"].astype("datetime64[ns]")) == list(
        pd.to_datetime(["2024-02-02", "2024-03-01", "2024-04-01"])
    )


def test_insert_or_replace_overwrites_same_pk(con) -> None:
    """If we re-ingest the same (series_id, reference_date, release_date), the value updates."""
    initial = _three_monthly_releases()
    insert_macro_releases(con, "AHETPI", initial, source="FRED_ALFRED")

    # Pretend a vintage correction: same release+reference, different value
    revised = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2024-01-31"]),
            "release_date": pd.to_datetime(["2024-02-02"]),
            "value": [29.51],
        }
    )
    insert_macro_releases(con, "AHETPI", revised, source="FRED_ALFRED", vintage="re-pull")

    stream = latest_release_stream(con, "AHETPI")
    feb_value = stream.loc[stream["release_date"] == pd.Timestamp("2024-02-02"), "value"].iloc[0]
    assert feb_value == 29.51


def test_same_release_date_two_reference_periods_picks_latest_period(con) -> None:
    """If on Mar 1 we publish Feb-2024 (new headline) AND a Jan-2024 revision, the
    stream should hold the Feb headline (29.65), not the Jan revision (29.55)."""
    rows = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "release_date": pd.to_datetime(["2024-03-01", "2024-03-01"]),
            "value": [29.55, 29.65],  # Jan revision, Feb headline
        }
    )
    insert_macro_releases(con, "AHETPI", rows, source="FRED_ALFRED")

    stream = latest_release_stream(con, "AHETPI")
    assert len(stream) == 1, "two rows on the same release_date must collapse to one"
    assert stream.iloc[0]["value"] == 29.65, "headline (latest reference_date) must win"


def test_insert_rejects_missing_columns(con) -> None:
    bad = pd.DataFrame({"release_date": pd.to_datetime(["2024-02-02"]), "value": [1.0]})
    with pytest.raises(KeyError, match="reference_date"):
        insert_macro_releases(con, "AHETPI", bad, source="FRED_ALFRED")


def test_stream_for_unknown_series_is_empty(con) -> None:
    stream = latest_release_stream(con, "DOES_NOT_EXIST")
    assert len(stream) == 0
