from __future__ import annotations

import pandas as pd
import pytest

from rollender_stein.bitemporal import insert_macro_releases, open_db
from rollender_stein.calendar import T0_DATE
from rollender_stein.numeraires.time import SERIES_ID, SOURCE, build_n_time


@pytest.fixture
def con():
    with open_db(":memory:") as c:
        yield c


def _seed_pre_and_post_t0_ahetpi(con) -> None:
    """Insert two AHETPI releases: one before T0 (so T0 has a value) and one after."""
    rows = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["1999-11-30", "1999-12-31"]),
            "release_date": pd.to_datetime(["1999-12-06", "2000-01-07"]),
            "value": [13.45, 13.50],
        }
    )
    insert_macro_releases(con, SERIES_ID, rows, source=SOURCE)


def test_n_time_is_exactly_100_at_t0(con) -> None:
    _seed_pre_and_post_t0_ahetpi(con)
    n_time = build_n_time(con, end=pd.Timestamp("2000-02-01"))

    assert n_time.loc[T0_DATE] == pytest.approx(100.0)
    assert n_time.name == "N_Time"


def test_n_time_jumps_on_release_day(con) -> None:
    """When the next AHETPI release lands, N_Time jumps to the new ratio."""
    _seed_pre_and_post_t0_ahetpi(con)
    n_time = build_n_time(con, end=pd.Timestamp("2000-02-01"))

    # 2000-01-07 is a Friday — first NYSE day with the new release.
    expected_jan07 = (13.50 / 13.45) * 100.0
    assert n_time.loc[pd.Timestamp("2000-01-07")] == pytest.approx(expected_jan07)
    # Day before (2000-01-06): still on the old release.
    assert n_time.loc[pd.Timestamp("2000-01-06")] == pytest.approx(100.0)


def test_build_n_time_raises_when_no_data(con) -> None:
    with pytest.raises(RuntimeError, match="no rows in macro_release"):
        build_n_time(con)


def test_build_n_time_raises_when_t0_unanchored(con) -> None:
    """If the first release is AFTER T0, T0 has NaN and we cannot index."""
    rows = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2000-12-31"]),
            "release_date": pd.to_datetime(["2001-01-05"]),
            "value": [14.00],
        }
    )
    insert_macro_releases(con, SERIES_ID, rows, source=SOURCE)
    with pytest.raises(RuntimeError, match="cannot index"):
        build_n_time(con, end=pd.Timestamp("2001-02-01"))
