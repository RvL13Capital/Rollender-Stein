from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rollender_stein.bitemporal import insert_macro_releases, open_db
from rollender_stein.calendar import T0_DATE
from rollender_stein.numeraires.liquidity import (
    ALL_SERIES,
    build_n_liq,
    extend_levels_with_growth,
)


@pytest.fixture
def con():
    with open_db(":memory:") as c:
        yield c


# ----- extend_levels_with_growth ---------------------------------------------


def test_extend_compounds_after_last_level() -> None:
    levels = pd.Series(
        [100.0, 102.0],
        index=pd.to_datetime(["2023-10-01", "2023-11-01"]),
        name="lvl",
    )
    growth_pct = pd.Series(
        [0.5, 1.0, 0.25],  # %, monthly
        index=pd.to_datetime(["2023-12-01", "2024-01-01", "2024-02-01"]),
    )
    out = extend_levels_with_growth(levels, growth_pct)
    # historical preserved
    assert out.loc["2023-10-01"] == 100.0
    assert out.loc["2023-11-01"] == 102.0
    # extension compounds: 102 * 1.005 * 1.01 * 1.0025
    assert out.loc["2023-12-01"] == pytest.approx(102.0 * 1.005)
    assert out.loc["2024-01-01"] == pytest.approx(102.0 * 1.005 * 1.01)
    assert out.loc["2024-02-01"] == pytest.approx(102.0 * 1.005 * 1.01 * 1.0025)


def test_extend_ignores_growth_at_or_before_last_level() -> None:
    """Growth rates dated within the level series' coverage are authoritative-skipped."""
    levels = pd.Series(
        [100.0, 102.0],
        index=pd.to_datetime(["2023-10-01", "2023-11-01"]),
    )
    growth_pct = pd.Series(
        [9.99, 9.99, 1.0],  # rates at/before 2023-11 must be ignored
        index=pd.to_datetime(["2023-10-01", "2023-11-01", "2023-12-01"]),
    )
    out = extend_levels_with_growth(levels, growth_pct)
    assert out.loc["2023-11-01"] == 102.0
    assert out.loc["2023-12-01"] == pytest.approx(102.0 * 1.01)


def test_extend_with_no_forward_growth_returns_levels() -> None:
    levels = pd.Series([100.0], index=pd.to_datetime(["2024-01-01"]))
    growth_pct = pd.Series([], index=pd.to_datetime([]), dtype=float)
    out = extend_levels_with_growth(levels, growth_pct)
    pd.testing.assert_series_equal(out, levels)


def test_extend_raises_on_empty_levels() -> None:
    with pytest.raises(ValueError, match="empty"):
        extend_levels_with_growth(pd.Series([], dtype=float), pd.Series([], dtype=float))


# ----- build_n_liq end-to-end with synthetic data ----------------------------


def _seed_minimal_panel(con, monthly_dates: list[str], wm_dates: list[str]) -> None:
    """Insert just enough rows in each series so the LOCF pipeline has values at T0."""
    # Seed dates anchored before T0=2000-01-03 so every LOCF query has a value at T0.
    monthly_idx = pd.to_datetime(monthly_dates)
    weekly_idx = pd.to_datetime(wm_dates)

    insert_macro_releases(
        con,
        ALL_SERIES["US_M2"],
        pd.DataFrame(
            {
                "reference_date": weekly_idx,
                "release_date": weekly_idx + pd.Timedelta(days=7),
                "value": [4500.0 + i * 5 for i in range(len(weekly_idx))],
            }
        ),
        source="FRED_ALFRED",
    )

    insert_macro_releases(
        con,
        ALL_SERIES["EZ_M3_LEVEL"],
        pd.DataFrame(
            {
                "reference_date": monthly_idx,
                "release_date": monthly_idx + pd.Timedelta(days=28),
                "value": [4_500_000_000_000.0 + i * 1e10 for i in range(len(monthly_idx))],
            }
        ),
        source="FRED",
    )
    insert_macro_releases(
        con,
        ALL_SERIES["EZ_M3_GROWTH"],
        pd.DataFrame({"reference_date": monthly_idx, "release_date": monthly_idx, "value": [0.3] * len(monthly_idx)}),
        source="FRED",
    )
    insert_macro_releases(
        con,
        ALL_SERIES["JP_M3_LEVEL"],
        pd.DataFrame(
            {
                "reference_date": monthly_idx,
                "release_date": monthly_idx + pd.Timedelta(days=28),
                "value": [600_000_000_000_000.0 + i * 1e12 for i in range(len(monthly_idx))],
            }
        ),
        source="FRED",
    )
    insert_macro_releases(
        con,
        ALL_SERIES["JP_M3_GROWTH"],
        pd.DataFrame({"reference_date": monthly_idx, "release_date": monthly_idx, "value": [0.1] * len(monthly_idx)}),
        source="FRED",
    )

    fx_dates = pd.bdate_range(monthly_idx[0], "2000-01-15")
    insert_macro_releases(
        con,
        ALL_SERIES["EURUSD"],
        pd.DataFrame(
            {"reference_date": fx_dates, "release_date": fx_dates, "value": [1.05] * len(fx_dates)}
        ),
        source="FRED",
    )
    insert_macro_releases(
        con,
        ALL_SERIES["USDJPY"],
        pd.DataFrame(
            {"reference_date": fx_dates, "release_date": fx_dates, "value": [105.0] * len(fx_dates)}
        ),
        source="FRED",
    )


def test_n_liq_is_exactly_100_at_t0(con) -> None:
    _seed_minimal_panel(
        con,
        monthly_dates=["1999-09-01", "1999-10-01", "1999-11-01", "1999-12-01"],
        wm_dates=["1999-11-01", "1999-11-08", "1999-11-15", "1999-11-22"],
    )
    n_liq = build_n_liq(con, end=pd.Timestamp("2000-01-15"))
    assert n_liq.loc[T0_DATE] == pytest.approx(100.0)
    assert n_liq.name == "N_Liq"


def test_n_liq_raises_when_t0_unanchored(con) -> None:
    """If a non-FX input has no release by T0, ocean at T0 is NaN and we can't index."""
    # FX must have values at T0 to isolate the M-aggregate path.
    fx_dates = pd.bdate_range("1999-06-01", "2000-02-01")
    insert_macro_releases(
        con,
        ALL_SERIES["EURUSD"],
        pd.DataFrame(
            {"reference_date": fx_dates, "release_date": fx_dates, "value": [1.05] * len(fx_dates)}
        ),
        source="FRED",
    )
    insert_macro_releases(
        con,
        ALL_SERIES["USDJPY"],
        pd.DataFrame(
            {"reference_date": fx_dates, "release_date": fx_dates, "value": [105.0] * len(fx_dates)}
        ),
        source="FRED",
    )
    # US M2 / M3 / growth seeded with releases only AFTER T0 → NaN at T0.
    post_t0 = pd.to_datetime(["2000-02-01"])
    for sid_key in ("US_M2", "EZ_M3_LEVEL", "EZ_M3_GROWTH", "JP_M3_LEVEL", "JP_M3_GROWTH"):
        insert_macro_releases(
            con,
            ALL_SERIES[sid_key],
            pd.DataFrame(
                {"reference_date": post_t0, "release_date": post_t0, "value": [1.0]}
            ),
            source="FRED",
        )
    with pytest.raises(RuntimeError, match="cannot index"):
        build_n_liq(con, end=pd.Timestamp("2000-03-01"))
