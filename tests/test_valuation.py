from __future__ import annotations

import pandas as pd
import pytest

from rollender_stein.calendar import T0_DATE
from rollender_stein.valuation import build_division_array


def _series(values: list[float], dates: list[str], name: str) -> pd.Series:
    return pd.Series(values, index=pd.to_datetime(dates), name=name)


def test_asset_in_time_at_t0_is_100() -> None:
    """At T0 with all inputs anchored at 100, every Asset_in_X must equal 100."""
    dates = [str(T0_DATE.date()), "2000-01-04", "2000-01-05"]
    asset = _series([1500.0, 1505.0, 1510.0], dates, "spx")
    n_time = _series([100.0, 100.0, 100.0], dates, "N_Time")
    n_liq = _series([100.0, 100.5, 101.0], dates, "N_Liq")
    n_gold = _series([100.0, 99.5, 99.0], dates, "N_Gold")

    da = build_division_array(asset, n_time=n_time, n_liquidity=n_liq, n_gold=n_gold)

    assert da.asset_indexed.loc[T0_DATE] == pytest.approx(100.0)
    assert da.asset_in_time.loc[T0_DATE] == pytest.approx(100.0)
    assert da.asset_in_liquidity.loc[T0_DATE] == pytest.approx(100.0)
    assert da.asset_in_gold.loc[T0_DATE] == pytest.approx(100.0)


def test_division_uses_t0_anchor() -> None:
    """If asset doubles while N_X stays flat, asset_in_X = 200."""
    dates = [str(T0_DATE.date()), "2010-01-04"]
    asset = _series([1500.0, 3000.0], dates, "spx")
    n_liq = _series([100.0, 100.0], dates, "N_Liq")

    da = build_division_array(asset, n_liquidity=n_liq)
    assert da.asset_in_liquidity.iloc[-1] == pytest.approx(200.0)


def test_asset_in_x_is_nan_where_numeraire_is_nan() -> None:
    """When N_Gold is NaN (pre-2006 case), asset_in_gold must also be NaN."""
    dates = [str(T0_DATE.date()), "2002-01-02", "2006-01-03"]
    asset = _series([1500.0, 1450.0, 1750.0], dates, "spx")
    n_gold = _series([float("nan"), float("nan"), 100.0], dates, "N_Gold")

    da = build_division_array(asset, n_gold=n_gold)
    assert pd.isna(da.asset_in_gold.loc[T0_DATE])
    assert pd.isna(da.asset_in_gold.loc[pd.Timestamp("2002-01-02")])
    assert da.asset_in_gold.loc[pd.Timestamp("2006-01-03")] == pytest.approx(
        (1750.0 / 1500.0) * 100.0
    )


def test_anchor_uses_ffill_when_t0_missing() -> None:
    """If asset has no value on T0 (e.g., NYSE was closed), use the latest prior value."""
    dates = ["1999-12-31", "2000-01-04"]
    asset = _series([1450.0, 1500.0], dates, "spx")
    n_time = _series([100.0, 100.5], [str(T0_DATE.date()), "2000-01-04"], "N_Time")

    da = build_division_array(asset, n_time=n_time)
    # Anchor pulled from 1999-12-31 (1450.0). Indexed asset on T0 = 1450/1450*100 = 100
    assert da.asset_indexed.loc[T0_DATE] == pytest.approx(100.0)


def test_nominal_kept_in_to_frame() -> None:
    dates = [str(T0_DATE.date()), "2010-01-04"]
    asset = _series([1500.0, 3000.0], dates, "spx")
    n_liq = _series([100.0, 150.0], dates, "N_Liq")

    df = build_division_array(asset, n_liquidity=n_liq).to_frame()
    assert "nominal_usd" in df.columns
    assert df.loc[pd.Timestamp("2010-01-04"), "nominal_usd"] == 3000.0
    assert "asset_in_liquidity" in df.columns
    assert "asset_in_gold" not in df.columns
