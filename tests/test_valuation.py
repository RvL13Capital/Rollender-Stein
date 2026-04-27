from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rollender_stein.calendar import T0_DATE
from rollender_stein.valuation import build_division_array


def _series(values: list[float], dates: list[str], name: str) -> pd.Series:
    return pd.Series(values, index=pd.to_datetime(dates), name=name)


def test_asset_in_x_at_t0_equals_nominal_when_n_x_is_100() -> None:
    """Asset_in_X(T0) = nominal(T0) / N_X(T0) * 100 = nominal(T0) since N_X(T0) = 100.
    The trajectory does NOT enter phase space at [100, 100, 100] — it enters at
    the asset's actual T0-deflated USD value, which preserves cross-asset comparability.
    """
    dates = [str(T0_DATE.date()), "2000-01-04", "2000-01-05"]
    asset = _series([1500.0, 1505.0, 1510.0], dates, "spx")
    n_time = _series([100.0, 100.0, 100.0], dates, "N_Time")
    n_liq = _series([100.0, 100.5, 101.0], dates, "N_Liq")
    n_gold = _series([100.0, 99.5, 99.0], dates, "N_Gold")

    da = build_division_array(asset, n_time=n_time, n_liquidity=n_liq, n_gold=n_gold)

    assert da.asset_in_time.loc[T0_DATE] == pytest.approx(1500.0)
    assert da.asset_in_liquidity.loc[T0_DATE] == pytest.approx(1500.0)
    assert da.asset_in_gold.loc[T0_DATE] == pytest.approx(1500.0)
    # asset_indexed is still informational — anchored at T0
    assert da.asset_indexed.loc[T0_DATE] == pytest.approx(100.0)


def test_division_uses_nominal_directly_not_pre_indexed() -> None:
    """When the asset doubles and N_X stays flat, Asset_in_X doubles in absolute
    USD terms — from $1500 to $3000."""
    dates = [str(T0_DATE.date()), "2010-01-04"]
    asset = _series([1500.0, 3000.0], dates, "spx")
    n_liq = _series([100.0, 100.0], dates, "N_Liq")

    da = build_division_array(asset, n_liquidity=n_liq)
    # nominal $3000 / N_Liq=100 * 100 = $3000 (T0-deflated USD)
    assert da.asset_in_liquidity.iloc[-1] == pytest.approx(3000.0)
    assert da.asset_in_liquidity.iloc[0] == pytest.approx(1500.0)


def test_btc_enters_phase_space_at_real_position_not_synthetic_100() -> None:
    """BTC starting in 2014 must enter phase space at its real T0-deflated price,
    not at [100, 100, 100]. This is the bug-fix the user identified."""
    dates = ["2014-09-17", "2024-12-31"]
    asset = _series([457.33, 93000.0], dates, "btc")
    n_time = _series([200.0, 350.0], dates, "N_Time")  # wages 2x then 3.5x from T0

    da = build_division_array(asset, n_time=n_time)
    # BTC at start: $457.33 / 200 * 100 = $228.665 in T0-wage-deflated USD
    assert da.asset_in_time.iloc[0] == pytest.approx(228.665)
    # BTC at end: $93000 / 350 * 100 = $26571.43
    assert da.asset_in_time.iloc[-1] == pytest.approx(26571.43, rel=1e-3)
    # NOT 100 at start — this is the whole point of the fix
    assert da.asset_in_time.iloc[0] != pytest.approx(100.0)


def test_two_assets_with_same_first_day_enter_at_proportional_positions() -> None:
    """Cross-asset comparability: if asset A is $1000 and asset B is $1 on the
    same date, their phase-space entry positions must differ by 1000x."""
    dates = ["2014-09-17", "2014-09-18"]
    asset_big = _series([1000.0, 1000.0], dates, "big")
    asset_small = _series([1.0, 1.0], dates, "small")
    n_time = _series([200.0, 200.0], dates, "N_Time")

    da_big = build_division_array(asset_big, n_time=n_time)
    da_small = build_division_array(asset_small, n_time=n_time)

    ratio = da_big.asset_in_time.iloc[0] / da_small.asset_in_time.iloc[0]
    assert ratio == pytest.approx(1000.0), (
        f"two assets at $1000 vs $1 must enter phase space at 1000x different "
        f"positions; got ratio {ratio}"
    )


def test_asset_in_x_is_nan_where_numeraire_is_nan() -> None:
    """When N_Gold is NaN (pre-2006 case), asset_in_gold must also be NaN."""
    dates = [str(T0_DATE.date()), "2002-01-02", "2006-01-03"]
    asset = _series([1500.0, 1450.0, 1750.0], dates, "spx")
    n_gold = _series([float("nan"), float("nan"), 100.0], dates, "N_Gold")

    da = build_division_array(asset, n_gold=n_gold)
    assert pd.isna(da.asset_in_gold.loc[T0_DATE])
    assert pd.isna(da.asset_in_gold.loc[pd.Timestamp("2002-01-02")])
    # On 2006-01-03 with N_Gold=100, Asset_in_Gold = $1750 / 100 * 100 = $1750
    assert da.asset_in_gold.loc[pd.Timestamp("2006-01-03")] == pytest.approx(1750.0)


def test_indexed_anchor_uses_ffill_when_t0_missing() -> None:
    """The asset_indexed informational series uses ffill when T0 is a holiday."""
    dates = ["1999-12-31", "2000-01-04"]
    asset = _series([1450.0, 1500.0], dates, "spx")
    n_time = _series([100.0, 100.5], [str(T0_DATE.date()), "2000-01-04"], "N_Time")

    da = build_division_array(asset, n_time=n_time)
    # asset_indexed anchored at 1999-12-31 (1450). At T0 the indexed value
    # forward-fills from 1999-12-31 → 1450/1450*100 = 100.
    assert da.asset_indexed.loc[T0_DATE] == pytest.approx(100.0)
    # asset_in_time uses nominal directly: nominal at T0 (ffilled from 1999-12-31)
    # = $1450; N_Time(T0) = 100; Asset_in_Time(T0) = 1450 / 100 * 100 = 1450.
    assert da.asset_in_time.loc[T0_DATE] == pytest.approx(1450.0)


def test_indexed_anchor_falls_back_to_first_valid_when_asset_starts_after_t0() -> None:
    """For assets that don't exist at T0, asset_indexed falls back to first-valid."""
    dates = ["2014-09-17", "2014-09-18", "2024-12-31"]
    asset = _series([457.33, 424.44, 93000.0], dates, "btc")
    n_time = _series([200.0, 200.5, 350.0], dates, "N_Time")

    da = build_division_array(asset, n_time=n_time)
    # asset_indexed at first valid = 100 (informational only)
    assert da.asset_indexed.loc[pd.Timestamp("2014-09-17")] == pytest.approx(100.0)


def test_anchor_raises_on_entirely_empty_asset() -> None:
    dates = ["2014-01-01", "2014-01-02"]
    asset = pd.Series([float("nan"), float("nan")], index=pd.to_datetime(dates))
    n_time = _series([100.0, 100.0], dates, "N_Time")
    with pytest.raises(RuntimeError, match="entirely empty"):
        build_division_array(asset, n_time=n_time)


def test_nominal_kept_in_to_frame() -> None:
    dates = [str(T0_DATE.date()), "2010-01-04"]
    asset = _series([1500.0, 3000.0], dates, "spx")
    n_liq = _series([100.0, 150.0], dates, "N_Liq")

    df = build_division_array(asset, n_liquidity=n_liq).to_frame()
    assert "nominal_usd" in df.columns
    assert df.loc[pd.Timestamp("2010-01-04"), "nominal_usd"] == 3000.0
    assert "asset_in_liquidity" in df.columns
    assert "asset_in_gold" not in df.columns


# ----- Audit patch 04: defensive guards ---------------------------------------


def test_no_warning_when_all_numeraires_anchored_at_100_at_t0() -> None:
    """The post-patch-06 healthy case: every numéraire = 100 exact at T0,
    no axis is in a different gauge, no warning fires."""
    import warnings as _w

    dates = [str(T0_DATE.date()), "2010-01-04"]
    asset = _series([1500.0, 3000.0], dates, "spx")
    n_time = _series([100.0, 130.0], dates, "N_Time")
    n_liq = _series([100.0, 150.0], dates, "N_Liq")
    n_e = _series([100.0, 200.0], dates, "N_Energy")

    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning is now an error
        build_division_array(asset, n_time=n_time, n_liquidity=n_liq, n_energy=n_e)


def test_warning_fires_when_numeraire_is_nan_at_t0() -> None:
    """Patch-06 known case: N_Gold is NaN at T0 (8-month gap). A
    RuntimeWarning surfaces this honestly."""
    dates = [str(T0_DATE.date()), "2006-01-03", "2010-01-04"]
    asset = _series([1500.0, 1700.0, 3000.0], dates, "spx")
    n_gold = _series([float("nan"), 100.0, 200.0], dates, "n_gold")

    with pytest.warns(RuntimeWarning, match=r"NaN at T0"):
        build_division_array(asset, n_gold=n_gold)


def test_warning_fires_when_numeraire_not_100_at_t0() -> None:
    """If a numéraire is not exactly 100 at T0 (anchor bug), warn loudly."""
    dates = [str(T0_DATE.date()), "2010-01-04"]
    asset = _series([1500.0, 3000.0], dates, "spx")
    n_liq = _series([99.5, 150.0], dates, "n_liquidity")  # 99.5 ≠ 100 at T0

    with pytest.warns(RuntimeWarning, match=r"99\.5"):
        build_division_array(asset, n_liquidity=n_liq)


def test_warning_fires_when_numeraire_has_no_t0_observation() -> None:
    """If T0 isn't even in the numéraire's index, warn — the axis has
    no defined gauge at the reference date."""
    # numéraire dates do NOT include T0_DATE
    dates = ["2010-01-04", "2020-01-06"]
    asset = _series([1500.0, 3000.0], dates, "spx")
    n_time = _series([100.0, 200.0], dates, "n_time")  # no T0 row!

    with pytest.warns(RuntimeWarning, match=r"no observation at T0"):
        build_division_array(asset, n_time=n_time)


def test_t0_invariant_tol_silences_warning_when_loose() -> None:
    """A loose tolerance (e.g. 1.0) silences warnings for slightly-off
    numéraires — useful when a caller knowingly accepts small anchor drift."""
    import warnings as _w

    dates = [str(T0_DATE.date()), "2010-01-04"]
    asset = _series([1500.0, 3000.0], dates, "spx")
    n_liq = _series([99.5, 150.0], dates, "n_liquidity")  # 0.5 off

    # With tight tolerance: warns
    with pytest.warns(RuntimeWarning, match=r"99\.5"):
        build_division_array(asset, n_liquidity=n_liq, t0_invariant_tol=1e-6)

    # With loose tolerance (1.0 absolute): silent
    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning → error
        build_division_array(asset, n_liquidity=n_liq, t0_invariant_tol=1.0)


def test_t0_invariant_tol_inf_disables_warning() -> None:
    """``t0_invariant_tol=float('inf')`` disables the value-deviation warning
    entirely (the NaN-at-T0 and missing-T0 branches still fire — those
    aren't gated by tol)."""
    import warnings as _w

    dates = [str(T0_DATE.date()), "2010-01-04"]
    asset = _series([1500.0, 3000.0], dates, "spx")
    n_liq = _series([50.0, 150.0], dates, "n_liquidity")  # very far from 100

    with _w.catch_warnings():
        _w.simplefilter("error")
        build_division_array(asset, n_liquidity=n_liq, t0_invariant_tol=float("inf"))


def test_zero_in_numeraire_produces_nan_not_inf() -> None:
    """Patch 04 division-by-zero guard: a zero in the denominator must
    propagate as NaN, not silent ±inf. This cannot happen on real data
    but is defensive against malformed numéraires."""
    import warnings as _w

    dates = [str(T0_DATE.date()), "2010-01-04", "2020-01-06"]
    asset = _series([1500.0, 3000.0, 4500.0], dates, "spx")
    # N_Time legitimately starts at 100, then drops to 0 — synthetic.
    n_time = _series([100.0, 0.0, 200.0], dates, "n_time")

    with _w.catch_warnings():
        _w.simplefilter("ignore")  # we don't care about other warnings here
        da = build_division_array(asset, n_time=n_time)
    # At the zero-denominator date, ratio must be NaN, not inf.
    assert pd.isna(da.asset_in_time.loc[pd.Timestamp("2010-01-04")])
    # Surrounding dates compute normally.
    assert pd.notna(da.asset_in_time.loc[T0_DATE])


# ----- Volume / dollar-turnover passthrough -----------------------------------


def test_volume_and_turnover_appear_in_to_frame_when_provided() -> None:
    dates = [str(T0_DATE.date()), "2010-01-04", "2020-01-06"]
    asset = _series([1500.0, 3000.0, 4500.0], dates, "spx")
    n_liq = _series([100.0, 150.0, 200.0], dates, "N_Liq")
    volume = _series([1e6, 2e6, 3e6], dates, "volume")
    turnover = volume * asset

    da = build_division_array(
        asset, n_liquidity=n_liq, volume=volume, dollar_turnover=turnover
    )
    df = da.to_frame()
    assert "volume" in df.columns
    assert "dollar_turnover" in df.columns
    assert df["volume"].iloc[-1] == pytest.approx(3e6)
    assert df["dollar_turnover"].iloc[-1] == pytest.approx(4500.0 * 3e6)


def test_volume_absent_by_default_for_backwards_compat() -> None:
    """Existing callers passing only numéraires get the original schema."""
    dates = [str(T0_DATE.date()), "2010-01-04"]
    asset = _series([1500.0, 3000.0], dates, "spx")
    n_liq = _series([100.0, 150.0], dates, "N_Liq")

    da = build_division_array(asset, n_liquidity=n_liq)
    assert da.volume is None
    assert da.dollar_turnover is None
    df = da.to_frame()
    assert "volume" not in df.columns
    assert "dollar_turnover" not in df.columns


def test_volume_aligned_to_numeraire_calendar() -> None:
    """Volume/turnover get reindexed to the same base calendar as the
    deflated axes — so dashboard slicing uses one consistent index."""
    # Numéraire spans more dates than the volume series.
    n_dates = pd.bdate_range(T0_DATE, periods=10)
    asset = pd.Series(np.linspace(1500, 1600, 10), index=n_dates, name="spx")
    n_liq = pd.Series(np.linspace(100, 110, 10), index=n_dates, name="N_Liq")
    # Volume only on the last 5 days.
    volume = pd.Series([1e6, 2e6, 3e6, 4e6, 5e6], index=n_dates[-5:], name="volume")

    da = build_division_array(asset, n_liquidity=n_liq, volume=volume)
    assert da.volume is not None
    # Same length as the numéraire calendar
    assert len(da.volume) == 10
    # Early days are NaN (no volume); late days have values
    assert da.volume.iloc[:5].isna().all()
    assert da.volume.iloc[-1] == pytest.approx(5e6)
