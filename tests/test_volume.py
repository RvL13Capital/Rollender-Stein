"""Tests for rollender_stein.volume — turnover, rolling z-score, coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rollender_stein.volume import (
    compute_dollar_turnover,
    coverage_ratio,
    rolling_volume_zscore,
)


def _bd(n: int, start: str = "2000-01-03") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


# ---------------------------------------------------------------------------
# compute_dollar_turnover
# ---------------------------------------------------------------------------


def test_dollar_turnover_elementwise_multiplication() -> None:
    idx = _bd(5)
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx)
    volume = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0], index=idx)
    out = compute_dollar_turnover(close, volume)
    assert list(out.values) == [1000.0, 2200.0, 3600.0, 5200.0, 7000.0]
    assert out.name == "dollar_turnover"


def test_dollar_turnover_inner_joins_misaligned_indexes() -> None:
    close = pd.Series([10.0, 11.0, 12.0], index=_bd(3, "2000-01-03"))
    volume = pd.Series([100.0, 200.0, 300.0], index=_bd(3, "2000-01-04"))
    out = compute_dollar_turnover(close, volume)
    # Two overlapping dates: 2000-01-04 and 2000-01-05
    assert len(out) == 2
    assert out.iloc[0] == pytest.approx(11.0 * 100.0)
    assert out.iloc[1] == pytest.approx(12.0 * 200.0)


def test_dollar_turnover_propagates_nan() -> None:
    idx = _bd(3)
    close = pd.Series([10.0, np.nan, 12.0], index=idx)
    volume = pd.Series([100.0, 200.0, np.nan], index=idx)
    out = compute_dollar_turnover(close, volume)
    assert out.iloc[0] == pytest.approx(1000.0)
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])


# ---------------------------------------------------------------------------
# rolling_volume_zscore
# ---------------------------------------------------------------------------


def test_zscore_warmup_below_min_periods_is_nan() -> None:
    idx = _bd(20)
    turnover = pd.Series(np.linspace(1e6, 2e6, 20), index=idx)
    z = rolling_volume_zscore(turnover, window=10, min_periods=5)
    # First 4 rows lack the min_periods=5; should be NaN
    assert z.iloc[:4].isna().all()
    # Row 4 (5th obs) is the first that has min_periods reached
    assert pd.notna(z.iloc[4])


def test_zscore_handles_negative_turnover_silently() -> None:
    """WTI Crude Futures (CL=F) closed at -$37.63 on 2020-04-20. With raw
    close * volume that produces negative turnover; ``np.log(negative)``
    emits a RuntimeWarning. The ``where(turnover > 0)`` filter masks both
    zero AND negative turnover in one vectorised pass, so neither warning
    nor inf reaches downstream consumers."""
    import warnings as _w

    idx = _bd(50)
    raw = np.full(50, 1e6, dtype="float64")
    raw[10] = -5e5  # WTI-style negative turnover day
    turnover = pd.Series(raw, index=idx)

    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning becomes a hard fail
        z = rolling_volume_zscore(turnover, window=20, min_periods=5)

    # Negative day itself is masked to NaN (log of masked NaN propagates).
    assert pd.isna(z.iloc[10])
    # No infinity anywhere.
    assert not np.isinf(z).any()


def test_zscore_log_zero_does_not_poison_window() -> None:
    """A single zero-volume day must not zero-out z-scores for the next
    `window` rows. Zeros are masked to NaN before log; rolling skips NaN."""
    idx = _bd(50)
    # Constant turnover with one bad day at index 10 (zero) and at index 25.
    raw = np.full(50, 1e6, dtype="float64")
    raw[10] = 0.0
    raw[25] = 0.0
    turnover = pd.Series(raw, index=idx)
    z = rolling_volume_zscore(turnover, window=20, min_periods=5)
    # Bad days themselves should be NaN (log(NaN) propagates).
    assert pd.isna(z.iloc[10])
    assert pd.isna(z.iloc[25])
    # Days adjacent to bad days, with otherwise constant turnover, should
    # produce z = 0/NaN-or-zero — NOT -inf or huge negative numbers.
    # (Constant log(turnover) → std = 0 → z = NaN per our std-mask.)
    surrounding = z.iloc[15:20]
    assert not (surrounding < -100).any()
    assert not np.isinf(surrounding.dropna()).any()


def test_zscore_constant_turnover_yields_nan_not_inf() -> None:
    """When rolling std is 0 (constant log-turnover within window), z must be
    NaN (per our std > 0 mask), not 0/0 = NaN-with-RuntimeWarning, and never inf."""
    idx = _bd(30)
    turnover = pd.Series(np.full(30, 5e6), index=idx)
    z = rolling_volume_zscore(turnover, window=10, min_periods=5)
    # All NaN once min_periods is met (std == 0 case)
    valid = z.iloc[5:]
    assert valid.isna().all()
    assert not np.isinf(z).any()


def test_zscore_no_lookahead() -> None:
    """Forensic check (Phase-7 spirit): z-score at row t must be computed
    from data only up to and including t. Truncating the input at t and
    rerunning must produce the same z-score at t."""
    idx = _bd(400)
    # Some structured trend so the z-score is nontrivial
    rng = np.random.default_rng(42)
    raw = np.exp(rng.normal(15, 0.5, 400))
    turnover = pd.Series(raw, index=idx)

    full_z = rolling_volume_zscore(turnover, window=63, min_periods=21)
    # Truncate at t=300 and recompute; row 300 must match exactly.
    truncated = turnover.iloc[: 300 + 1]
    truncated_z = rolling_volume_zscore(truncated, window=63, min_periods=21)
    full_at_300 = full_z.iloc[300]
    trunc_at_300 = truncated_z.iloc[300]
    assert full_at_300 == pytest.approx(trunc_at_300, abs=1e-12), (
        f"z[300] differs: full={full_at_300} truncated={trunc_at_300}"
    )


def test_zscore_rejects_invalid_windows() -> None:
    s = pd.Series([1.0, 2.0, 3.0], index=_bd(3))
    with pytest.raises(ValueError, match="window must be >= 2"):
        rolling_volume_zscore(s, window=1)
    with pytest.raises(ValueError, match="min_periods"):
        rolling_volume_zscore(s, window=10, min_periods=11)
    with pytest.raises(ValueError, match="min_periods"):
        rolling_volume_zscore(s, window=10, min_periods=1)


def test_zscore_post_warmup_centered_around_zero() -> None:
    """For stationary log-turnover, the rolling z-score should hover around 0
    (mean) with finite spread."""
    rng = np.random.default_rng(0)
    n = 1000
    idx = _bd(n)
    # Stationary log-normal: log-turnover ~ N(15, 0.3)
    log_t = rng.normal(15, 0.3, n)
    turnover = pd.Series(np.exp(log_t), index=idx)
    z = rolling_volume_zscore(turnover, window=252, min_periods=63)
    # After warmup, z should be roughly centered with std ≈ 1
    z_post_warmup = z.iloc[252:].dropna()
    assert abs(z_post_warmup.mean()) < 0.3
    assert 0.7 < z_post_warmup.std() < 1.3


# ---------------------------------------------------------------------------
# coverage_ratio
# ---------------------------------------------------------------------------


def test_coverage_ratio_empty() -> None:
    assert coverage_ratio(pd.Series([], dtype="float64")) == 0.0


def test_coverage_ratio_all_present() -> None:
    assert coverage_ratio(pd.Series([1.0, 2.0, 3.0])) == 1.0


def test_coverage_ratio_all_nan() -> None:
    assert coverage_ratio(pd.Series([np.nan, np.nan, np.nan])) == 0.0


def test_coverage_ratio_mixed() -> None:
    s = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
    assert coverage_ratio(s) == pytest.approx(0.6)
