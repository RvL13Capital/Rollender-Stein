"""Daily traded-volume metrics — turnover, z-score, coverage.

Volume is the only AVE input that is *not* a numéraire and *not* deflated. It
serves as a conviction signal in the Phase-Space dashboard: high-volume points
mark where the market actually transacted at a given purchasing-power
coordinate; low-volume points mark drift or illiquidity.

Forensic notes:

1. **Use raw close, never ``adj_close``.** yfinance retroactively split-adjusts
   both ``Close`` (divided by split factor) and ``Volume`` (multiplied by split
   factor), so ``raw_close * volume`` cancels splits and yields the true
   historical USD turnover for the day. ``adj_close * volume`` would distort
   by the accumulated dividend yield (adj_close has dividend reinvestment
   baked in; volume does not).

2. **Mask zeros before ``log``.** Trading halts and bad-data days produce
   ``volume == 0`` (and therefore ``turnover == 0``). ``np.log(0) = -inf``
   would poison the rolling mean / std for the entire window length
   (`window=252` trading days ≈ one year of unusable z-scores). We replace
   zeros with NaN before the log transform; pandas' ``rolling()`` skips NaN
   inside the window.

3. **Rolling, not full-sample, z-score.** Over 25 years dollar-turnover grew
   by orders of magnitude (more capital, more retail, lower fees). A
   full-sample z-score would always rank 2024 highest and 2000 lowest —
   uninformative. A 252-day rolling z-score detects volume *regime shifts
   relative to the recent past*, which is the actually interesting signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW: int = 252  # trading days ≈ 1 year
# Emit z-scores after a quarter rather than waiting a full year — keeps brand-new
# tickers (yfinance recent IPOs) renderable instead of all-floor for 252 days.
DEFAULT_MIN_PERIODS: int = 63


def compute_dollar_turnover(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Return per-day USD turnover ≈ ``close * volume``.

    Inputs must be the **raw** (unadjusted) close and the yfinance volume
    column — see module docstring for the split-cancellation argument.

    The two series are inner-joined on their indexes, so days present in
    only one are dropped. Element-wise multiplication propagates NaN.
    """
    aligned_close, aligned_vol = close.align(volume, join="inner")
    out = aligned_close.astype("float64") * aligned_vol.astype("float64")
    return out.rename("dollar_turnover")


def rolling_volume_zscore(
    turnover: pd.Series,
    *,
    window: int = DEFAULT_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> pd.Series:
    """Rolling z-score of ``log(turnover)`` over a trailing ``window``.

    Zeros are masked to NaN before ``np.log`` so a single trading-halt day
    cannot poison the next ``window`` observations with ``-inf``.

    The window is **strictly trailing** (no centering, no future observations
    used) — at row ``t`` only data from ``[t-window+1, t]`` enters the
    statistic. This preserves the AVE's no-look-ahead invariant.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if min_periods < 2 or min_periods > window:
        raise ValueError(
            f"min_periods must be in [2, window={window}], got {min_periods}"
        )

    safe = turnover.replace(0, np.nan)
    log_t = np.log(safe)
    rolling = log_t.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std()
    # Where std == 0 (pathologically constant turnover within the window),
    # replace with NaN so we get NaN propagation instead of 0/0 = NaN+warning.
    std = std.where(std > 0)
    z: pd.Series = (log_t - mean) / std
    return z.rename("vol_z_score")


def coverage_ratio(s: pd.Series) -> float:
    """Fraction of non-NaN observations in ``s`` — 0.0 for empty input."""
    n = len(s)
    if n == 0:
        return 0.0
    return float(s.notna().sum()) / float(n)
