"""Statistical pattern extraction on the AVE outputs.

These are DESCRIPTIVE diagnostics — z-scores, correlations, residual statistics —
NOT predictive signals. Mean-reversion of an asset's gold-multiplier z-score is
a hypothesis, not a fact. Correlation matrices describe past joint movement,
which can break in regime shifts. Kalman residual autocorrelation tells you
whether the model captures structure, not where prices go next.

Anyone using these outputs to size positions or time entries should:

  1. Walk-forward backtest any rule expressed on top of them.
  2. Model transaction costs, slippage, and rebalancing taxes explicitly.
  3. Regime-segment the data — a signal that worked 2006-2019 may be inverted
     in 2024+ if structural conditions changed (and Pattern 3 below quantifies
     exactly when the structure broke).

If you want a forecasting system, build it on top of these primitives — but
treat any apparent edge with extreme skepticism until proven on data the
parameters never saw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rollender_stein.persist import DEFAULT_DERIVED_ROOT


@dataclass(frozen=True)
class ValuationZScoreRow:
    ticker: str
    n_obs: int
    current: float
    lt_geom_mean: float          # geometric mean (exp of mean log)
    z_score: float               # (log(current) - mean log) / std log
    drawdown_pct: float          # current vs all-time peak, %
    momentum_12m_log_pct: float  # log %-change over last 252 trading days
    momentum_60m_log_pct: float  # log %-change over last 1260 trading days


def compute_valuation_z_scores(
    division_dir: Path,
    *,
    column: str = "asset_in_gold",
    min_obs: int = 252,
) -> pd.DataFrame:
    """For each division-array parquet under ``division_dir``, summarize where
    the asset's current valuation sits vs its own history.

    Z-score is computed in log space so multiplicative moves are symmetric:
    a 2x asset and a 0.5x asset get equally large |z|.

    Returns a DataFrame indexed by ticker, sorted by z_score descending.
    """
    rows: list[ValuationZScoreRow] = []
    for f in sorted(division_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if column not in df.columns:
            continue
        s = df[column].dropna()
        if len(s) < min_obs:
            continue
        log_s = np.log(s)
        mean_log = float(log_s.mean())
        std_log = float(log_s.std())
        if std_log <= 0 or not np.isfinite(std_log):
            continue
        current = float(s.iloc[-1])
        z = (np.log(current) - mean_log) / std_log
        peak = float(s.max())
        drawdown_pct = (current / peak - 1) * 100

        if len(s) > 252:
            mom_12m = float((np.log(s.iloc[-1]) - np.log(s.iloc[-252])) * 100)
        else:
            mom_12m = float("nan")
        if len(s) > 252 * 5:
            mom_60m = float((np.log(s.iloc[-1]) - np.log(s.iloc[-252 * 5])) * 100)
        else:
            mom_60m = float("nan")

        rows.append(
            ValuationZScoreRow(
                ticker=f.stem,
                n_obs=len(s),
                current=current,
                lt_geom_mean=float(np.exp(mean_log)),
                z_score=float(z),
                drawdown_pct=float(drawdown_pct),
                momentum_12m_log_pct=mom_12m,
                momentum_60m_log_pct=mom_60m,
            )
        )

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "n_obs", "current", "lt_geom_mean", "z_score", "drawdown_pct",
                "momentum_12m_log_pct", "momentum_60m_log_pct",
            ]
        )
        empty.index.name = "ticker"
        return empty
    out = pd.DataFrame([r.__dict__ for r in rows]).set_index("ticker")
    return out.sort_values("z_score", ascending=False)


def compute_correlation_matrix(
    division_dir: Path,
    *,
    column: str = "asset_in_gold",
    return_window: int = 21,
    min_overlap: int = 252 * 5,
) -> pd.DataFrame:
    """Cross-asset correlation in log-returns of ``column`` at ``return_window``.

    Returns a (n_assets x n_assets) DataFrame. Tickers with fewer than
    ``min_overlap`` non-NaN return observations are dropped.
    """
    returns: dict[str, pd.Series] = {}
    for f in sorted(division_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if column not in df.columns:
            continue
        s = df[column].dropna()
        if len(s) < min_overlap + return_window:
            continue
        # log returns to keep multiplicative moves symmetric
        with np.errstate(invalid="ignore"):
            r = np.log(s).diff(return_window).dropna()
        returns[f.stem] = r

    if not returns:
        return pd.DataFrame()

    df = pd.DataFrame(returns)
    df = df.dropna(axis=1, thresh=int(len(df) * 0.5))
    return df.corr()


@dataclass(frozen=True)
class KalmanResidualDiagnostics:
    n_obs: int
    mean: float
    std: float
    autocorr_1: float
    autocorr_5: float
    autocorr_21: float
    autocorr_63: float
    recent_12m_std: float
    recent_to_alltime_std_ratio: float
    last_residual: float
    last_residual_in_recent_sigmas: float


def compute_kalman_residual_diagnostics(
    residuals_path: Path,
) -> KalmanResidualDiagnostics:
    """Diagnostics on the Phase-4 Kalman residuals.

    Low |AR(k)| → model captures structure. A spike in recent residual
    variance vs all-time variance signals a regime shift the structural
    model can no longer explain — important context for any signal built
    on top of N_Gold.
    """
    res = pd.read_parquet(residuals_path)["residual"].dropna()
    n = len(res)
    if n < 252:
        raise RuntimeError(f"residuals series too short ({n} obs)")
    recent = res.iloc[-252:]
    alltime_std = float(res.std())
    recent_std = float(recent.std())
    return KalmanResidualDiagnostics(
        n_obs=n,
        mean=float(res.mean()),
        std=alltime_std,
        autocorr_1=float(res.autocorr(lag=1)),
        autocorr_5=float(res.autocorr(lag=5)),
        autocorr_21=float(res.autocorr(lag=21)),
        autocorr_63=float(res.autocorr(lag=63)),
        recent_12m_std=recent_std,
        recent_to_alltime_std_ratio=recent_std / alltime_std if alltime_std > 0 else float("nan"),
        last_residual=float(res.iloc[-1]),
        last_residual_in_recent_sigmas=(
            float(res.iloc[-1] / recent_std) if recent_std > 0 else float("nan")
        ),
    )


def dump_pattern_report(
    *,
    root: Path = DEFAULT_DERIVED_ROOT,
    column: str = "asset_in_gold",
) -> dict[str, Any]:
    """Compute all three patterns and save under ``root/patterns/``.

    Outputs:
      - patterns/valuation_z_scores.parquet
      - patterns/correlation_matrix.parquet
      - patterns/kalman_residual_diagnostics.json
    """
    patterns_dir = root / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)

    z = compute_valuation_z_scores(root / "divisions", column=column)
    z.to_parquet(patterns_dir / "valuation_z_scores.parquet")

    corr = compute_correlation_matrix(root / "divisions", column=column)
    corr.to_parquet(patterns_dir / "correlation_matrix.parquet")

    diag = compute_kalman_residual_diagnostics(
        root / "kalman" / "residuals.parquet"
    )
    diag_dict = diag.__dict__.copy()
    (patterns_dir / "kalman_residual_diagnostics.json").write_text(
        json.dumps(diag_dict, indent=2, sort_keys=True)
    )

    return {
        "z_scores_rows": len(z),
        "correlation_assets": len(corr),
        "kalman_diagnostics": diag_dict,
    }
