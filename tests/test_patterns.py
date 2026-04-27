from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rollender_stein.patterns import (
    compute_correlation_matrix,
    compute_kalman_residual_diagnostics,
    compute_valuation_z_scores,
    dump_pattern_report,
)


def _seed_division(tmp_path: Path, ticker: str, values: list[float]) -> None:
    """Write a minimal division-array parquet with `asset_in_gold` only."""
    div_dir = tmp_path / "divisions"
    div_dir.mkdir(parents=True, exist_ok=True)
    idx = pd.bdate_range("2010-01-04", periods=len(values))
    df = pd.DataFrame({"asset_in_gold": values}, index=idx)
    df.index.name = "trade_date"
    df.to_parquet(div_dir / f"{ticker}.parquet")


def _seed_residuals(tmp_path: Path, values: list[float]) -> None:
    kal_dir = tmp_path / "kalman"
    kal_dir.mkdir(parents=True, exist_ok=True)
    idx = pd.bdate_range("2010-01-04", periods=len(values))
    df = pd.DataFrame({"residual": values}, index=idx)
    df.index.name = "trade_date"
    df.to_parquet(kal_dir / "residuals.parquet")


def test_z_scores_skip_short_series(tmp_path) -> None:
    _seed_division(tmp_path, "SHORT", [100.0, 105.0])  # well below min_obs
    out = compute_valuation_z_scores(tmp_path / "divisions")
    assert "SHORT" not in out.index


def test_z_scores_log_symmetry(tmp_path) -> None:
    """Asset that's 2x its lt_geom_mean has |z| == |z| of an asset at 0.5x its lt_geom_mean."""
    rng = np.random.default_rng(0)
    base = np.exp(rng.normal(0, 0.1, 1000))  # roughly log-normal around 1
    _seed_division(tmp_path, "UP", [*list(base), base[-1] * 2.0])
    _seed_division(tmp_path, "DOWN", [*list(base), base[-1] * 0.5])
    out = compute_valuation_z_scores(tmp_path / "divisions")
    assert out.loc["UP", "z_score"] == pytest.approx(-out.loc["DOWN", "z_score"], rel=0.1)


def test_z_scores_drawdown_zero_at_peak(tmp_path) -> None:
    values = [100.0 + i for i in range(300)]  # monotonic increase
    _seed_division(tmp_path, "TRENDING", values)
    out = compute_valuation_z_scores(tmp_path / "divisions")
    assert out.loc["TRENDING", "drawdown_pct"] == pytest.approx(0.0)


def test_correlation_diagonal_is_one(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n = 252 * 6
    a = np.cumsum(rng.normal(0, 0.01, n)) + 5
    b = np.cumsum(rng.normal(0, 0.01, n)) + 5
    _seed_division(tmp_path, "A", list(np.exp(a)))
    _seed_division(tmp_path, "B", list(np.exp(b)))
    corr = compute_correlation_matrix(tmp_path / "divisions")
    assert corr.shape == (2, 2)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
    assert corr.loc["B", "B"] == pytest.approx(1.0)


def test_correlation_two_perfectly_aligned_series(tmp_path) -> None:
    rng = np.random.default_rng(0)
    a = np.exp(np.cumsum(rng.normal(0, 0.01, 252 * 6)))
    _seed_division(tmp_path, "X", list(a))
    _seed_division(tmp_path, "Y", list(a * 2))  # exact same returns, double level
    corr = compute_correlation_matrix(tmp_path / "divisions")
    assert corr.loc["X", "Y"] == pytest.approx(1.0)


def test_kalman_diagnostics_white_noise_input(tmp_path) -> None:
    rng = np.random.default_rng(42)
    res = rng.normal(0, 1.0, 2000)
    _seed_residuals(tmp_path, list(res))
    diag = compute_kalman_residual_diagnostics(tmp_path / "kalman" / "residuals.parquet")
    assert diag.n_obs == 2000
    assert abs(diag.autocorr_1) < 0.1, "white noise must have ~0 AR(1)"
    assert abs(diag.std - 1.0) < 0.1


def test_kalman_diagnostics_recent_variance_blowup_detected(tmp_path) -> None:
    """Synthetic regime shift: last 252 obs have 5x the std of the prior 1500."""
    rng = np.random.default_rng(7)
    early = rng.normal(0, 1.0, 1500)
    late = rng.normal(0, 5.0, 252)
    _seed_residuals(tmp_path, list(np.concatenate([early, late])))
    diag = compute_kalman_residual_diagnostics(tmp_path / "kalman" / "residuals.parquet")
    assert diag.recent_to_alltime_std_ratio > 1.5, (
        f"recent variance blowup must surface in the ratio; got "
        f"{diag.recent_to_alltime_std_ratio}"
    )


def test_dump_pattern_report_writes_three_artifacts(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n = 252 * 6
    for tk, drift in [("A", 0.0001), ("B", -0.0001), ("C", 0.0)]:
        a = np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
        _seed_division(tmp_path, tk, list(a))
    _seed_residuals(tmp_path, list(rng.normal(0, 1.0, 2000)))

    summary = dump_pattern_report(root=tmp_path)
    assert (tmp_path / "patterns" / "valuation_z_scores.parquet").exists()
    assert (tmp_path / "patterns" / "correlation_matrix.parquet").exists()
    assert (tmp_path / "patterns" / "kalman_residual_diagnostics.json").exists()

    z = pd.read_parquet(tmp_path / "patterns" / "valuation_z_scores.parquet")
    assert set(z.index) == {"A", "B", "C"}

    diag = json.loads(
        (tmp_path / "patterns" / "kalman_residual_diagnostics.json").read_text()
    )
    assert diag["n_obs"] == 2000
    assert summary["z_scores_rows"] == 3
