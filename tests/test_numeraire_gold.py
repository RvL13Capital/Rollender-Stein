from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rollender_stein.numeraires.gold import EXOG_COLS, fit_gold_model


def _synthetic_panel(n_obs: int = 400, seed: int = 42) -> pd.DataFrame:
    """Simulate the model exactly: y = mu + beta·x + eps, mu random walk."""
    rng = np.random.default_rng(seed)

    eta = rng.normal(0.0, 5.0, n_obs)
    mu = 800.0 + np.cumsum(eta)

    tips = rng.normal(1.5, 1.0, n_obs)
    dxy = 100.0 + np.cumsum(rng.normal(0.0, 0.3, n_obs))
    vix = np.clip(rng.normal(20.0, 5.0, n_obs), 8.0, 80.0)

    beta = np.array([-50.0, -2.0, 0.5])
    eps = rng.normal(0.0, 10.0, n_obs)
    y = mu + beta[0] * tips + beta[1] * dxy + beta[2] * vix + eps

    dates = pd.bdate_range("2003-01-02", periods=n_obs)
    return pd.DataFrame(
        {"XAU": y, "TIPS": tips, "DXY": dxy, "VIX": vix},
        index=dates,
    )


def test_fit_returns_filtered_state_indexed_by_clean_panel() -> None:
    panel = _synthetic_panel(n_obs=200)
    fit = fit_gold_model(panel)

    assert len(fit.filtered_state) == len(fit.panel_clean) == 200
    assert fit.filtered_state.index.equals(fit.panel_clean.index)
    assert fit.filtered_state.notna().all()


def test_fit_drops_rows_with_nan_exog() -> None:
    panel = _synthetic_panel(n_obs=200)
    panel.loc[panel.index[:50], "TIPS"] = np.nan  # simulate pre-2003

    fit = fit_gold_model(panel)
    assert len(fit.panel_clean) == 150
    assert fit.panel_clean.index.min() == panel.index[50]


def test_fit_recovers_latent_level_qualitatively() -> None:
    """The filtered state should track the simulated true level, not just be noise."""
    panel = _synthetic_panel(n_obs=500, seed=7)
    fit = fit_gold_model(panel)

    # Correlate filtered state with the true mu we simulated.
    rng = np.random.default_rng(7)
    eta = rng.normal(0.0, 5.0, 500)
    true_mu = 800.0 + np.cumsum(eta)

    corr = np.corrcoef(fit.filtered_state.values, true_mu)[0, 1]
    assert corr > 0.9, f"filtered state should track latent level (corr={corr:.3f})"


def test_fit_raises_on_missing_columns() -> None:
    panel = _synthetic_panel(n_obs=50).drop(columns=["VIX"])
    with pytest.raises(KeyError, match="VIX"):
        fit_gold_model(panel)


def test_fit_raises_on_empty_clean_panel() -> None:
    panel = _synthetic_panel(n_obs=50)
    panel["TIPS"] = np.nan
    with pytest.raises(RuntimeError, match="no rows"):
        fit_gold_model(panel)
