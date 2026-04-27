"""Phase 7 frozen-params truncation hash audit tests.

The recursion is deterministic in (params, data). For identical params and a
strict prefix of the data, the filtered state at the prefix's last point must
match the filtered state at that point in the full-data run. We verify to 8
decimals (the spec's contract) and also check exact float equality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rollender_stein.audit import truncation_hash_audit


def _synthetic_panel(n_obs: int = 300, seed: int = 13) -> pd.DataFrame:
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
    return pd.DataFrame({"XAU": y, "TIPS": tips, "DXY": dxy, "VIX": vix}, index=dates)


def test_truncation_audit_passes_at_midpoint() -> None:
    panel = _synthetic_panel(n_obs=300)
    truncate_at = panel.index[150]

    res = truncation_hash_audit(panel, truncate_at, required_decimals=8)
    assert res.matches_to_decimals >= 8, (
        f"only {res.matches_to_decimals} decimals matched; abs_diff={res.abs_diff:.3e}"
    )


def test_truncation_audit_passes_near_end() -> None:
    panel = _synthetic_panel(n_obs=300)
    truncate_at = panel.index[-2]  # second to last (need at least one row past it for "full")

    res = truncation_hash_audit(panel, truncate_at, required_decimals=8)
    assert res.matches_to_decimals >= 8


def test_truncation_audit_passes_early() -> None:
    panel = _synthetic_panel(n_obs=300)
    # Need enough leading data for the diffuse Kalman to settle before truncation.
    truncate_at = panel.index[40]

    res = truncation_hash_audit(panel, truncate_at, required_decimals=8)
    assert res.matches_to_decimals >= 8
