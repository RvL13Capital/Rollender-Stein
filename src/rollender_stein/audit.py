"""Phase 7 — Truncation Hash Audit.

Goal: prove the Kalman filter does not leak future information into past
state estimates. The literal spec test (refit MLE on truncated data; compare
the filtered state at the truncation point) conflates parameter-estimation
drift with state-recursion leakage — under MLE refit, parameters change
across observation sets even when no future data leaks into the recursion,
so the literal test is uninformative.

We instead use the **frozen-params variant**: fit MLE once on the full panel,
freeze the resulting parameter vector, then run the Kalman filter at those
frozen parameters on a prefix of the data. The filtered state at the
truncation point must be IDENTICAL (to numerical precision) between the
full-panel run and the truncated run. Any discrepancy is mathematical
proof of a leak.

Determinism note: the Kalman filter is a deterministic function of (params,
data). For an identical params vector, the recursion on a strict prefix of
the data must produce a strict prefix of the same filtered state. Even
single-bit discrepancies are bugs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from rollender_stein.numeraires.gold import EXOG_COLS, fit_gold_model


@dataclass(frozen=True)
class TruncationHashResult:
    truncate_at: pd.Timestamp
    full_state: float
    truncated_state: float
    abs_diff: float
    matches_to_decimals: int  # the largest k such that round(full, k) == round(trunc, k)


def truncation_hash_audit(
    panel: pd.DataFrame,
    truncate_at: pd.Timestamp,
    *,
    required_decimals: int = 8,
) -> TruncationHashResult:
    """Run the frozen-params truncation audit at ``truncate_at``.

    Steps:
      1. Fit the full gold model on ``panel`` via MLE; record params.
      2. Re-instantiate the model on ``panel.loc[:truncate_at]`` (inclusive).
      3. Run the Kalman filter at the frozen params (no refit).
      4. Compare ``filtered_state[0]`` at ``truncate_at`` between the two runs.

    Raises ``AssertionError`` if the absolute difference exceeds 10**(-required_decimals).
    """
    full = fit_gold_model(panel)
    if truncate_at not in full.panel_clean.index:
        raise KeyError(
            f"truncate_at {truncate_at.date()} is not in the cleaned panel index "
            f"(range {full.panel_clean.index.min().date()} .. "
            f"{full.panel_clean.index.max().date()})",
        )
    full_state_at_t = float(full.filtered_state.loc[truncate_at])

    truncated_panel = panel.loc[:truncate_at]
    truncated_clean = truncated_panel.dropna(subset=["XAU", *EXOG_COLS])
    if truncate_at not in truncated_clean.index:
        raise KeyError(f"truncate_at {truncate_at.date()} dropped from truncated clean panel")

    truncated_model = sm.tsa.UnobservedComponents(
        endog=truncated_clean["XAU"],
        level="local level",
        exog=truncated_clean[EXOG_COLS],
        initialization="approximate_diffuse",
    )
    frozen_results = truncated_model.filter(full.results.params)
    trunc_filtered = pd.Series(
        np.asarray(frozen_results.filtered_state[0]),
        index=truncated_clean.index,
    )
    trunc_state_at_t = float(trunc_filtered.loc[truncate_at])

    abs_diff = abs(full_state_at_t - trunc_state_at_t)
    tol = 10.0 ** (-required_decimals)

    # Find the largest k such that the rounded values still agree (cap at 15).
    matches = 0
    for k in range(15, -1, -1):
        if round(full_state_at_t, k) == round(trunc_state_at_t, k):
            matches = k
            break

    if abs_diff > tol:
        raise AssertionError(
            f"truncation hash audit FAILED at {truncate_at.date()}: "
            f"full={full_state_at_t!r} vs truncated={trunc_state_at_t!r}, "
            f"abs_diff={abs_diff:.3e} > tol={tol:.3e}. "
            f"This indicates state-recursion leakage; rewrite Phase 4.",
        )

    return TruncationHashResult(
        truncate_at=truncate_at,
        full_state=full_state_at_t,
        truncated_state=trunc_state_at_t,
        abs_diff=abs_diff,
        matches_to_decimals=matches,
    )
