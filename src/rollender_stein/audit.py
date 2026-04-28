"""Phase 7 — Truncation Hash Audit (look-ahead software guard).

**Scope (audit finding 12.M-6):** this is a *software guard* against
implementation regressions, not a theoretical Kalman test. For any
correctly-implemented deterministic Kalman filter the hash is theorem-
guaranteed to match — that's exactly the point. The detection set is
empty for any sane implementation today, so a passing run says nothing
new about the math.

What the test catches is *future code drift* — a refactor that
accidentally substitutes ``smoothed_state`` for ``filtered_state``, a
junior engineer who passes the wrong slice, a dependency upgrade that
changes statsmodels' state-vector layout. In that scenario the recursion
stops being a deterministic function of (params, data alone) and the
hash diverges. Treat a failure as "an implementation invariant broke";
do not infer "the Kalman math is wrong".

The literal spec test (refit MLE on truncated data; compare the filtered
state at the truncation point) conflates parameter-estimation drift with
state-recursion leakage — under MLE refit, parameters change across
observation sets even when no future data leaks into the recursion, so
the literal test is uninformative.

We instead use the **frozen-params variant**: fit MLE once on the full
panel, freeze the resulting parameter vector, then run the Kalman filter
at those frozen parameters on a prefix of the data. The filtered state at
the truncation point must be IDENTICAL (to numerical precision) between
the full-panel run and the truncated run.
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
            f"look-ahead software guard tripped at {truncate_at.date()}: "
            f"full={full_state_at_t!r} vs truncated={trunc_state_at_t!r}, "
            f"abs_diff={abs_diff:.3e} > tol={tol:.3e}. "
            "The Kalman recursion at frozen params should be a deterministic "
            "function of (params, data); a divergence here means an "
            "implementation invariant broke (e.g. filtered/smoothed mix-up, "
            "wrong index slice, statsmodels state-vector layout drift). "
            "Investigate the recent code path; do NOT infer the Kalman math "
            "itself is wrong."
        )

    return TruncationHashResult(
        truncate_at=truncate_at,
        full_state=full_state_at_t,
        truncated_state=trunc_state_at_t,
        abs_diff=abs_diff,
        matches_to_decimals=matches,
    )
