"""Last-Observation-Carried-Forward anchored to release_date.

Phase 2.1 of the AVE spec. The only sanctioned mechanism for
synchronizing lower-frequency macro data into the daily AVE pipeline.

Forbidden alternatives that look correct but leak future data:
- ``DataFrame.interpolate()`` over a reference_date axis
- ``DataFrame.fillna(method='ffill')`` indexed on reference_date
- Any join keyed on reference_date instead of release_date
"""

from __future__ import annotations

import pandas as pd

RELEASE_COL_DEFAULT = "release_date"
REFERENCE_COL_DEFAULT = "reference_date"


def forward_fill_to_calendar(
    macro: pd.DataFrame,
    calendar_idx: pd.DatetimeIndex,
    *,
    release_col: str = RELEASE_COL_DEFAULT,
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Forward-fill releases onto a daily calendar with no look-ahead.

    On each day ``d`` in ``calendar_idx``, the row holds the most recent
    release whose ``release_date <= d``, or NaN if no release has occurred yet.

    Parameters
    ----------
    macro
        Frame with at least ``release_col`` (datetime) and one or more value columns.
        May include ``reference_date``; it is preserved through the join only if
        listed in ``value_cols``. Duplicate ``release_date`` values are rejected —
        deduplicate (e.g. keep the latest vintage per release date) before calling.
    calendar_idx
        The target daily index. Typically ``master_calendar()``.
    release_col
        Column name holding the public-knowledge date. Default ``"release_date"``.
    value_cols
        Columns to forward-fill. If ``None``, all columns except ``release_col``
        and ``"reference_date"`` are used.
    """
    if release_col not in macro.columns:
        raise KeyError(f"macro frame missing required column {release_col!r}")
    if not pd.api.types.is_datetime64_any_dtype(macro[release_col]):
        raise TypeError(f"{release_col!r} must be a datetime dtype")

    if value_cols is None:
        value_cols = [
            c for c in macro.columns if c not in {release_col, REFERENCE_COL_DEFAULT}
        ]
    if not value_cols:
        raise ValueError("no value columns to forward-fill")

    if macro.empty:
        # Pre-empty `merge_asof` to dodge a pandas 3.0 dtype-mismatch crash:
        # `pd.to_datetime([])` materializes as `<M8[s]>` while a populated
        # bdate_range is `<M8[us]>`; `merge_asof` then raises MergeError.
        # The semantically-correct result for "no macro releases yet ingested"
        # is a calendar-shaped frame of NaN values — exactly what callers
        # expect (forward-fill of nothing into the future is nothing).
        out = pd.DataFrame(
            {col: pd.Series(index=calendar_idx, dtype="float64") for col in value_cols},
        )
        out.index.name = "trade_date"
        return out

    macro_sorted = macro.sort_values(release_col, kind="mergesort").reset_index(drop=True)
    if macro_sorted[release_col].duplicated().any():
        raise ValueError(
            f"duplicate {release_col} values found; deduplicate before forward-filling",
        )

    left = pd.DataFrame(index=calendar_idx).reset_index(names="trade_date")
    right = macro_sorted[[release_col, *value_cols]]

    merged = pd.merge_asof(
        left=left,
        right=right,
        left_on="trade_date",
        right_on=release_col,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.set_index("trade_date").drop(columns=[release_col])
