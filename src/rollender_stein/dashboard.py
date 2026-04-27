"""Phase 6 — 3D Phase-Space Attractor dashboard.

Renders the asset's trajectory through the multidimensional valuation space
defined by the AVE numéraires. The plot encodes up to **six** dimensions:

- 3 spatial axes: ``Asset_in_X`` for X ∈ {Energy, Liquidity, Gold, Time}
- Marker / line color: chronological time (Viridis gradient)
- Marker size: nominal USD price — lets the "Fiat Illusion" jump out
  visually (a marker swelling without trajectory motion = nominal growth
  that buys nothing absolute).
- Marker opacity: rolling 252-day z-score of dollar turnover, mapped to
  [0.25, 1.0]. Solid markers = the crowd transacted heavily at this
  purchasing-power coordinate (real conviction). Faint markers = drift on
  thin volume. Falls back to a constant 0.7 when the asset has no usable
  volume (indexes, futures, sparse coverage).

Two render modes:
- ``animate=False`` (default): static figure showing the full trajectory at once.
- ``animate=True``: keyframed build-up — frames advance in chronological order,
  controlled by Play/Pause buttons and a date slider. Frame cadence is set by
  ``frame_step`` (in trading days). Default 21 days ≈ monthly.
"""

from __future__ import annotations

from os import PathLike
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from rollender_stein.valuation import DivisionArray
from rollender_stein.volume import coverage_ratio, rolling_volume_zscore

# Opacity-encoding constants (volume z-score → marker opacity).
_OPACITY_FLOOR = 0.25  # below this is invisible to the eye; preserves "drift" markers
_OPACITY_DEFAULT = 0.7  # used when volume coverage is too sparse to z-score reliably
_OPACITY_COVERAGE_THRESHOLD = 0.5  # require ≥50% non-NaN turnover to enable per-marker opacity
# Z-score → opacity mapping: opacity = clip(0.5 + 0.25 * z, floor, 1.0).
# z = 0 → 0.5 (mid); z = +2 → 1.0 (cap); z ≤ -1 → floor.
_OPACITY_Z_OFFSET = 0.5
_OPACITY_Z_SLOPE = 0.25
# Warm-up rows (where the rolling window has no z-score yet) collapse to floor —
# treated as "we don't yet know how much conviction this represents."
_OPACITY_WARMUP_FILL_Z = -2.0

AXIS_LABELS: dict[str, str] = {
    "asset_in_energy": "Physics (Thermodynamic MWh Value)",
    "asset_in_liquidity": "Printer (Global Systemic M2 Value)",
    "asset_in_gold": "Money (Filtered Core Gold Value)",
    "asset_in_time": "Time (Human Labor / AHETPI)",
}


def _scene_layout(x: str, y: str, z: str, df: pd.DataFrame) -> dict[str, Any]:
    """Locked scene config — fixed axis ranges so animation doesn't auto-rescale."""

    def _range(col: str) -> list[float]:
        lo, hi = float(df[col].min()), float(df[col].max())
        pad = max((hi - lo) * 0.05, 1.0)
        return [lo - pad, hi + pad]

    return dict(
        xaxis_title=AXIS_LABELS.get(x, x),
        yaxis_title=AXIS_LABELS.get(y, y),
        zaxis_title=AXIS_LABELS.get(z, z),
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        # uirevision keeps camera + aspect across frames; without it, Plotly
        # resets the 3D view on every frame change → the visible "jumping".
        uirevision="ave-phase-space",
        aspectmode="cube",
        xaxis=dict(
            range=_range(x),
            autorange=False,
            zeroline=True,
            zerolinewidth=3,
            zerolinecolor="red",
        ),
        yaxis=dict(
            range=_range(y),
            autorange=False,
            zeroline=True,
            zerolinewidth=3,
            zerolinecolor="red",
        ),
        zaxis=dict(
            range=_range(z),
            autorange=False,
            zeroline=True,
            zerolinewidth=3,
            zerolinecolor="red",
        ),
    )


def _hovertemplate(x: str, y: str, z: str, *, with_volume: bool = False) -> str:
    # Audit B-Minor / patch 04: the values on each axis are USD (T0-deflated),
    # not multipliers. Format with $ prefix and no "x" suffix to match.
    # customdata[1] (asset_in_time) is also a USD value — same convention.
    head = (
        "<b>Date:</b> %{text}<br>"
        "<b>Nominal Fiat Price:</b> $%{customdata[0]:,.2f}<br>"
        "<b>Time-deflated USD:</b> $%{customdata[1]:,.2f}<br>"
    )
    if with_volume:
        # customdata[2] = dollar turnover (USD), customdata[3] = vol z-score.
        head += (
            "<b>Daily Turnover:</b> $%{customdata[2]:,.0f}<br>"
            "<b>Vol z-score (252d):</b> %{customdata[3]:+.2f}<br>"
        )
    return head + (
        "<hr>"
        f"{x}: $%{{x:,.2f}}<br>"
        f"{y}: $%{{y:,.2f}}<br>"
        f"{z}: $%{{z:,.2f}}<extra></extra>"
    )


def _customdata_array(
    df: pd.DataFrame,
    *,
    z_series: pd.Series | None = None,
) -> np.ndarray:
    nominal = df["nominal_usd"].to_numpy()
    in_time = (
        df["asset_in_time"].to_numpy()
        if "asset_in_time" in df.columns
        else np.full(len(df), np.nan)
    )
    layers = [nominal, in_time]
    if z_series is not None:
        turnover = (
            df["dollar_turnover"].to_numpy()
            if "dollar_turnover" in df.columns
            else np.full(len(df), np.nan)
        )
        layers.append(turnover)
        layers.append(z_series.to_numpy())
    return np.stack(layers, axis=-1)


def _zscore_to_opacity(z: pd.Series) -> np.ndarray:
    """Map a volume z-score series to per-row opacity in [floor, 1.0].

    Warm-up rows (NaN z-score) collapse to the floor — visually marked as
    "low conviction, possibly because the rolling window hasn't filled yet,"
    which is the right epistemic stance.
    """
    filled = z.fillna(_OPACITY_WARMUP_FILL_Z)
    op = (_OPACITY_Z_OFFSET + _OPACITY_Z_SLOPE * filled).clip(_OPACITY_FLOOR, 1.0)
    return op.to_numpy()


def _time_alpha_rgba(time_numeric: np.ndarray, alphas: np.ndarray) -> list[str]:
    """Encode (time-color * per-marker alpha) as ``rgba(...)`` strings.

    Plotly 3D markers accept an array of color strings, which is the only
    supported way to vary alpha *per marker*. ``marker.opacity`` is scalar
    only; baking the alpha into RGBA color strings is the documented
    workaround. The line trace continues to use ``colorscale="Viridis"`` over
    the numeric ``time_numeric`` array, so the time gradient stays visible.

    **Caller responsibility — call this exactly once per figure** on the
    full (subsampled) arrays, then slice the result for animation frames.
    Per-frame recomputation is O(N²) total work AND would re-normalize
    ``time_numeric`` against each frame's prefix (not the full sample),
    which silently shifts the color of every historical marker as the
    animation plays — a forensic-determinism violation.
    """
    # Hard guard against silent WebGL crashes: a NaN slipping through to
    # ``rgba(R, G, B, nan)`` does not error in CSS parsing — the canvas
    # quietly fails to paint that marker (or the whole trace, depending on
    # browser). Better to raise loudly here than to ship a broken HTML.
    if not np.isfinite(alphas).all():
        raise ValueError(
            "non-finite values in marker alpha array — would corrupt RGBA "
            "strings into 'rgba(r, g, b, nan)' and silently fail in WebGL."
        )

    lo = float(time_numeric.min())
    hi = float(time_numeric.max())
    span = hi - lo if hi > lo else 1.0
    normed = (time_numeric - lo) / span
    rgb_strs = sample_colorscale("Viridis", normed.tolist())
    # Plotly's sample_colorscale returns "rgb(R, G, B)" deterministically.
    # Slicing [4:-1] strips "rgb(" and ")" at C-level — much faster than
    # `.removeprefix().rstrip()` inside a Python hot loop over thousands of
    # markers. List-comp is the right shape here (CPython optimises it
    # better than np.char vectorised string ops for short f-string outputs).
    return [
        f"rgba({rgb[4:-1]}, {float(a):.4f})"
        for rgb, a in zip(rgb_strs, alphas, strict=True)
    ]


def _marker_dict(
    *,
    sizes: np.ndarray,
    color_data: np.ndarray | list[str],
    color_range: tuple[float, float],
    is_rgba: bool,
    fallback_opacity: float,
) -> dict[str, Any]:
    """Build a single ``marker=dict(...)`` payload from precomputed inputs.

    ``color_data`` is whatever the caller decided to pass — either a slice
    of the precomputed RGBA-string list (when ``is_rgba=True``) or a slice
    of the numeric ``time_numeric`` array (when ``is_rgba=False``). This
    function does NO normalisation, NO RGBA computation, NO hot-path
    string formatting — those happen once in ``build_phase_space_figure``
    on the full data, then are sliced into frames.

    The split is deliberate: per-frame work is now O(slice-size) data
    copying only, no recomputation. Pre-fix, every animation frame called
    ``_time_alpha_rgba`` on its own prefix slice, which (a) was O(N²)
    cumulative, and (b) re-normalised the Viridis scale against each
    frame's prefix-min/max, causing every historical marker's hue to
    shift as the animation extended — a silent forensic-determinism
    violation.
    """
    if is_rgba:
        return dict(size=sizes, color=color_data, opacity=1.0)
    return dict(
        size=sizes,
        color=color_data,
        colorscale="Viridis",
        opacity=fallback_opacity,
        showscale=False,
        cmin=color_range[0],
        cmax=color_range[1],
    )


def _resolve_opacity(df_full: pd.DataFrame) -> tuple[np.ndarray | float, pd.Series | None]:
    """Decide whether to use per-marker opacity from volume z-scores.

    Returns ``(opacity, z_series)``:
      - ``opacity`` is either a numpy array (one entry per row of ``df_full``)
        or the scalar fallback ``_OPACITY_DEFAULT``.
      - ``z_series`` is the underlying z-score series when per-marker opacity
        is active; ``None`` when falling back. Used by the hover template.

    Falls back to scalar opacity in three situations:
      1. No ``dollar_turnover`` column at all (legacy DivisionArrays).
      2. Sparse coverage of *positive* turnover — both NaN and zero are
         disqualifying (yfinance stores 0 for index-only series like
         ``^SP500TR`` rather than NULL, so a NaN-only check would slip past).
      3. The resulting z-score series itself is mostly NaN (defensive — for
         e.g. all-equal turnover where rolling std is 0).
    """
    if "dollar_turnover" not in df_full.columns:
        return _OPACITY_DEFAULT, None
    turnover = df_full["dollar_turnover"]
    # Both NaN and zero are "no real trading happened that day". Mask zeros
    # to NaN before measuring coverage so all-zero series (e.g. ^SP500TR
    # index data) fall through to the scalar fallback rather than rendering
    # uniformly floor-opacity markers.
    positive = turnover.where(turnover > 0)
    if coverage_ratio(positive) < _OPACITY_COVERAGE_THRESHOLD:
        return _OPACITY_DEFAULT, None
    z = rolling_volume_zscore(turnover)
    if coverage_ratio(z) < _OPACITY_COVERAGE_THRESHOLD:
        return _OPACITY_DEFAULT, None
    return _zscore_to_opacity(z), z


def build_phase_space_figure(
    da: DivisionArray,
    *,
    x: str = "asset_in_energy",
    y: str = "asset_in_liquidity",
    z: str = "asset_in_gold",
    title: str = "Absolute Valuation Engine: 3D Phase-Space Attractor",
    marker_scale: float | None = None,
    animate: bool = False,
    frame_step: int = 21,
    frame_duration_ms: int = 50,
    subsample: int | None = None,
) -> go.Figure:
    """Build the interactive 3D figure.

    Parameters
    ----------
    da
        The division array from ``valuation.build_division_array``.
    x, y, z
        Column names selecting which ``asset_in_X`` series go on each axis.
        Default (Energy, Liquidity, Gold) matches the spec; substitute
        ``"asset_in_time"`` for any axis whose numéraire is unavailable.
    title
        Figure title.
    marker_scale
        Divide nominal USD by this to size markers. Auto-scales if ``None``.
    animate
        If True, return a keyframed animation that builds the trajectory in
        chronological order with Play/Pause controls and a date slider.
    frame_step
        Trading-day stride between animation frames. 21 ≈ monthly. Smaller
        values produce smoother animation but larger HTML files.
    frame_duration_ms
        Per-frame transition duration when Play is pressed.
    subsample
        Take every N-th row before plotting. ``None`` = auto: 1 for static,
        ``ceil(len/1500)`` for animated mode (keeps HTML file size manageable
        — every frame stores its full cumulative trajectory, so unbounded
        daily resolution produces tens of MB of redundant data).
    """
    df_full = da.to_frame().dropna(subset=[x, y, z])

    if df_full.empty:
        raise RuntimeError(
            "no rows with all three axes populated; check numéraire coverage"
        )

    # Opacity from volume z-score is computed on the FULL daily resolution.
    # Subsampling for animation must happen *after* — sub-sampling first and
    # then z-scoring would inflate the effective rolling window (e.g.
    # 252 daily points becomes 252 * subsample trading days), distorting the
    # statistic. We slice opacity by the same positional indexer as the df.
    opacity_full, z_full = _resolve_opacity(df_full)

    if subsample is None:
        subsample = max(1, -(-len(df_full) // 1500)) if animate else 1
    if subsample > 1:
        # Build a positional index list, then take positions on both df and
        # opacity / z. Always include the actual last row — `iloc[::N]` drops
        # it when len(df) is not a multiple of N+1 (e.g. len=5110, step=4
        # misses the final date by 1).
        position_list = list(range(0, len(df_full), subsample))
        if position_list[-1] != len(df_full) - 1:
            position_list.append(len(df_full) - 1)
        positions = np.asarray(position_list, dtype=np.int64)
        df = df_full.iloc[positions]
        opacity = (
            opacity_full[positions]
            if isinstance(opacity_full, np.ndarray)
            else opacity_full
        )
        z_sub = z_full.iloc[positions] if z_full is not None else None
    else:
        df = df_full
        opacity = opacity_full
        z_sub = z_full

    if marker_scale is None:
        marker_scale = max(float(df["nominal_usd"].median()) / 10.0, 1.0)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DivisionArray frame must have a DatetimeIndex")
    time_numeric = df.index.astype("int64").to_numpy() / 1e9
    sizes = (df["nominal_usd"] / marker_scale).clip(lower=2, upper=40).to_numpy()
    customdata = _customdata_array(df, z_series=z_sub)
    text = df.index.strftime("%Y-%m-%d").to_numpy()
    color_range = (float(time_numeric.min()), float(time_numeric.max()))
    has_volume = z_sub is not None

    # Pre-compute the marker color array EXACTLY ONCE on the full (subsampled)
    # data. Animation frames will slice this — they must not recompute, both
    # for performance (O(N) instead of O(N²)) and for forensic correctness
    # (per-frame recomputation re-normalises Viridis against each prefix's
    # min/max, silently shifting historical marker hues). See _marker_dict
    # docstring for the failure mode.
    base_colors: np.ndarray | list[str]
    fallback_opacity: float
    if isinstance(opacity, np.ndarray):
        is_rgba = True
        base_colors = _time_alpha_rgba(time_numeric, opacity)
        fallback_opacity = _OPACITY_DEFAULT  # unused in rgba branch
    else:
        is_rgba = False
        base_colors = time_numeric
        fallback_opacity = float(opacity)

    if not animate:
        fig = go.Figure(
            go.Scatter3d(
                x=df[x],
                y=df[y],
                z=df[z],
                mode="lines+markers",
                line=dict(color=time_numeric, colorscale="Viridis", width=4),
                marker=_marker_dict(
                    sizes=sizes,
                    color_data=base_colors,
                    color_range=color_range,
                    is_rgba=is_rgba,
                    fallback_opacity=fallback_opacity,
                ),
                text=text,
                customdata=customdata,
                hovertemplate=_hovertemplate(x, y, z, with_volume=has_volume),
            )
        )
        fig.update_layout(
            title=title,
            scene=_scene_layout(x, y, z, df),
            template="plotly_dark",
        )
        return fig

    # ---- animated build-up ----
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")

    n = len(df)
    frame_endpoints: list[int] = list(range(frame_step - 1, n, frame_step))
    if frame_endpoints[-1] != n - 1:
        frame_endpoints.append(n - 1)

    def _trace_through(end_inclusive: int) -> go.Scatter3d:
        end = end_inclusive + 1
        # Pure slicing only — base_colors was precomputed once on the full
        # arrays. Both list[str] and np.ndarray support [:end] efficiently.
        return go.Scatter3d(
            x=df[x].iloc[:end].to_numpy(),
            y=df[y].iloc[:end].to_numpy(),
            z=df[z].iloc[:end].to_numpy(),
            mode="lines+markers",
            line=dict(
                color=time_numeric[:end],
                colorscale="Viridis",
                width=4,
                cmin=color_range[0],
                cmax=color_range[1],
            ),
            marker=_marker_dict(
                sizes=sizes[:end],
                color_data=base_colors[:end],
                color_range=color_range,
                is_rgba=is_rgba,
                fallback_opacity=fallback_opacity,
            ),
            text=text[:end],
            customdata=customdata[:end],
            hovertemplate=_hovertemplate(x, y, z, with_volume=has_volume),
        )

    initial_trace = _trace_through(0)
    frames = [
        go.Frame(
            data=[_trace_through(end_idx)],
            name=text[end_idx],
        )
        for end_idx in frame_endpoints
    ]

    fig = go.Figure(data=[initial_trace], frames=frames)

    play_args = [
        None,
        {
            "frame": {"duration": frame_duration_ms, "redraw": True},
            "fromcurrent": True,
            "transition": {"duration": 0},
            "mode": "immediate",
        },
    ]
    pause_args = [
        [None],
        {
            "frame": {"duration": 0, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 0},
        },
    ]

    slider_steps = [
        {
            "args": [
                [f.name],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": f.name[:7],  # YYYY-MM
            "method": "animate",
        }
        for f in frames
    ]

    fig.update_layout(
        title=title,
        scene=_scene_layout(x, y, z, df),
        template="plotly_dark",
        uirevision="ave-phase-space",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top",
                "pad": {"t": 60, "r": 10},
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": play_args},
                    {"label": "⏸ Pause", "method": "animate", "args": pause_args},
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 14},
                    "prefix": "Datum: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": slider_steps,
            }
        ],
    )
    return fig


def save_dashboard_html(fig: go.Figure, path: str | PathLike[str]) -> None:
    """Write a self-contained HTML rendering of ``fig``."""
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
