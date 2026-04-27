"""Phase 6 — 3D Phase-Space Attractor dashboard.

Renders the asset's trajectory through the multidimensional valuation space
defined by the AVE numéraires. The 3D plot uses three of the four ``Asset_in_X``
arrays as spatial axes; chronological time is encoded in the marker color
gradient (4th dimension); nominal USD price drives marker size (5th dimension)
to make the "Fiat Illusion" visually obvious — markers swelling without
trajectory motion means nominal price growth that buys nothing absolute.

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

from rollender_stein.valuation import DivisionArray

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


def _hovertemplate(x: str, y: str, z: str) -> str:
    return (
        "<b>Date:</b> %{text}<br>"
        "<b>Nominal Fiat Price:</b> $%{customdata[0]:,.2f}<br>"
        "<b>Time Multiplier:</b> %{customdata[1]:.2f}x<br>"
        "<hr>"
        f"{x}: %{{x:.2f}}x<br>"
        f"{y}: %{{y:.2f}}x<br>"
        f"{z}: %{{z:.2f}}x<extra></extra>"
    )


def _customdata_array(df: pd.DataFrame) -> np.ndarray:
    nominal = df["nominal_usd"].to_numpy()
    in_time = (
        df["asset_in_time"].to_numpy()
        if "asset_in_time" in df.columns
        else np.full(len(df), np.nan)
    )
    return np.stack([nominal, in_time], axis=-1)


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
    df = da.to_frame().dropna(subset=[x, y, z])

    if df.empty:
        raise RuntimeError(
            "no rows with all three axes populated; check numéraire coverage"
        )

    if subsample is None:
        subsample = max(1, -(-len(df) // 1500)) if animate else 1
    if subsample > 1:
        sampled = df.iloc[::subsample]
        # Always include the actual last row — `iloc[::N]` drops it when
        # len(df) is not a multiple of N+1 (e.g. len=5110, step=4 misses the
        # final date by 1). Without this, animation ends a few days early.
        if not sampled.empty and sampled.index[-1] != df.index[-1]:
            sampled = pd.concat([sampled, df.iloc[[-1]]])
        df = sampled

    if marker_scale is None:
        marker_scale = max(float(df["nominal_usd"].median()) / 10.0, 1.0)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DivisionArray frame must have a DatetimeIndex")
    time_numeric = df.index.astype("int64").to_numpy() / 1e9
    sizes = (df["nominal_usd"] / marker_scale).clip(lower=2, upper=40).to_numpy()
    customdata = _customdata_array(df)
    text = df.index.strftime("%Y-%m-%d").to_numpy()
    color_range = (float(time_numeric.min()), float(time_numeric.max()))

    if not animate:
        fig = go.Figure(
            go.Scatter3d(
                x=df[x],
                y=df[y],
                z=df[z],
                mode="lines+markers",
                line=dict(color=time_numeric, colorscale="Viridis", width=4),
                marker=dict(
                    size=sizes,
                    color=time_numeric,
                    colorscale="Viridis",
                    opacity=0.7,
                    showscale=False,
                    cmin=color_range[0],
                    cmax=color_range[1],
                ),
                text=text,
                customdata=customdata,
                hovertemplate=_hovertemplate(x, y, z),
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
            marker=dict(
                size=sizes[:end],
                color=time_numeric[:end],
                colorscale="Viridis",
                opacity=0.7,
                showscale=False,
                cmin=color_range[0],
                cmax=color_range[1],
            ),
            text=text[:end],
            customdata=customdata[:end],
            hovertemplate=_hovertemplate(x, y, z),
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
