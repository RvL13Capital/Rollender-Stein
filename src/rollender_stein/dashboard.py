"""Phase 6 — 3D Phase-Space Attractor dashboard.

Renders the asset's trajectory through the multidimensional valuation space
defined by the AVE numéraires. The 3D plot uses three of the four ``Asset_in_X``
arrays as spatial axes; chronological time is encoded in the marker color
gradient (4th dimension); nominal USD price drives marker size (5th dimension)
to make the "Fiat Illusion" visually obvious — markers swelling without
trajectory motion means nominal price growth that buys nothing absolute.
"""

from __future__ import annotations

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


def _resolve_axis(da_frame: pd.DataFrame, key: str) -> pd.Series:
    if key not in da_frame.columns:
        raise KeyError(
            f"axis {key!r} not present in division array; "
            f"available: {[c for c in da_frame.columns if c.startswith('asset_in')]}"
        )
    return da_frame[key]


def build_phase_space_figure(
    da: DivisionArray,
    *,
    x: str = "asset_in_energy",
    y: str = "asset_in_liquidity",
    z: str = "asset_in_gold",
    title: str = "Absolute Valuation Engine: 3D Phase-Space Attractor",
    marker_scale: float | None = None,
) -> go.Figure:
    """Build the interactive 3D figure.

    Parameters
    ----------
    da
        The division array from ``valuation.build_division_array``.
    x, y, z
        Column names selecting which Asset_in_X series go on each axis.
        Default (Energy, Liquidity, Gold) matches the spec; substitute
        ``"asset_in_time"`` for any axis whose numéraire is unavailable.
    title
        Figure title.
    marker_scale
        Divide nominal USD by this to size markers. Auto-scales if None
        (target ~10px median).
    """
    df = da.to_frame().dropna(subset=[x, y, z])

    if df.empty:
        raise RuntimeError(
            "no rows with all three axes populated; check numéraire coverage"
        )

    if marker_scale is None:
        marker_scale = max(float(df["nominal_usd"].median()) / 10.0, 1.0)

    time_numeric = df.index.astype("int64").to_numpy() / 1e9
    sizes = (df["nominal_usd"] / marker_scale).clip(lower=2, upper=40).to_numpy()

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
            ),
            text=df.index.strftime("%Y-%m-%d"),
            customdata=np.stack(
                [
                    df["nominal_usd"].to_numpy(),
                    (
                        df["asset_in_time"].to_numpy()
                        if "asset_in_time" in df.columns
                        else np.full(len(df), np.nan)
                    ),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>Date:</b> %{text}<br>"
                "<b>Nominal Fiat Price:</b> $%{customdata[0]:,.2f}<br>"
                "<b>Time Multiplier:</b> %{customdata[1]:.2f}x<br>"
                "<hr>"
                f"{x}: %{{x:.2f}}x<br>"
                f"{y}: %{{y:.2f}}x<br>"
                f"{z}: %{{z:.2f}}x<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=AXIS_LABELS.get(x, x),
            yaxis_title=AXIS_LABELS.get(y, y),
            zaxis_title=AXIS_LABELS.get(z, z),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            xaxis=dict(zeroline=True, zerolinewidth=3, zerolinecolor="red"),
            yaxis=dict(zeroline=True, zerolinewidth=3, zerolinecolor="red"),
            zaxis=dict(zeroline=True, zerolinewidth=3, zerolinecolor="red"),
        ),
        template="plotly_dark",
    )
    return fig


def save_dashboard_html(fig: go.Figure, path: str | "PathLike") -> None:
    """Write a self-contained HTML rendering of ``fig``."""
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
