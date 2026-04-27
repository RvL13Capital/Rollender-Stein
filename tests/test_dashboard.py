from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rollender_stein.calendar import T0_DATE
from rollender_stein.dashboard import build_phase_space_figure
from rollender_stein.valuation import build_division_array


def _toy_division_array():
    dates = pd.bdate_range(T0_DATE, "2010-01-29")
    n = len(dates)
    asset = pd.Series(1500.0 * (1.0 + np.linspace(0, 1, n)), index=dates, name="spx")
    n_time = pd.Series(100.0 * (1.0 + np.linspace(0, 0.3, n)), index=dates, name="N_Time")
    n_liq = pd.Series(100.0 * (1.0 + np.linspace(0, 0.5, n)), index=dates, name="N_Liq")
    n_gold = pd.Series(100.0 * (1.0 + np.linspace(0, 0.4, n)), index=dates, name="N_Gold")
    return build_division_array(asset, n_time=n_time, n_liquidity=n_liq, n_gold=n_gold)


def test_figure_renders_with_default_axes() -> None:
    """Default x=energy is missing here; should raise."""
    da = _toy_division_array()
    with pytest.raises(KeyError, match="asset_in_energy"):
        build_phase_space_figure(da)


def test_figure_renders_with_time_substituted_for_energy() -> None:
    da = _toy_division_array()
    fig = build_phase_space_figure(
        da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold"
    )
    # One trace, the 3D scatter
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.type == "scatter3d"
    assert len(trace.x) > 0
    # Scene axis labels should be the friendly Time/Liq/Gold strings
    assert "Time" in fig.layout.scene.xaxis.title.text
    assert "Printer" in fig.layout.scene.yaxis.title.text
    assert "Money" in fig.layout.scene.zaxis.title.text


def test_figure_drops_rows_with_any_axis_nan() -> None:
    """Rows where any of the chosen axes is NaN must be excluded."""
    da = _toy_division_array()
    # Inject NaN into asset_in_gold for first half — drop those rows from plot
    half = len(da.asset_in_gold) // 2
    da.asset_in_gold.iloc[:half] = np.nan

    fig = build_phase_space_figure(da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold")
    plotted_dates = pd.to_datetime(fig.data[0].text)
    assert len(plotted_dates) == len(da.asset_in_gold) - half
    # Earliest plotted date is past the NaN region
    assert plotted_dates.min() >= da.asset_in_gold.dropna().index.min()


def test_empty_after_dropna_raises() -> None:
    da = _toy_division_array()
    da.asset_in_gold.iloc[:] = np.nan
    with pytest.raises(RuntimeError, match="no rows"):
        build_phase_space_figure(da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold")
