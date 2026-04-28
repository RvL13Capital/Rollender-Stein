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


def test_animated_figure_has_chronological_frames() -> None:
    da = _toy_division_array()
    fig = build_phase_space_figure(
        da,
        x="asset_in_time",
        y="asset_in_liquidity",
        z="asset_in_gold",
        animate=True,
        frame_step=5,
        subsample=1,
    )
    assert len(fig.frames) > 0
    # Frame cardinality grows monotonically (we replay a build-up).
    from itertools import pairwise

    sizes = [len(frame.data[0].x) for frame in fig.frames]
    assert all(b >= a for a, b in pairwise(sizes)), (
        f"frame trajectory lengths should be non-decreasing, got {sizes}"
    )
    # Frame names are dates in chronological order.
    names = [frame.name for frame in fig.frames]
    parsed = pd.to_datetime(names)
    assert (parsed.to_series().diff().dropna() >= pd.Timedelta(0)).all()
    # Last frame includes every plotted row.
    last_size = sizes[-1]
    expected_rows = len(
        da.to_frame().dropna(subset=["asset_in_time", "asset_in_liquidity", "asset_in_gold"])
    )
    assert last_size == expected_rows


def test_animated_figure_has_play_and_slider() -> None:
    da = _toy_division_array()
    fig = build_phase_space_figure(
        da,
        x="asset_in_time",
        y="asset_in_liquidity",
        z="asset_in_gold",
        animate=True,
        frame_step=10,
    )
    # Play+Pause buttons configured.
    assert fig.layout.updatemenus is not None and len(fig.layout.updatemenus) == 1
    button_labels = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert any("Play" in lab for lab in button_labels)
    assert any("Pause" in lab for lab in button_labels)
    # Slider matches frame count.
    assert fig.layout.sliders is not None and len(fig.layout.sliders) == 1
    assert len(fig.layout.sliders[0].steps) == len(fig.frames)


def test_static_default_unchanged_when_animate_false() -> None:
    da = _toy_division_array()
    fig = build_phase_space_figure(
        da,
        x="asset_in_time",
        y="asset_in_liquidity",
        z="asset_in_gold",
        animate=False,
    )
    assert not fig.frames
    assert not fig.layout.updatemenus
    assert not fig.layout.sliders


# ---------------------------------------------------------------------------
# Volume / opacity integration
# ---------------------------------------------------------------------------


def _toy_division_array_with_volume(coverage: float = 1.0):
    """Toy frame with non-trivial dollar_turnover for opacity tests."""
    dates = pd.bdate_range(T0_DATE, periods=400)
    n = len(dates)
    asset = pd.Series(1500.0 * (1.0 + np.linspace(0, 1, n)), index=dates, name="spx")
    n_time = pd.Series(100.0 * (1.0 + np.linspace(0, 0.3, n)), index=dates, name="N_Time")
    n_liq = pd.Series(100.0 * (1.0 + np.linspace(0, 0.5, n)), index=dates, name="N_Liq")
    n_gold = pd.Series(100.0 * (1.0 + np.linspace(0, 0.4, n)), index=dates, name="N_Gold")
    rng = np.random.default_rng(1)
    raw_turnover = np.exp(rng.normal(15, 0.4, n))
    turnover = pd.Series(raw_turnover, index=dates, name="dollar_turnover")
    if coverage < 1.0:
        # Knock out the first (1-coverage) fraction
        n_drop = int(n * (1 - coverage))
        turnover.iloc[:n_drop] = np.nan
    return build_division_array(
        asset,
        n_time=n_time,
        n_liquidity=n_liq,
        n_gold=n_gold,
        dollar_turnover=turnover,
    )


def _alpha_from_rgba(rgba: str) -> float:
    """Parse the alpha channel out of an ``rgba(r, g, b, a)`` string."""
    body = rgba.removeprefix("rgba(").rstrip(")")
    return float(body.split(",")[3].strip())


def test_static_marker_uses_legacy_numeric_color_when_no_volume() -> None:
    """Asset without dollar_turnover renders with the legacy scalar opacity
    and numeric Viridis colormap (no per-marker alpha encoding needed)."""
    da = _toy_division_array()  # no volume
    fig = build_phase_space_figure(
        da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold"
    )
    marker = fig.data[0].marker
    # Scalar opacity preserved
    assert isinstance(marker.opacity, float | int)
    assert marker.opacity == pytest.approx(0.7)
    # Color stays the numeric time array (NOT a list of rgba strings)
    color_arr = np.asarray(marker.color)
    assert color_arr.dtype.kind in ("i", "f"), (
        f"expected numeric color array for no-volume case, got {color_arr.dtype}"
    )


def test_static_marker_color_is_rgba_array_when_volume_present() -> None:
    """With volume, per-marker alpha is baked into RGBA color strings —
    Plotly's only path to vary marker opacity per point in 3D scatter."""
    da = _toy_division_array_with_volume()
    fig = build_phase_space_figure(
        da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold"
    )
    marker = fig.data[0].marker
    # marker.opacity is now scalar 1.0; alpha lives in the color strings
    assert marker.opacity == pytest.approx(1.0)
    colors = list(marker.color)
    assert len(colors) == len(fig.data[0].x)
    assert all(isinstance(c, str) and c.startswith("rgba(") for c in colors)
    alphas = np.array([_alpha_from_rgba(c) for c in colors])
    # All alphas in the documented [0.25, 1.0] envelope
    assert alphas.min() >= 0.25 - 1e-12
    assert alphas.max() <= 1.0 + 1e-12
    # Variance across markers — i.e. opacity actually varies
    assert alphas.std() > 0.05


def test_opacity_falls_back_when_coverage_below_threshold() -> None:
    """Even with a dollar_turnover column, sparse coverage triggers the
    scalar-fallback safety path (e.g. legacy ingests, brand-new tickers)."""
    da = _toy_division_array_with_volume(coverage=0.3)
    fig = build_phase_space_figure(
        da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold"
    )
    marker = fig.data[0].marker
    assert isinstance(marker.opacity, float | int)
    assert marker.opacity == pytest.approx(0.7)
    # Color stays numeric (no rgba strings in low-coverage fallback)
    color_arr = np.asarray(marker.color)
    assert color_arr.dtype.kind in ("i", "f")


def test_opacity_falls_back_when_turnover_is_all_zeros() -> None:
    """Indexes like ^SP500TR have ``volume = 0`` in yfinance (not NULL).
    Naive NaN-coverage would pass; we explicitly require *positive* turnover
    coverage to trigger per-marker opacity, otherwise an index would render
    uniformly floor-faint."""
    dates = pd.bdate_range(T0_DATE, periods=400)
    n = len(dates)
    asset = pd.Series(1500.0 * (1.0 + np.linspace(0, 1, n)), index=dates, name="spx")
    n_time = pd.Series(100.0 * (1.0 + np.linspace(0, 0.3, n)), index=dates, name="N_Time")
    n_liq = pd.Series(100.0 * (1.0 + np.linspace(0, 0.5, n)), index=dates, name="N_Liq")
    n_gold = pd.Series(100.0 * (1.0 + np.linspace(0, 0.4, n)), index=dates, name="N_Gold")
    # Turnover is fully populated but all zero — like an index ETF placeholder
    turnover = pd.Series(np.zeros(n), index=dates, name="dollar_turnover")

    da = build_division_array(
        asset, n_time=n_time, n_liquidity=n_liq, n_gold=n_gold,
        dollar_turnover=turnover,
    )
    fig = build_phase_space_figure(
        da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold"
    )
    marker = fig.data[0].marker
    assert isinstance(marker.opacity, float | int)
    assert marker.opacity == pytest.approx(0.7)


def test_hover_template_includes_volume_lines_when_volume_present() -> None:
    da = _toy_division_array_with_volume()
    fig = build_phase_space_figure(
        da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold"
    )
    template = fig.data[0].hovertemplate
    assert "Daily Turnover" in template
    assert "Vol z-score" in template


def test_hover_template_omits_volume_lines_when_no_volume() -> None:
    da = _toy_division_array()
    fig = build_phase_space_figure(
        da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold"
    )
    template = fig.data[0].hovertemplate
    assert "Daily Turnover" not in template
    assert "Vol z-score" not in template


def test_animation_color_arrays_match_frame_lengths() -> None:
    """Per-frame the rgba color array length must equal the frame's marker
    count (otherwise Plotly silently truncates)."""
    da = _toy_division_array_with_volume()
    fig = build_phase_space_figure(
        da,
        x="asset_in_time",
        y="asset_in_liquidity",
        z="asset_in_gold",
        animate=True,
        frame_step=20,
        subsample=1,
    )
    for frame in fig.frames:
        colors = frame.data[0].marker.color
        assert hasattr(colors, "__len__")
        assert len(colors) == len(frame.data[0].x)
        # All entries are rgba strings
        assert all(isinstance(c, str) and c.startswith("rgba(") for c in colors)


def test_animation_marker_colors_stable_across_frames() -> None:
    """Color-drift regression guard. Pre-fix, ``_time_alpha_rgba`` was called
    inside ``_trace_through`` with each frame's prefix-slice of
    ``time_numeric``, so it re-normalised the Viridis scale against
    ``time_numeric[:end].min()/.max()``. Result: a marker at position N had
    a different RGB color in frame 100 (where it sat near the prefix max)
    than in frame 6500 (where it sat much earlier in the prefix). The
    historical past silently shifted color as the future arrived — a
    forensic-determinism violation.

    The fix precomputes RGBA strings once on the full subsampled array;
    frames slice the precomputed list. This test asserts that property:
    a marker at the same position has bit-identical color in every frame
    that contains it."""
    da = _toy_division_array_with_volume()
    fig = build_phase_space_figure(
        da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold",
        animate=True, frame_step=20, subsample=1,
    )
    target_pos = 50
    colors_seen: set[str] = set()
    frames_containing_pos = 0
    for frame in fig.frames:
        colors = frame.data[0].marker.color
        if len(colors) > target_pos:
            colors_seen.add(colors[target_pos])
            frames_containing_pos += 1
    assert frames_containing_pos > 5, (
        "test setup error: target position not present in enough frames"
    )
    assert len(colors_seen) == 1, (
        f"marker at position {target_pos} drifted across {frames_containing_pos} "
        f"frames — saw {len(colors_seen)} distinct colors instead of 1. "
        f"This means colorscale is being re-normalised per frame, which "
        f"silently mutates historical marker hues during animation."
    )


def test_time_alpha_rgba_raises_on_nan_alpha() -> None:
    """Hard guard against the WebGL silent-fail. A NaN slipping into an
    alpha value would render as ``rgba(R, G, B, nan)`` — CSS doesn't error,
    but WebGL canvas fails to paint that marker (or the whole trace,
    browser-dependent). Better to raise loudly here than to ship a broken
    HTML the user won't notice until they hover and see nothing."""
    from rollender_stein.dashboard import _time_alpha_rgba

    time_numeric = np.array([0.0, 1.0, 2.0])
    alphas_with_nan = np.array([0.5, np.nan, 0.7])
    with pytest.raises(ValueError, match="non-finite"):
        _time_alpha_rgba(time_numeric, alphas_with_nan)

    alphas_with_inf = np.array([0.5, np.inf, 0.7])
    with pytest.raises(ValueError, match="non-finite"):
        _time_alpha_rgba(time_numeric, alphas_with_inf)


def test_time_alpha_rgba_format_is_valid_for_clean_input() -> None:
    """Clean input must produce well-formed RGBA strings parseable by CSS."""
    from rollender_stein.dashboard import _time_alpha_rgba

    time_numeric = np.array([0.0, 1.0, 2.0, 3.0])
    alphas = np.array([0.25, 0.5, 0.75, 1.0])
    out = _time_alpha_rgba(time_numeric, alphas)
    assert len(out) == 4
    for s in out:
        assert s.startswith("rgba(") and s.endswith(")"), s
        # rgba(R, G, B, A) — four comma-separated values
        body = s[5:-1]
        parts = [p.strip() for p in body.split(",")]
        assert len(parts) == 4, f"expected 4 components, got {len(parts)} in {s}"
        # First three are numeric (RGB); last is alpha float
        for p in parts[:3]:
            assert p.replace(".", "").replace("-", "").isdigit() or p == "0"
        assert 0.0 <= float(parts[3]) <= 1.0


# ---------------------------------------------------------------------------
# Recency-fade overlay
# ---------------------------------------------------------------------------


def test_recency_disabled_by_default_is_unchanged() -> None:
    """Default behaviour (recency_window_days=None) must produce a figure
    bit-identical to pre-recency behaviour — backwards compatibility."""
    da = _toy_division_array_with_volume()
    fig_a = build_phase_space_figure(
        da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold",
    )
    fig_b = build_phase_space_figure(
        da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold",
        recency_window_days=None,
    )
    # Same opacity-derived alpha values (parsed from rgba colors)
    a = [_alpha_from_rgba(c) for c in fig_a.data[0].marker.color]
    b = [_alpha_from_rgba(c) for c in fig_b.data[0].marker.color]
    assert a == b


def test_recency_factor_array_basic_schedule() -> None:
    """Piecewise schedule: focus zone full, transition zone fades, ancient floor."""
    from rollender_stein.dashboard import _recency_factor_array

    # Hand-computed ages relative to focus 2025-12-31:
    #   2025-12-31 → 0 days (focus, factor=1.0)
    #   2025-09-08 → 114 days (in window<=200, factor=1.0)
    #   2025-03-16 → 290 days (fade: progress=(290-200)/400=0.225, factor≈0.798)
    #   2023-12-31 → 731 days (ancient: factor=floor=0.10)
    dates = pd.DatetimeIndex(
        ["2023-12-31", "2025-03-16", "2025-09-08", "2025-12-31"]
    )
    focus = pd.Timestamp("2025-12-31")
    factor = _recency_factor_array(
        dates, focus_date=focus, window_days=200, fade_days=400, floor=0.10,
    )
    assert factor[3] == pytest.approx(1.0)              # age 0
    assert factor[2] == pytest.approx(1.0)              # age 114, in window
    expected_fade = 1.0 - ((290 - 200) / 400) * (1.0 - 0.10)
    assert factor[1] == pytest.approx(expected_fade, abs=0.01)
    assert factor[0] == pytest.approx(0.10)             # ancient


def test_recency_factor_validation() -> None:
    from rollender_stein.dashboard import _recency_factor_array

    idx = pd.DatetimeIndex(["2024-01-01", "2024-06-01"])
    focus = pd.Timestamp("2024-06-01")
    with pytest.raises(ValueError, match="window_days"):
        _recency_factor_array(idx, focus_date=focus, window_days=-1, fade_days=10, floor=0.1)
    with pytest.raises(ValueError, match="fade_days"):
        _recency_factor_array(idx, focus_date=focus, window_days=10, fade_days=0, floor=0.1)
    with pytest.raises(ValueError, match="floor"):
        _recency_factor_array(idx, focus_date=focus, window_days=10, fade_days=10, floor=1.5)


def test_recency_fades_old_markers_more_than_recent() -> None:
    """End-to-end: with recency on, the earliest plotted marker has lower
    alpha than the latest one. Without recency, the relationship is governed
    only by the volume z-score (essentially no monotonic trend on synthetic
    data)."""
    da = _toy_division_array_with_volume()
    fig = build_phase_space_figure(
        da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold",
        recency_window_days=30, recency_fade_days=60, recency_floor=0.10,
    )
    alphas = np.array([_alpha_from_rgba(c) for c in fig.data[0].marker.color])
    # The earliest marker should be near the floor (or below if volume also
    # contributes a fade); the latest should be in the upper band.
    assert alphas[0] <= 0.20, f"earliest alpha={alphas[0]:.3f} should be near floor 0.10"
    assert alphas[-1] >= 0.40, f"latest alpha={alphas[-1]:.3f} should be in focus zone"


def test_recency_promotes_scalar_fallback_to_per_marker_array() -> None:
    """Asset without volume normally renders with scalar opacity 0.7. With
    recency on, the scalar gets broadcast and decayed → per-marker array."""
    da = _toy_division_array()  # no volume
    fig = build_phase_space_figure(
        da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold",
        recency_window_days=20, recency_fade_days=40, recency_floor=0.05,
    )
    marker = fig.data[0].marker
    # Color now an rgba array (was numeric in the no-volume baseline test)
    colors = list(marker.color)
    assert all(isinstance(c, str) and c.startswith("rgba(") for c in colors)
    alphas = np.array([_alpha_from_rgba(c) for c in colors])
    # Old end faded
    assert alphas[0] < alphas[-1]
    # Recent end at the volume fallback level (0.7) since no volume to modulate
    assert alphas[-1] == pytest.approx(0.7, abs=0.05)


def test_recency_focus_date_can_be_shifted_to_past() -> None:
    """Focus date defaults to last point but can be set explicitly. Setting
    it to the middle of the trajectory shifts the bright window backwards."""
    da = _toy_division_array_with_volume()
    full = pd.DataFrame(da.to_frame())
    midpoint = full.index[len(full) // 2]
    fig = build_phase_space_figure(
        da, x="asset_in_time", y="asset_in_liquidity", z="asset_in_gold",
        recency_window_days=10, recency_fade_days=20, recency_floor=0.10,
        focus_date=midpoint,
    )
    alphas = np.array([_alpha_from_rgba(c) for c in fig.data[0].marker.color])
    text_dates = list(fig.data[0].text)
    # Marker at midpoint should be bright; first AND last should be faded
    mid_idx = text_dates.index(midpoint.strftime("%Y-%m-%d"))
    assert alphas[mid_idx] > alphas[0]
    assert alphas[mid_idx] > alphas[-1]


def test_subsampling_preserves_full_resolution_zscore() -> None:
    """Audit point: opacity must be computed at FULL resolution before
    subsampling, then sliced. This guarantees the rolling-window definition
    (252 trading days) stays correct regardless of the animation stride.

    Validation: the per-marker alpha for a date that appears in BOTH the
    full and subsampled figures must be bit-for-bit identical."""
    da = _toy_division_array_with_volume()
    fig_full = build_phase_space_figure(
        da,
        x="asset_in_time",
        y="asset_in_liquidity",
        z="asset_in_gold",
        subsample=1,
    )
    fig_sub = build_phase_space_figure(
        da,
        x="asset_in_time",
        y="asset_in_liquidity",
        z="asset_in_gold",
        subsample=4,
    )
    alphas_full = np.array([_alpha_from_rgba(c) for c in fig_full.data[0].marker.color])
    alphas_sub = np.array([_alpha_from_rgba(c) for c in fig_sub.data[0].marker.color])
    text_full = list(fig_full.data[0].text)
    text_sub = list(fig_sub.data[0].text)
    matched = 0
    for date_str, a_sub in zip(text_sub, alphas_sub, strict=True):
        if date_str in text_full:
            a_full = alphas_full[text_full.index(date_str)]
            assert a_sub == pytest.approx(a_full, abs=1e-3), (
                f"date {date_str}: subsampled alpha {a_sub} ≠ full {a_full}"
            )
            matched += 1
    assert matched > 10, f"only {matched} dates overlapped — subsampling broken"
