from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rollender_stein.bitemporal import insert_macro_releases, open_db
from rollender_stein.calendar import T0_DATE
from rollender_stein.numeraires.gold import (
    EXOG_COLS,
    SERIES_IDS,
    SOURCE_FRED,
    SOURCE_YAHOO,
    XAU_SERIES_ID,
    assemble_panel,
    build_n_gold,
    fit_gold_model,
)


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


# ----- assemble_panel: integration through the bitemporal store ----------------


@pytest.fixture
def gold_con():
    with open_db(":memory:") as c:
        yield c


def _seed_daily(
    con,
    series_id: str,
    source: str,
    dates: pd.DatetimeIndex,
    values: list[float],
) -> None:
    insert_macro_releases(
        con,
        series_id,
        pd.DataFrame(
            {"reference_date": dates, "release_date": dates, "value": values}
        ),
        source=source,
    )


def test_assemble_panel_returns_master_calendar_shape(gold_con) -> None:
    """assemble_panel reads each series from the bitemporal store, LOCFs onto
    the NYSE master calendar, and returns a 4-column frame indexed by trading days.
    """
    dates = pd.bdate_range("1999-12-15", "2000-01-15")
    n = len(dates)
    seeds = {
        "XAU":  (SOURCE_YAHOO, [288.0 + i * 0.10 for i in range(n)]),
        "TIPS": (SOURCE_FRED,  [3.0   + i * 0.01 for i in range(n)]),
        "DXY":  (SOURCE_FRED,  [105.0 + i * 0.05 for i in range(n)]),
        "VIX":  (SOURCE_FRED,  [22.0  - i * 0.10 for i in range(n)]),
    }
    for short, (src, vals) in seeds.items():
        _seed_daily(gold_con, SERIES_IDS[short], src, dates, vals)

    panel = assemble_panel(gold_con, end=pd.Timestamp("2000-01-15"))
    assert list(panel.columns) == ["XAU", "TIPS", "DXY", "VIX"]
    # NYSE calendar from T0=2000-01-03 to 2000-01-15 → 9 trading days
    assert T0_DATE in panel.index
    # All values present at T0 since seeds cover the period.
    t0_row = panel.loc[T0_DATE]
    assert all(pd.notna(t0_row.values)), f"unexpected NaN at T0: {t0_row.to_dict()}"
    # XAU values match the latest pre-T0 seed (LOCF picks latest release ≤ T0).
    expected_xau_at_t0 = panel.loc[T0_DATE, "XAU"]
    pre_t0_seeds = [288.0 + i * 0.1 for i, d in enumerate(dates) if d <= T0_DATE]
    assert expected_xau_at_t0 == pytest.approx(pre_t0_seeds[-1])


def test_assemble_panel_propagates_nan_for_missing_series(gold_con) -> None:
    """If TIPS has releases only after T0, the assembled panel has TIPS=NaN
    on T0 — the Kalman fit will then drop those rows."""
    dates_full = pd.bdate_range("1999-12-15", "2003-06-30")
    _seed_daily(gold_con, SERIES_IDS["XAU"], SOURCE_YAHOO, dates_full, [288.0] * len(dates_full))
    _seed_daily(gold_con, SERIES_IDS["DXY"], SOURCE_FRED, dates_full, [105.0] * len(dates_full))
    _seed_daily(gold_con, SERIES_IDS["VIX"], SOURCE_FRED, dates_full, [22.0] * len(dates_full))
    # TIPS only from 2003-01-02 onward (mirrors real DFII10 history).
    tips_dates = pd.bdate_range("2003-01-02", "2003-06-30")
    _seed_daily(gold_con, SERIES_IDS["TIPS"], SOURCE_FRED, tips_dates, [2.0] * len(tips_dates))

    panel = assemble_panel(gold_con, end=pd.Timestamp("2003-06-30"))
    assert pd.isna(panel.loc[T0_DATE, "TIPS"]), "TIPS leaked into pre-2003 rows"
    assert pd.isna(panel.loc[pd.Timestamp("2002-12-31"), "TIPS"])
    assert pd.notna(panel.loc[pd.Timestamp("2003-01-02"), "TIPS"])
    # Other series are filled throughout.
    for col in ("XAU", "DXY", "VIX"):
        assert pd.notna(panel.loc[T0_DATE, col]), f"{col} unexpectedly NaN at T0"


def test_assemble_panel_raises_when_series_not_ingested(gold_con) -> None:
    """If any of the four series has zero rows, assemble_panel must fail loudly
    (the Phase 4 model can't run without all four)."""
    # Seed only 3 out of 4
    dates = pd.bdate_range("1999-12-15", "2000-01-15")
    _seed_daily(gold_con, SERIES_IDS["XAU"], SOURCE_YAHOO, dates, [288.0] * len(dates))
    _seed_daily(gold_con, SERIES_IDS["DXY"], SOURCE_FRED, dates, [105.0] * len(dates))
    _seed_daily(gold_con, SERIES_IDS["VIX"], SOURCE_FRED, dates, [22.0] * len(dates))
    # TIPS missing entirely
    with pytest.raises(RuntimeError, match=SERIES_IDS["TIPS"]):
        assemble_panel(gold_con, end=pd.Timestamp("2000-01-15"))


def test_assemble_panel_columns_are_short_names_not_series_ids(gold_con) -> None:
    """Panel columns are the short names (XAU/TIPS/DXY/VIX) — Phase 4
    fit_gold_model expects exactly that."""
    dates = pd.bdate_range("1999-12-15", "2000-01-15")
    for short in ("XAU", "TIPS", "DXY", "VIX"):
        _seed_daily(
            gold_con,
            SERIES_IDS[short],
            SOURCE_FRED,
            dates,
            [1.0] * len(dates),
        )
    panel = assemble_panel(gold_con, end=pd.Timestamp("2000-01-15"))
    assert set(panel.columns) == {"XAU", *EXOG_COLS}
    # Real series IDs (e.g. "GC=F", "DFII10") must NOT show up as columns.
    for sid in SERIES_IDS.values():
        if sid not in {"XAU", "TIPS", "DXY", "VIX"}:
            assert sid not in panel.columns


# ----- build_n_gold (post-patch-06: raw XAU, not Kalman) -----------------------


def test_build_n_gold_anchors_at_t0_exactly_when_xau_present(gold_con) -> None:
    """If XAU has a release on or before T0, N_Gold(T0) = 100.0 exact."""
    dates = pd.bdate_range("1999-12-20", "2000-02-15")
    values = [280.0 + i * 0.5 for i in range(len(dates))]
    _seed_daily(gold_con, XAU_SERIES_ID, SOURCE_YAHOO, dates, values)

    n_gold = build_n_gold(gold_con, end=pd.Timestamp("2000-02-15"))

    assert n_gold.loc[T0_DATE] == pytest.approx(100.0, abs=1e-12)
    # Names and shape sanity
    assert n_gold.name == "N_Gold"
    # Following days move proportionally
    next_day = pd.Timestamp("2000-01-04")
    expected = (values[dates.searchsorted(next_day)] / values[dates.searchsorted(T0_DATE)]) * 100.0
    assert n_gold.loc[next_day] == pytest.approx(expected, rel=1e-9)


def test_build_n_gold_anchors_at_first_valid_when_t0_uncovered(gold_con) -> None:
    """If XAU starts AFTER T0 (real-world case for GC=F at 2000-08-30),
    N_Gold anchors at the first available date instead. Pre-anchor dates are NaN."""
    # Mirror the real-world gap: XAU starts 2000-08-30, T0 is 2000-01-03.
    dates = pd.bdate_range("2000-08-30", "2000-12-31")
    values = [280.0 + i * 0.1 for i in range(len(dates))]
    _seed_daily(gold_con, XAU_SERIES_ID, SOURCE_YAHOO, dates, values)

    n_gold = build_n_gold(gold_con, end=pd.Timestamp("2000-12-31"))

    # T0 has no XAU value → NaN
    assert pd.isna(n_gold.loc[T0_DATE])
    # First-valid date anchors at 100.0
    first_xau_date = pd.Timestamp("2000-08-30")
    assert n_gold.loc[first_xau_date] == pytest.approx(100.0, abs=1e-12)


def test_build_n_gold_raises_when_xau_not_ingested(gold_con) -> None:
    with pytest.raises(RuntimeError, match=XAU_SERIES_ID):
        build_n_gold(gold_con, end=pd.Timestamp("2000-12-31"))


def test_build_n_gold_does_not_use_kalman(gold_con) -> None:
    """Patch-06 invariant: build_n_gold reads ONLY the XAU series.
    Even if TIPS/DXY/VIX are absent, build_n_gold must still work."""
    dates = pd.bdate_range("1999-12-20", "2000-02-15")
    _seed_daily(gold_con, XAU_SERIES_ID, SOURCE_YAHOO, dates, [280.0] * len(dates))
    # Deliberately do NOT seed TIPS/DXY/VIX
    n_gold = build_n_gold(gold_con, end=pd.Timestamp("2000-02-15"))
    # If Kalman was being run, it would raise from assemble_panel because
    # TIPS/DXY/VIX aren't seeded. The fact that we get a usable Series
    # confirms build_n_gold uses raw XAU only.
    assert n_gold.notna().any()
    assert n_gold.loc[T0_DATE] == pytest.approx(100.0, abs=1e-12)


def test_build_n_gold_locf_uses_release_date(gold_con) -> None:
    """LOCF: a release dated 2000-01-03 must NOT be visible on 2000-01-02."""
    # Seed three releases: 2000-01-03, 2000-01-04, 2000-01-05.
    dates = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"])
    _seed_daily(gold_con, XAU_SERIES_ID, SOURCE_YAHOO, dates, [280.0, 285.0, 290.0])

    n_gold = build_n_gold(gold_con, end=pd.Timestamp("2000-01-15"))
    # 2000-01-03 is the first release → first non-NaN
    pre_t0 = pd.Timestamp("1999-12-31")
    if pre_t0 in n_gold.index:
        assert pd.isna(n_gold.loc[pre_t0])
    assert n_gold.loc[pd.Timestamp("2000-01-03")] == pytest.approx(100.0, abs=1e-12)
    expected_jan04 = 285.0 / 280.0 * 100.0
    assert n_gold.loc[pd.Timestamp("2000-01-04")] == pytest.approx(expected_jan04, rel=1e-9)


def test_kalman_remains_independent_of_build_n_gold(gold_con) -> None:
    """The diagnostic Kalman fit must still work even though it no longer
    drives N_Gold. This guards the Phase 4.5 demotion: fit_gold_model and
    its outputs are preserved exactly."""
    dates = pd.bdate_range("2003-01-02", "2003-09-30")
    n = len(dates)
    # Seed all four series so the Kalman has a clean panel.
    _seed_daily(gold_con, SERIES_IDS["XAU"],  SOURCE_YAHOO, dates, [350.0 + i for i in range(n)])
    _seed_daily(gold_con, SERIES_IDS["TIPS"], SOURCE_FRED,  dates, [2.0]   * n)
    _seed_daily(gold_con, SERIES_IDS["DXY"],  SOURCE_FRED,  dates, [100.0] * n)
    _seed_daily(gold_con, SERIES_IDS["VIX"],  SOURCE_FRED,  dates, [20.0]  * n)

    panel = assemble_panel(gold_con, end=pd.Timestamp("2003-09-30"))
    fit = fit_gold_model(panel)
    # Cleaned panel covers NYSE trading days in 2003-01-02..2003-09-30. The exact
    # row count differs from `n` (pandas bdate_range vs NYSE) but must be > 0.
    assert len(fit.filtered_state) > 100
    assert fit.results is not None
    # Filtered state index falls within the cleaned panel range.
    assert fit.filtered_state.index.min() >= pd.Timestamp("2003-01-02")
