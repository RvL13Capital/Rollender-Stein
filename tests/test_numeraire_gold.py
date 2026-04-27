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
    assemble_panel,
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
