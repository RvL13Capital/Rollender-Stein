from __future__ import annotations

import pandas as pd
import pytest

from rollender_stein.bitemporal import insert_macro_releases, open_db
from rollender_stein.calendar import T0_DATE
from rollender_stein.numeraires.energy import (
    BBL_TO_MWH_DIVISOR,
    BRENT_SERIES,
    MWH_PRICE_FLOOR_USD,
    SOURCE,
    build_n_energy,
)


@pytest.fixture
def con():
    with open_db(":memory:") as c:
        yield c


def _seed_brent(con, dates: list[str], values: list[float]) -> None:
    idx = pd.to_datetime(dates)
    insert_macro_releases(
        con,
        BRENT_SERIES,
        pd.DataFrame({"reference_date": idx, "release_date": idx, "value": values}),
        source=SOURCE,
    )


def test_n_energy_is_exactly_100_at_t0(con) -> None:
    """Brent at T0=2000-01-03 ≈ $25/bbl → MWh ≈ $14.7/MWh, well above the
    new $0.10 floor. N_Energy(T0) = 100 by construction (anchor / anchor)."""
    _seed_brent(
        con,
        dates=["1999-12-30", "2000-01-03", "2000-01-04"],
        values=[25.50, 25.55, 25.60],
    )
    n_energy = build_n_energy(con, end=pd.Timestamp("2000-01-15"))
    assert n_energy.loc[T0_DATE] == pytest.approx(100.0)
    assert n_energy.name == "N_Energy"


def test_n_energy_applies_mwh_floor(con) -> None:
    """When raw Brent/1.699 falls BELOW the $0.10 floor (synthetic — never
    observed historically), the floor binds and the daily value is clipped."""
    # Construct a date with raw value below the new $0.10 floor.
    # raw = brent/1.699 < 0.10  →  brent < 0.17. Use $0.05/bbl.
    _seed_brent(
        con,
        dates=["1999-12-30", "2000-01-03", "2030-01-02"],
        values=[40.0, 40.0, 0.05],
    )
    n = build_n_energy(con, end=pd.Timestamp("2030-01-15"))
    # Anchor: 40 / 1.699 = 23.54 (no floor binding at T0)
    # 2030-01-02 raw: 0.05 / 1.699 = 0.0294 → clipped to 0.10
    # N_Energy = 0.10 / 23.54 * 100 ≈ 0.4248
    expected = (MWH_PRICE_FLOOR_USD / (40.0 / BBL_TO_MWH_DIVISOR)) * 100.0
    assert n.loc[pd.Timestamp("2030-01-02")] == pytest.approx(expected, rel=1e-9)


def test_n_energy_does_not_clip_real_april_2020_brent(con) -> None:
    """The post-patch-03 floor of $0.10 is far below the April 2020 Brent
    crash value of ~$9/bbl. Verify that 2020-04-21 retains its raw value
    (no floor binding) under the new floor."""
    _seed_brent(
        con,
        dates=["1999-12-30", "2000-01-03", "2020-04-21"],
        values=[25.55, 25.55, 9.12],
    )
    n = build_n_energy(con, end=pd.Timestamp("2020-04-25"))
    # raw 9.12 / 1.699 = 5.37, above $0.10 floor → no clip
    # N_Energy = 5.37 / 15.04 * 100 ≈ 35.7
    expected = (9.12 / BBL_TO_MWH_DIVISOR) / (25.55 / BBL_TO_MWH_DIVISOR) * 100.0
    assert n.loc[pd.Timestamp("2020-04-21")] == pytest.approx(expected, rel=1e-9)


def test_n_energy_warns_when_floor_binds_at_t0(con) -> None:
    """If a contrived T0 Brent value is below the floor, a RuntimeWarning
    fires explaining the bias. This guards the audit M-2 condition."""
    # Synthetic: T0 raw value below $0.10 → floor binds at the anchor
    _seed_brent(
        con,
        dates=["1999-12-30", "2000-01-03"],
        values=[0.05, 0.05],  # both below $0.17/bbl threshold
    )
    with pytest.warns(RuntimeWarning, match="floor.*binds at T0"):
        build_n_energy(con, end=pd.Timestamp("2000-01-15"))


def test_n_energy_floor_does_not_warn_on_real_brent_levels(con) -> None:
    """The patch-03 floor must NOT warn on any historically observed Brent
    level. Test the lowest sensible scenario (Brent ~$10/bbl)."""
    _seed_brent(
        con,
        dates=["1999-12-30", "2000-01-03"],
        values=[10.0, 10.0],
    )
    # raw 10/1.699 = 5.88, far above $0.10 — no warning
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning becomes an error here
        build_n_energy(con, end=pd.Timestamp("2000-01-15"))


def test_n_energy_raises_when_no_data(con) -> None:
    with pytest.raises(RuntimeError, match="no rows"):
        build_n_energy(con)


def test_n_energy_raises_when_t0_unanchored(con) -> None:
    """If Brent's first release is after T0, anchor is undefined."""
    _seed_brent(con, dates=["2001-06-01"], values=[28.0])
    with pytest.raises(RuntimeError, match="cannot index"):
        build_n_energy(con, end=pd.Timestamp("2001-12-01"))


def test_brent_normalization_uses_correct_divisor(con) -> None:
    """Sanity: at 2x Brent and same T0, N_Energy = 200."""
    _seed_brent(
        con,
        dates=["1999-12-30", "2000-01-03", "2010-01-04"],
        values=[40.0, 40.0, 80.0],  # T0 = $40, 2010 = $80 (2x)
    )
    n = build_n_energy(con, end=pd.Timestamp("2010-01-10"))
    assert n.loc[pd.Timestamp("2010-01-04")] == pytest.approx(200.0)
