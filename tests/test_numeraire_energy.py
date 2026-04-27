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
    """Brent at T0=2000-01-03 was about $25.55/bbl. The MWh cost is then
    25.55/1.699 ≈ $15.04, but the floor of $20 kicks in. N_Energy(T0) = 100
    regardless of the absolute level."""
    _seed_brent(
        con,
        dates=["1999-12-30", "2000-01-03", "2000-01-04"],
        values=[25.50, 25.55, 25.60],
    )
    n_energy = build_n_energy(con, end=pd.Timestamp("2000-01-15"))
    assert n_energy.loc[T0_DATE] == pytest.approx(100.0)
    assert n_energy.name == "N_Energy"


def test_n_energy_applies_mwh_floor(con) -> None:
    """If raw Brent/1.699 is below $20/MWh, the value is clipped at $20."""
    # April 2020 collapse: Brent briefly traded near $9-12/bbl. After /1.699
    # that's ~$5-7/MWh — should clip to $20.
    _seed_brent(
        con,
        dates=["1999-12-30", "2000-01-03", "2020-04-21"],
        values=[40.0, 40.0, 9.12],   # T0 normal, COVID collapse
    )
    # T0 raw: 40/1.699 = 23.54 (above floor) → anchor = 23.54
    # Expected April 2020 value: 9.12/1.699 = 5.37 < 20 → clipped to 20.
    # N_Energy_April = (20 / 23.54) * 100 ≈ 84.95
    n = build_n_energy(con, end=pd.Timestamp("2020-04-25"))
    expected = (MWH_PRICE_FLOOR_USD / (40.0 / BBL_TO_MWH_DIVISOR)) * 100.0
    assert n.loc[pd.Timestamp("2020-04-21")] == pytest.approx(expected, rel=1e-9)


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
