from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rollender_stein.bitemporal import (
    insert_asset_prices,
    insert_macro_releases,
    open_db,
)
from rollender_stein.numeraires.gold import SERIES_IDS as GOLD_SIDS
from rollender_stein.numeraires.gold import SOURCE_FRED, SOURCE_YAHOO
from rollender_stein.numeraires.liquidity import ALL_SERIES as LIQ_ALL_SERIES
from rollender_stein.numeraires.time import SERIES_ID as TIME_SID
from rollender_stein.persist import (
    dump_all_artifacts,
    dump_division_array,
    dump_numeraires,
    dump_phase4_panel,
)


@pytest.fixture
def seeded_con():
    """In-memory DB seeded with enough realistic data to build all numéraires."""
    with open_db(":memory:") as con:
        # Enough range to cover T0 + early Kalman fit window.
        full = pd.bdate_range("1995-01-02", "2010-12-31")
        n = len(full)

        # AHETPI: monthly. Generate one row per month.
        monthly = pd.date_range("1995-01-01", "2010-12-01", freq="MS")
        m = len(monthly)
        insert_macro_releases(
            con,
            TIME_SID,
            pd.DataFrame(
                {
                    "reference_date": monthly,
                    "release_date": monthly + pd.Timedelta(days=35),
                    "value": np.linspace(11.0, 19.0, m),
                }
            ),
            source="FRED_ALFRED",
        )

        # Liquidity inputs: weekly US M2, monthly EZ/JP M3 (level + growth), daily FX.
        weekly = pd.date_range("1995-01-04", "2010-12-31", freq="W-WED")
        insert_macro_releases(
            con,
            LIQ_ALL_SERIES["US_M2"],
            pd.DataFrame(
                {
                    "reference_date": weekly,
                    "release_date": weekly + pd.Timedelta(days=10),
                    "value": np.linspace(3000.0, 9000.0, len(weekly)),
                }
            ),
            source="FRED",
        )
        for sid_key, lo, hi in [
            ("EZ_M3_LEVEL", 4_500_000_000_000.0, 9_000_000_000_000.0),
            ("JP_M3_LEVEL", 600_000_000_000_000.0, 1_200_000_000_000_000.0),
        ]:
            insert_macro_releases(
                con,
                LIQ_ALL_SERIES[sid_key],
                pd.DataFrame(
                    {
                        "reference_date": monthly,
                        "release_date": monthly + pd.Timedelta(days=35),
                        "value": np.linspace(lo, hi, m),
                    }
                ),
                source="FRED",
            )
        for sid_key, val in [("EZ_M3_GROWTH", 0.3), ("JP_M3_GROWTH", 0.1)]:
            insert_macro_releases(
                con,
                LIQ_ALL_SERIES[sid_key],
                pd.DataFrame(
                    {
                        "reference_date": monthly,
                        "release_date": monthly,
                        "value": [val] * m,
                    }
                ),
                source="FRED",
            )
        for sid_key, val in [("EURUSD", 1.10), ("USDJPY", 110.0)]:
            insert_macro_releases(
                con,
                LIQ_ALL_SERIES[sid_key],
                pd.DataFrame(
                    {"reference_date": full, "release_date": full, "value": [val] * n}
                ),
                source="FRED",
            )

        # Energy: daily Brent.
        insert_macro_releases(
            con,
            "RBRTE",
            pd.DataFrame(
                {
                    "reference_date": full,
                    "release_date": full,
                    "value": np.linspace(20.0, 70.0, n),
                }
            ),
            source="EIA",
        )

        # Phase 4 inputs: XAU + TIPS + DXY + VIX (all daily, all from 1995).
        rng = np.random.default_rng(42)
        xau_vals = 280.0 + np.cumsum(rng.normal(0, 1.0, n))
        # TIPS only from 2003 onward (mirror real DFII10 history).
        tips_dates = full[full >= "2003-01-02"]
        tips_vals = 2.0 + 0.5 * np.sin(np.arange(len(tips_dates)) / 50.0)
        dxy_vals = 100.0 + 5.0 * np.sin(np.arange(n) / 100.0)
        vix_vals = 18.0 + 8.0 * np.abs(np.sin(np.arange(n) / 30.0))
        insert_macro_releases(
            con,
            GOLD_SIDS["XAU"],
            pd.DataFrame({"reference_date": full, "release_date": full, "value": xau_vals}),
            source=SOURCE_YAHOO,
        )
        insert_macro_releases(
            con,
            GOLD_SIDS["TIPS"],
            pd.DataFrame(
                {"reference_date": tips_dates, "release_date": tips_dates, "value": tips_vals}
            ),
            source=SOURCE_FRED,
        )
        insert_macro_releases(
            con,
            GOLD_SIDS["DXY"],
            pd.DataFrame({"reference_date": full, "release_date": full, "value": dxy_vals}),
            source=SOURCE_FRED,
        )
        insert_macro_releases(
            con,
            GOLD_SIDS["VIX"],
            pd.DataFrame({"reference_date": full, "release_date": full, "value": vix_vals}),
            source=SOURCE_FRED,
        )

        # Asset price for the division test.
        asset_dates = pd.bdate_range("2000-01-03", "2010-12-31")
        insert_asset_prices(
            con,
            "TEST_ASSET",
            pd.DataFrame(
                {
                    "trade_date": asset_dates,
                    "close": np.linspace(100.0, 1000.0, len(asset_dates)),
                }
            ),
            source="UNITTEST",
        )
        yield con


def test_dump_numeraires_writes_four_parquets(seeded_con, tmp_path) -> None:
    info = dump_numeraires(seeded_con, end=pd.Timestamp("2010-12-31"), root=tmp_path)
    assert set(info.keys()) == {"n_time", "n_liquidity", "n_energy", "n_gold"}
    for slug, art in info.items():
        p = Path(art.path)
        assert p.exists() and p.suffix == ".parquet", f"{slug} not written"
        df = pd.read_parquet(p)
        assert df.index.name == "trade_date"
        assert len(df.columns) == 1


def test_dump_phase4_panel_has_four_columns(seeded_con, tmp_path) -> None:
    info = dump_phase4_panel(seeded_con, end=pd.Timestamp("2010-12-31"), root=tmp_path)
    df = pd.read_parquet(info.path)
    assert set(df.columns) == {"XAU", "TIPS", "DXY", "VIX"}
    assert info.rows == len(df)


def test_dump_division_array_round_trips(seeded_con, tmp_path) -> None:
    info = dump_division_array(
        seeded_con, "TEST_ASSET", end=pd.Timestamp("2010-12-31"), root=tmp_path
    )
    df = pd.read_parquet(info.path)
    assert "nominal_usd" in df.columns
    assert "asset_in_time" in df.columns
    assert "asset_in_liquidity" in df.columns
    assert "asset_in_energy" in df.columns
    assert "asset_in_gold" in df.columns


def test_dump_all_artifacts_writes_manifest(seeded_con, tmp_path) -> None:
    manifest = dump_all_artifacts(
        seeded_con,
        tickers=["TEST_ASSET"],
        end=pd.Timestamp("2010-12-31"),
        root=tmp_path,
    )
    # manifest written
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk["t0_date"] == "2000-01-03"
    assert "n_time" in on_disk["numeraires"]
    assert "TEST_ASSET" in on_disk["divisions"]
    # in-memory return matches what's on disk
    assert manifest["t0_date"] == on_disk["t0_date"]
    # kalman params written as JSON
    params_path = Path(on_disk["kalman"]["params_path"])
    assert params_path.exists()
    params = json.loads(params_path.read_text())
    assert "log_likelihood" in params
    assert "params" in params
    assert "beta.TIPS" in params["params"]


def test_dump_division_array_raises_for_missing_asset(seeded_con, tmp_path) -> None:
    with pytest.raises(RuntimeError, match="no rows in asset_price"):
        dump_division_array(
            seeded_con, "DOES_NOT_EXIST",
            end=pd.Timestamp("2010-12-31"), root=tmp_path,
        )
