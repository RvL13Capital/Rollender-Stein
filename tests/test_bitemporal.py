from __future__ import annotations

import pandas as pd
import pytest

from rollender_stein.bitemporal import (
    insert_macro_releases,
    latest_release_stream,
    open_db,
)


@pytest.fixture
def con():
    with open_db(":memory:") as c:
        yield c


def _three_monthly_releases() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
            "release_date": pd.to_datetime(["2024-02-02", "2024-03-01", "2024-04-01"]),
            "value": [29.50, 29.65, 29.80],
        }
    )


def test_schema_creates_three_tables(con) -> None:
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchdf()
    assert set(tables["table_name"]) >= {"macro_release", "asset_price", "fx_close"}


def test_insert_then_stream_returns_one_row_per_release(con) -> None:
    rows = _three_monthly_releases()
    n = insert_macro_releases(con, "AHETPI", rows, source="FRED_ALFRED", vintage="2024-04-15")
    assert n == 3

    stream = latest_release_stream(con, "AHETPI")
    assert len(stream) == 3
    assert list(stream["value"]) == [29.50, 29.65, 29.80]
    assert list(stream["release_date"].astype("datetime64[ns]")) == list(
        pd.to_datetime(["2024-02-02", "2024-03-01", "2024-04-01"])
    )


def test_insert_or_replace_overwrites_same_pk(con) -> None:
    """If we re-ingest the same (series_id, reference_date, release_date), the value updates."""
    initial = _three_monthly_releases()
    insert_macro_releases(con, "AHETPI", initial, source="FRED_ALFRED")

    # Pretend a vintage correction: same release+reference, different value
    revised = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2024-01-31"]),
            "release_date": pd.to_datetime(["2024-02-02"]),
            "value": [29.51],
        }
    )
    insert_macro_releases(con, "AHETPI", revised, source="FRED_ALFRED", vintage="re-pull")

    stream = latest_release_stream(con, "AHETPI")
    feb_value = stream.loc[stream["release_date"] == pd.Timestamp("2024-02-02"), "value"].iloc[0]
    assert feb_value == 29.51


def test_same_release_date_two_reference_periods_picks_latest_period(con) -> None:
    """If on Mar 1 we publish Feb-2024 (new headline) AND a Jan-2024 revision, the
    stream should hold the Feb headline (29.65), not the Jan revision (29.55)."""
    rows = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "release_date": pd.to_datetime(["2024-03-01", "2024-03-01"]),
            "value": [29.55, 29.65],  # Jan revision, Feb headline
        }
    )
    insert_macro_releases(con, "AHETPI", rows, source="FRED_ALFRED")

    stream = latest_release_stream(con, "AHETPI")
    assert len(stream) == 1, "two rows on the same release_date must collapse to one"
    assert stream.iloc[0]["value"] == 29.65, "headline (latest reference_date) must win"


def test_insert_rejects_missing_columns(con) -> None:
    bad = pd.DataFrame({"release_date": pd.to_datetime(["2024-02-02"]), "value": [1.0]})
    with pytest.raises(KeyError, match="reference_date"):
        insert_macro_releases(con, "AHETPI", bad, source="FRED_ALFRED")


def test_stream_for_unknown_series_is_empty(con) -> None:
    stream = latest_release_stream(con, "DOES_NOT_EXIST")
    assert len(stream) == 0


# ----- migrate_publication_lags (audit patch 02) ------------------------------


def test_migrate_publication_lags_updates_listed_series(con) -> None:
    """Rows for series_ids listed in PUBLICATION_LAG_BD with non-zero lag
    must have their release_dates shifted forward by BDay(lag)."""
    from rollender_stein.bitemporal import migrate_publication_lags

    # Seed DFII10 (lag=1 BD) with release == reference
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    rows = pd.DataFrame(
        {"reference_date": dates, "release_date": dates, "value": [4.21, 4.18, 4.15]}
    )
    insert_macro_releases(con, "DFII10", rows, source="FRED")

    # Pre-migration: release == reference
    pre = con.execute(
        "SELECT release_date FROM macro_release WHERE series_id='DFII10' ORDER BY reference_date"
    ).fetchall()
    assert [r[0] for r in pre] == [d.date() for d in dates]

    n = migrate_publication_lags(con)
    assert n == 3

    # Post-migration: release == reference + 1 BD
    post = con.execute(
        "SELECT release_date FROM macro_release WHERE series_id='DFII10' ORDER BY reference_date"
    ).fetchall()
    expected = [(d + pd.tseries.offsets.BDay(1)).date() for d in dates]
    assert [r[0] for r in post] == expected


def test_migrate_publication_lags_is_idempotent(con) -> None:
    """Running the migration twice must not change anything after the first run."""
    from rollender_stein.bitemporal import migrate_publication_lags

    dates = pd.to_datetime(["2023-11-01", "2023-12-01"])
    rows = pd.DataFrame(
        {"reference_date": dates, "release_date": dates, "value": [1.0, 2.0]}
    )
    insert_macro_releases(con, "MABMM301EZM189S", rows, source="FRED")

    n1 = migrate_publication_lags(con)
    n2 = migrate_publication_lags(con)
    assert n1 == 2
    assert n2 == 0  # second run finds no `release == reference` rows for this series


def test_migrate_publication_lags_does_not_touch_unlisted_series(con) -> None:
    """Series NOT in PUBLICATION_LAG_BD must be untouched by the migration."""
    from rollender_stein.bitemporal import migrate_publication_lags

    dates = pd.to_datetime(["2020-06-15", "2020-07-15"])
    rows = pd.DataFrame(
        {"reference_date": dates, "release_date": dates, "value": [1.0, 2.0]}
    )
    insert_macro_releases(con, "SOME_RANDOM_SERIES", rows, source="UNKNOWN")

    migrate_publication_lags(con)

    post = con.execute(
        "SELECT release_date FROM macro_release WHERE series_id='SOME_RANDOM_SERIES' "
        "ORDER BY reference_date"
    ).fetchall()
    # Untouched: release_date still equals reference_date
    assert [r[0] for r in post] == [d.date() for d in dates]


def test_release_after_reference_invariant_blocks_bad_insert(con) -> None:
    """Audit patch 01: insert_macro_releases must reject release_date <
    reference_date at the application boundary.

    The inline CHECK in _SCHEMA_DDL enforces this at the DB level for fresh
    DBs; the ValueError raised by _validate_release_after_reference enforces
    it for pre-existing DBs (DuckDB does not yet support ALTER TABLE ADD
    CONSTRAINT CHECK)."""
    bad = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2024-01-15"]),
            "release_date":   pd.to_datetime(["2024-01-01"]),  # BEFORE reference!
            "value": [42.0],
        }
    )
    with pytest.raises(ValueError, match=r"release_date.*reference_date"):
        insert_macro_releases(con, "BAD_SERIES", bad, source="UNITTEST")


def test_release_equal_to_reference_is_allowed(con) -> None:
    """For daily series with no publication lag, release == reference is valid."""
    good = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2024-01-15"]),
            "release_date":   pd.to_datetime(["2024-01-15"]),
            "value": [42.0],
        }
    )
    insert_macro_releases(con, "EQUAL_SERIES", good, source="UNITTEST")
    n = con.execute(
        "SELECT COUNT(*) FROM macro_release WHERE series_id='EQUAL_SERIES'"
    ).fetchone()[0]
    assert n == 1


def test_release_after_reference_is_allowed(con) -> None:
    """Standard ALFRED-style data: release > reference by some lag → allowed."""
    rows = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2024-01-31"]),
            "release_date":   pd.to_datetime(["2024-02-15"]),  # 15 days later — fine
            "value": [42.0],
        }
    )
    insert_macro_releases(con, "GOOD_SERIES", rows, source="UNITTEST")
    n = con.execute(
        "SELECT COUNT(*) FROM macro_release WHERE series_id='GOOD_SERIES'"
    ).fetchone()[0]
    assert n == 1


def test_release_after_reference_validation_message_lists_offending_rows(con) -> None:
    """The error message includes the first offending rows so the user can
    debug their input."""
    bad = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2024-01-15", "2024-02-15"]),
            "release_date":   pd.to_datetime(["2024-01-01", "2024-02-15"]),
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="2024-01-15"):
        insert_macro_releases(con, "BAD_SERIES", bad, source="UNITTEST")


def test_migrate_publication_lags_warns_on_pk_collision_and_skips(con) -> None:
    """Pathological case: an un-migrated row's BDay-shifted release_date
    happens to coincide with an existing revision row's PK. Migration must
    detect the collision, warn, and leave the un-migrated row in place
    rather than rolling back or overwriting the revision."""
    from rollender_stein.bitemporal import migrate_publication_lags

    # DFII10 has lag = 1 BD. Insert two rows for ref=2024-01-15:
    #   - (release=2024-01-15) → un-migrated; would shift to 2024-01-16
    #   - (release=2024-01-16) → existing revision row, would collide
    rows = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2024-01-15", "2024-01-15"]),
            "release_date":   pd.to_datetime(["2024-01-15", "2024-01-16"]),
            "value":          [100.0, 105.0],  # different values
        }
    )
    insert_macro_releases(con, "DFII10", rows, source="FRED")

    with pytest.warns(RuntimeWarning, match=r"PK collision"):
        n = migrate_publication_lags(con)

    # The colliding row was excluded; nothing else to migrate. n == 0.
    assert n == 0
    # Both rows preserved exactly:
    post = con.execute(
        "SELECT release_date, value FROM macro_release WHERE series_id='DFII10' "
        "ORDER BY release_date"
    ).fetchdf()
    assert len(post) == 2
    assert post["value"].tolist() == [100.0, 105.0]


def test_migrate_publication_lags_collision_does_not_delete_other_rows(con) -> None:
    """Mixed batch: one row collides, another doesn't. The non-colliding
    row migrates correctly; the collision row stays un-migrated. Critically,
    the collision row must NOT be deleted by the broad DELETE clause."""
    from rollender_stein.bitemporal import migrate_publication_lags

    # DFII10: ref=2024-01-15 collides (existing row at release=2024-01-16);
    # ref=2024-02-15 does not collide.
    rows = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                ["2024-01-15", "2024-01-15", "2024-02-15"]
            ),
            "release_date":   pd.to_datetime(
                ["2024-01-15", "2024-01-16", "2024-02-15"]
            ),
            "value": [100.0, 105.0, 200.0],
        }
    )
    insert_macro_releases(con, "DFII10", rows, source="FRED")

    with pytest.warns(RuntimeWarning, match=r"PK collision"):
        migrate_publication_lags(con)

    post = con.execute(
        "SELECT reference_date, release_date, value FROM macro_release "
        "WHERE series_id='DFII10' ORDER BY reference_date, release_date"
    ).fetchdf()
    # All three rows preserved (collision row not deleted by the tightened DELETE):
    # - Jan 15 / Jan 15: un-migrated (collision excluded)
    # - Jan 15 / Jan 16: existing revision
    # - Feb 15 / Feb 16: migrated successfully (Feb 15 was Thursday, +1 BD = Fri Feb 16)
    assert len(post) == 3
    # Specifically: the un-migrated Jan 15 row at release=Jan 15 still exists
    jan15_at_jan15 = post[
        (pd.to_datetime(post["reference_date"]) == pd.Timestamp("2024-01-15"))
        & (pd.to_datetime(post["release_date"]) == pd.Timestamp("2024-01-15"))
    ]
    assert len(jan15_at_jan15) == 1, "collision row was incorrectly deleted by migration"
    # The Feb 15 row was migrated to a non-equal release_date.
    feb15_rows = post[pd.to_datetime(post["reference_date"]) == pd.Timestamp("2024-02-15")]
    assert len(feb15_rows) == 1
    assert pd.to_datetime(feb15_rows["release_date"].iloc[0]) > pd.Timestamp("2024-02-15")


def test_migrate_publication_lags_preserves_alfred_anchored_rows(con) -> None:
    """For series like WM2NS with mixed data (some rows from ALFRED with
    legitimate release_date != reference_date, some from live-endpoint with
    release == reference), migration must touch ONLY the live-endpoint rows.

    This was the audit's biggest landmine: blanket update would corrupt the
    ALFRED-derived rows. The WHERE release_date == reference_date guard
    prevents this."""
    from rollender_stein.bitemporal import migrate_publication_lags

    # Mix: one ALFRED-style row (release > reference by 7 days) + one live row
    alfred_ref = pd.Timestamp("2010-01-04")
    alfred_rel = pd.Timestamp("2010-01-11")
    live_dates = pd.to_datetime(["2020-01-06", "2020-01-13"])
    rows = pd.DataFrame(
        {
            "reference_date": [alfred_ref, *live_dates],
            "release_date":   [alfred_rel, *live_dates],
            "value":          [10.0, 20.0, 30.0],
        }
    )
    insert_macro_releases(con, "WM2NS", rows, source="FRED")

    migrate_publication_lags(con)

    post = con.execute(
        "SELECT reference_date, release_date FROM macro_release "
        "WHERE series_id='WM2NS' ORDER BY reference_date"
    ).fetchall()
    # ALFRED row preserved exactly
    assert pd.Timestamp(post[0][0]) == alfred_ref
    assert pd.Timestamp(post[0][1]) == alfred_rel
    # Live rows shifted by 5 BD (WM2NS lag)
    for i, ref in enumerate(live_dates):
        expected_release = ref + pd.tseries.offsets.BDay(5)
        assert pd.Timestamp(post[i + 1][1]) == expected_release
