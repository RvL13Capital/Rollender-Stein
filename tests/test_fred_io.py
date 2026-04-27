from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from rollender_stein.io.fred import fetch_alfred_first_release, fetch_fred_observations


def _mock_session(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None

    sess = MagicMock()
    sess.get.return_value = resp
    return sess


def test_parses_three_observations() -> None:
    sess = _mock_session(
        {
            "observations": [
                {
                    "date": "2024-01-31",
                    "realtime_start": "2024-02-02",
                    "realtime_end": "2024-03-01",
                    "value": "29.50",
                },
                {
                    "date": "2024-02-29",
                    "realtime_start": "2024-03-01",
                    "realtime_end": "2024-04-01",
                    "value": "29.65",
                },
                {
                    "date": "2024-03-31",
                    "realtime_start": "2024-04-01",
                    "realtime_end": "9999-12-31",
                    "value": "29.80",
                },
            ],
        }
    )
    df = fetch_alfred_first_release("AHETPI", "key", session=sess)

    assert list(df.columns) == ["reference_date", "release_date", "value"]
    assert df["value"].tolist() == [29.50, 29.65, 29.80]
    assert df["reference_date"].iloc[0] == pd.Timestamp("2024-01-31")
    assert df["release_date"].iloc[0] == pd.Timestamp("2024-02-02")


def test_drops_missing_values() -> None:
    """FRED encodes missing values as `'.'`; rows with `.` must be dropped."""
    sess = _mock_session(
        {
            "observations": [
                {
                    "date": "2024-01-31",
                    "realtime_start": "2024-02-02",
                    "realtime_end": "2024-03-01",
                    "value": ".",
                },
                {
                    "date": "2024-02-29",
                    "realtime_start": "2024-03-01",
                    "realtime_end": "2024-04-01",
                    "value": "29.65",
                },
            ],
        }
    )
    df = fetch_alfred_first_release("AHETPI", "key", session=sess)
    assert len(df) == 1
    assert df["value"].iloc[0] == 29.65


def test_empty_observations_returns_typed_empty_frame() -> None:
    sess = _mock_session({"observations": []})
    df = fetch_alfred_first_release("AHETPI", "key", session=sess)
    assert df.empty
    assert df.columns.tolist() == ["reference_date", "release_date", "value"]
    assert df["reference_date"].dtype == "datetime64[ns]"
    assert df["release_date"].dtype == "datetime64[ns]"


def test_default_realtime_end_is_max_date_sentinel() -> None:
    """Regression: FRED rejects realtime_end > FRED's "today" with HTTP 400 when
    the local clock runs ahead of FRED's. Default must be ``"9999-12-31"`` so we
    never depend on local-vs-FRED clock alignment."""
    sess = _mock_session({"observations": []})
    fetch_alfred_first_release("AHETPI", "key", session=sess)
    params = sess.get.call_args.kwargs["params"]
    assert params["realtime_end"] == "9999-12-31"


def test_passes_expected_query_params() -> None:
    sess = _mock_session({"observations": []})
    fetch_alfred_first_release(
        "AHETPI",
        "secret_key",
        realtime_start="1990-01-01",
        realtime_end="2026-04-27",
        session=sess,
    )
    # Inspect the params passed to .get()
    call = sess.get.call_args
    params = call.kwargs["params"]
    assert params["series_id"] == "AHETPI"
    assert params["api_key"] == "secret_key"
    assert params["file_type"] == "json"
    assert params["output_type"] == 4
    assert params["realtime_start"] == "1990-01-01"
    assert params["realtime_end"] == "2026-04-27"


def test_fetch_fred_observations_sets_release_to_reference_for_unknown_series() -> None:
    """For series NOT in the publication-lag table, release_date == reference_date.

    This is the post-patch-02 default — only series with explicit non-zero
    lag in PUBLICATION_LAG_BD get the BDay offset.
    """
    sess = _mock_session(
        {
            "observations": [
                {"date": "2024-01-02", "value": "4.21"},
                {"date": "2024-01-03", "value": "4.18"},
                {"date": "2024-01-04", "value": "."},  # missing — must be dropped
                {"date": "2024-01-05", "value": "4.15"},
            ],
        }
    )
    df = fetch_fred_observations("UNKNOWN_SERIES_NO_LAG", "key", session=sess)

    assert len(df) == 3
    assert df["value"].tolist() == [4.21, 4.18, 4.15]
    assert (df["reference_date"] == df["release_date"]).all()


def test_fetch_fred_observations_applies_publication_lag() -> None:
    """For series listed in PUBLICATION_LAG_BD with non-zero lag, the
    release_date is offset from reference_date by the configured BDays."""
    from rollender_stein.io.fred import PUBLICATION_LAG_BD

    sess = _mock_session(
        {
            "observations": [
                {"date": "2024-01-02", "value": "4.21"},  # Tue
                {"date": "2024-01-03", "value": "4.18"},  # Wed
            ],
        }
    )
    # DFII10 has lag = 1 BD per the table.
    assert PUBLICATION_LAG_BD["DFII10"] == 1
    df = fetch_fred_observations("DFII10", "key", session=sess)
    assert len(df) == 2
    # 2024-01-02 (Tue) + 1 BD = 2024-01-03 (Wed)
    # 2024-01-03 (Wed) + 1 BD = 2024-01-04 (Thu)
    assert df.loc[0, "release_date"] == pd.Timestamp("2024-01-03")
    assert df.loc[1, "release_date"] == pd.Timestamp("2024-01-04")


def test_fetch_fred_observations_applies_30bd_lag_for_monthly_aggregates() -> None:
    """The biggest-deal patch-02 case: monthly EZ/JP M3 aggregates get a
    +30 BD offset, closing the audit C-3 look-ahead injection."""
    sess = _mock_session(
        {
            "observations": [
                {"date": "2023-11-01", "value": "16025573773633.299"},
            ],
        }
    )
    df = fetch_fred_observations("MABMM301EZM189S", "key", session=sess)
    assert len(df) == 1
    # 30 BD after 2023-11-01 (Wed) is well into December
    expected = pd.Timestamp("2023-11-01") + pd.tseries.offsets.BDay(30)
    assert df.loc[0, "release_date"] == expected


def test_fetch_fred_observations_no_realtime_params() -> None:
    """Live endpoint must NOT pass realtime_start/end (avoids vintage limit)."""
    sess = _mock_session({"observations": []})
    fetch_fred_observations("VIXCLS", "key", session=sess)
    params = sess.get.call_args.kwargs["params"]
    assert "realtime_start" not in params
    assert "realtime_end" not in params
    assert "output_type" not in params  # default = 1 (current)
    assert params["observation_start"] == "1990-01-01"
    assert params["observation_end"] == "9999-12-31"
