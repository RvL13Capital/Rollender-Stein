from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from rollender_stein.io.fred import fetch_alfred_first_release


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
