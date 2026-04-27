from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from rollender_stein.io.eia import PAGE_SIZE, fetch_eia_petroleum_spot


def _mock_resp(payload: dict) -> MagicMock:
    m = MagicMock()
    m.json.return_value = payload
    m.raise_for_status.return_value = None
    return m


def _mock_session(*payloads: dict) -> MagicMock:
    s = MagicMock()
    s.get.side_effect = [_mock_resp(p) for p in payloads]
    return s


def test_parses_single_page() -> None:
    payload = {
        "response": {
            "total": 3,
            "data": [
                {"period": "2024-01-02", "value": "75.20"},
                {"period": "2024-01-03", "value": "75.40"},
                {"period": "2024-01-04", "value": "."},  # missing — must drop
            ],
        }
    }
    df = fetch_eia_petroleum_spot("RBRTE", "key", session=_mock_session(payload))
    assert df["value"].tolist() == [75.20, 75.40]
    assert (df["reference_date"] == df["release_date"]).all()


def test_paginates_when_total_exceeds_page_size() -> None:
    """When response.total > one page, the loader must request additional offsets."""
    period_dates = pd.bdate_range("2010-01-04", periods=PAGE_SIZE).strftime("%Y-%m-%d")
    page1 = {
        "response": {
            "total": PAGE_SIZE + 2,
            "data": [{"period": d, "value": f"{75.0 + i}"} for i, d in enumerate(period_dates)],
        }
    }
    page2 = {
        "response": {
            "total": PAGE_SIZE + 2,
            "data": [
                {"period": "2030-01-01", "value": "76.0"},
                {"period": "2030-01-02", "value": "76.1"},
            ],
        }
    }
    sess = _mock_session(page1, page2)
    df = fetch_eia_petroleum_spot("RBRTE", "key", session=sess)
    assert len(df) == PAGE_SIZE + 2
    # Verify offsets sent on each call
    calls = sess.get.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["params"]["offset"] == 0
    assert calls[1].kwargs["params"]["offset"] == PAGE_SIZE


def test_empty_response_returns_typed_empty_frame() -> None:
    payload = {"response": {"total": 0, "data": []}}
    df = fetch_eia_petroleum_spot("RBRTE", "key", session=_mock_session(payload))
    assert df.empty
    assert df["reference_date"].dtype == "datetime64[ns]"


def test_query_params_use_v2_facet_syntax() -> None:
    payload = {"response": {"total": 0, "data": []}}
    sess = _mock_session(payload)
    fetch_eia_petroleum_spot("RBRTE", "secret_key", start="2000-01-01", session=sess)
    p = sess.get.call_args.kwargs["params"]
    assert p["api_key"] == "secret_key"
    assert p["frequency"] == "daily"
    assert p["facets[series][]"] == "RBRTE"
    assert p["start"] == "2000-01-01"
    assert p["sort[0][column]"] == "period"
    assert p["sort[0][direction]"] == "asc"
    assert p["length"] == PAGE_SIZE
