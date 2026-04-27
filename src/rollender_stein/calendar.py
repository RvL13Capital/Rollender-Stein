"""Master NYSE business calendar anchored to T0.

Phase 1.1 of the AVE spec. T0 is the Genesis Timestamp:
the first NYSE close of the millennium. Every numéraire and
asset index is normalized to exactly 100.00 at this moment.
"""

from __future__ import annotations

import pandas as pd
import pandas_market_calendars as mcal

T0: pd.Timestamp = pd.Timestamp("2000-01-03 17:00:00", tz="UTC")
T0_DATE: pd.Timestamp = pd.Timestamp("2000-01-03")

_NYSE = mcal.get_calendar("NYSE")


def master_calendar(
    start: pd.Timestamp = T0_DATE,
    end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """All NYSE trading days in [start, end]. Defaults: T0 → today (UTC).

    The returned index is naive (no tz). Holiday-adjusted via
    pandas_market_calendars — weekends and NYSE holidays are excluded.
    """
    if end is None:
        end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    schedule = _NYSE.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index, name="trade_date")
