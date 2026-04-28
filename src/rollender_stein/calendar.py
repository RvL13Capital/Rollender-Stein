"""Master NYSE business calendar anchored to T0.

Phase 1.1 of the AVE spec. T0 is the Genesis Timestamp: the first NYSE
trading day of the millennium (Monday, January 3, 2000). Every numéraire
is normalized to exactly 100.00 on this date.

Note on the literal ``T0`` constant below: the value ``2000-01-03 17:00:00 UTC``
corresponds to noon US Eastern (12:00 EST), not the actual NYSE close
which is 16:00 EST = 21:00 UTC. The audit (m-1) flagged this. Downstream
code uses only ``T0_DATE`` (the date), so the time-of-day mismatch is
harmless. The naïve date is the load-bearing constant; the timestamped
``T0`` is left intact to avoid touching the public API.
"""

from __future__ import annotations

import pandas as pd
import pandas_market_calendars as mcal

# T0_DATE (date-only) is what every downstream caller actually uses.
T0_DATE: pd.Timestamp = pd.Timestamp("2000-01-03")
# T0 (timestamped) is preserved for API compatibility but is only the date
# part that matters; the 17:00 UTC offset is noon ET, not NYSE close —
# documented above. Do not rely on the time component.
T0: pd.Timestamp = pd.Timestamp("2000-01-03 17:00:00", tz="UTC")

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
        # `Timestamp.utcnow()` is deprecated in pandas 2.x and removed in 3.x.
        # `Timestamp.now(tz="UTC")` is the future-proof replacement; we strip
        # the tz back off because downstream callers (DuckDB, master_calendar
        # consumers) all use naive timestamps.
        end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    schedule = _NYSE.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index, name="trade_date")
