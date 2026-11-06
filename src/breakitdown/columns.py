"""Column conventions: which columns hold the unit id, the flight, the event time and the exposure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Columns:
    """Names of the columns breakitdown reads from metric tables and from a flight mapping.

    Set it once on the ``Experiment``; a ``MetricTable`` or ``FlightMapping`` with a different schema
    can override it with its own ``columns=``.

    Attributes
    ----------
    unit:
        Randomization unit id (ruid). Units are the independent observations of every test.
    flight:
        The flight (variant) the unit is assigned to. Values are compared as strings.
    time:
        Event time of a row: DATE, TIMESTAMP, string (parsed with ``time_format``, default ISO
        ``yyyy-MM-dd``) or ``yyyyMMdd`` integers. Rows are bucketed by the calendar day of this
        column and filtered to the experiment's date range; tables are never date-suffixed.
    exposure:
        Optional time of a unit's first exposure, with the same accepted types as ``time``. When
        set, rows before a unit's exposure are dropped and daily denominators count the unit from
        its exposure day on. If both ``time`` and ``exposure`` are timestamps the cut-off is exact,
        otherwise it is by day.
    time_format:
        Spark datetime pattern for string or integer ``time``/``exposure`` columns, e.g. ``"yyyyMMdd"``.
    """

    unit: str = "ruid"
    flight: str = "flight"
    time: str = "event_time"
    exposure: Optional[str] = None
    time_format: Optional[str] = None
