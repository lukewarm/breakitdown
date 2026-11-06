"""Shared constants and helpers: dates, segment specs, column quoting and analysis periods."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

OVERALL = "overall"
"""Segment name and value of results that are not broken down."""

NULL_LABEL = "(null)"
"""How a NULL segment value is shown. NULLs form their own segment instead of being dropped."""

SEPARATOR = "|"
"""Joins the columns of a multi-column segment: segment ``os|country`` has values like ``iOS|US``."""

DateLike = Union[str, _dt.date, _dt.datetime]
SegmentSpec = Tuple[str, ...]
SegmentsLike = Union[None, str, Sequence[Union[str, Sequence[str]]]]


def to_date(value: DateLike) -> _dt.date:
    """Parse ``YYYY-MM-DD`` / ``YYYYMMDD`` strings, dates and datetimes into a ``date``."""
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
            try:
                return _dt.datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Unrecognised date {value!r}; use YYYY-MM-DD or YYYYMMDD.")
    raise TypeError(f"Expected a date, datetime or string, got {type(value).__name__}.")


def normalize_segments(spec: SegmentsLike, what: str = "segment") -> Tuple[SegmentSpec, ...]:
    """Normalise a segment spec into a tuple of column tuples.

    ``"os"`` -> ``(("os",),)``, ``"os|country"`` -> ``(("os", "country"),)`` and
    ``["os", ("os", "country")]`` -> ``(("os",), ("os", "country"))``. Duplicates are dropped and
    the order is kept. The overall (not broken down) result is always produced and is not part of
    the spec.
    """
    if spec is None:
        return ()
    items = [spec] if isinstance(spec, str) else list(spec)
    out: List[SegmentSpec] = []
    for item in items:
        if isinstance(item, str):
            cols = tuple(part.strip() for part in item.split(SEPARATOR))
        else:
            cols = tuple(item)
        if not cols or not all(isinstance(c, str) and c for c in cols):
            raise ValueError(f"Invalid {what} {item!r}: expected a column name or a sequence of column names.")
        if len(set(cols)) != len(cols):
            raise ValueError(f"Invalid {what} {item!r}: a column appears twice.")
        if cols not in out:
            out.append(cols)
    return tuple(out)


def segment_columns(specs: Iterable[SegmentSpec]) -> Tuple[str, ...]:
    """The distinct columns used by a set of segment specs, in first-seen order."""
    return tuple(dict.fromkeys(col for spec in specs for col in spec))


def segment_name(spec: SegmentSpec) -> str:
    return SEPARATOR.join(spec) if spec else OVERALL


def quote(name: str) -> str:
    """Backtick-quote a column name so Spark takes dots and spaces in it literally."""
    return "`" + name.replace("`", "``") + "`"


@dataclass(frozen=True)
class Period:
    """A date range that is analysed as one block.

    ``name`` labels the period in the results. Every analysis has a single ``"ab"`` period today.
    Retro A/A (roadmap) adds an ``"aa"`` period over the pre-experiment range and runs it through
    the same pipeline, so results already carry a ``period`` column.
    """

    start: _dt.date
    end: _dt.date
    name: str = "ab"

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Period start {self.start} is after its end {self.end}.")

    @property
    def days(self) -> List[_dt.date]:
        return [self.start + _dt.timedelta(days=i) for i in range((self.end - self.start).days + 1)]
