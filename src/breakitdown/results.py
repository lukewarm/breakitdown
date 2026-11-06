"""ExperimentResults: the tidy comparison table, convenience views and an HTML rendering."""

from __future__ import annotations

import html
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import stats
from ._common import OVERALL, SEPARATOR
from .aggregate import CUMULATIVE, DAILY

SegmentArg = Union[None, str, Sequence[str]]

_ROW_COLORS = {"better": "#d7f0dd", "worse": "#f8d7da", "changed": "#fff3cd"}


class ExperimentResults:
    """What ``Experiment.analyze`` returns.

    ``table`` has one row per comparison: metric (x quantile) x treatment x granularity
    (``"cumulative"`` or ``"daily"``) x day x user segment value x metric segment value. The README
    lists every column. The methods below are filters over that table.

    Attributes
    ----------
    population:
        Units per metric table, flight and user segment (cumulative, and exposed-by-day when daily
        results were computed). These are the denominators of all metrics.
    sufficient_stats:
        Per metric table, the raw per-flight sufficient statistics collected from Spark, with
        columns named ``<metric>__<statistic>``. Useful for debugging and for re-analysis.
    settings:
        The analysis options used.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        population: pd.DataFrame,
        sufficient_stats: Dict[str, pd.DataFrame],
        flights: Sequence[str],
        expected_traffic: Optional[Mapping[str, float]],
        settings: Mapping[str, Any],
    ):
        self._table = table
        self.population = population
        self.sufficient_stats = sufficient_stats
        self.flights = tuple(flights)
        self.expected_traffic = dict(expected_traffic) if expected_traffic else None
        self.settings = dict(settings)

    # ------------------------------------------------------------------------------------- views

    @property
    def table(self) -> pd.DataFrame:
        """All comparisons (a copy)."""
        return self._table.copy()

    def overall(self, granularity: str = CUMULATIVE) -> pd.DataFrame:
        """Comparisons without any breakdown."""
        t = self._table
        rows = (t["granularity"] == granularity) & (t["user_segment"] == OVERALL) & (t["metric_segment"] == OVERALL)
        return t[rows].reset_index(drop=True)

    def by_user_segment(
        self, segment: SegmentArg = None, *, granularity: str = CUMULATIVE, metric_segment: str = OVERALL
    ) -> pd.DataFrame:
        """User-segment breakdowns, e.g. ``by_user_segment("os")`` or ``by_user_segment(("os", "country"))``.

        With a specific segment, its columns are added back (``os``, ``country``) next to
        ``user_segment_value``.
        """
        return self._segments("user_segment", segment, granularity, "metric_segment", metric_segment)

    def by_metric_segment(
        self, segment: SegmentArg = None, *, granularity: str = CUMULATIVE, user_segment: str = OVERALL
    ) -> pd.DataFrame:
        """Metric-segment breakdowns, with ``segment_share_control`` and ``segment_contribution``."""
        return self._segments("metric_segment", segment, granularity, "user_segment", user_segment)

    def daily(
        self,
        metric: Optional[str] = None,
        *,
        user_segment: str = OVERALL,
        user_segment_value: Optional[str] = None,
        metric_segment: str = OVERALL,
        metric_segment_value: Optional[str] = None,
    ) -> pd.DataFrame:
        """Daily results (each day on its own, with units exposed by that day as the denominator)."""
        t = self._table
        rows = (t["granularity"] == DAILY) & (t["user_segment"] == user_segment) & (t["metric_segment"] == metric_segment)
        if user_segment_value is not None:
            rows &= t["user_segment_value"] == user_segment_value
        if metric_segment_value is not None:
            rows &= t["metric_segment_value"] == metric_segment_value
        if metric is not None:
            rows &= t["metric"] == metric
        return t[rows].reset_index(drop=True)

    def hte(self, granularity: str = CUMULATIVE) -> pd.DataFrame:
        """One row per heterogeneity test (metric x treatment x user segment), most significant first."""
        keys = ["metric_table", "metric", "label", "quantile", "treatment", "user_segment", "metric_segment",
                "metric_segment_value"]
        t = self._table
        tested = t[(t["granularity"] == granularity) & t["hte_p_value"].notna()]
        if tested.empty:
            return pd.DataFrame(columns=keys + ["segments_tested", "hte_p_value", "hte_detected", "hte_pattern"])
        summary = tested.groupby(keys, dropna=False, sort=False).agg(
            segments_tested=("hte_tested", "sum"),
            hte_p_value=("hte_p_value", "first"),
            hte_detected=("hte_detected", "first"),
            hte_pattern=("hte_pattern", "first"),
        )
        return summary.reset_index().sort_values("hte_p_value", kind="mergesort").reset_index(drop=True)

    def srm(self, granularity: str = CUMULATIVE) -> pd.DataFrame:
        """Sample-ratio-mismatch test for each control/treatment pair.

        One row per metric table, user segment value and treatment (and day, for daily). Each pair is
        tested on its own units against the expected split: the same test as the ``srm_p_value``
        column. Needs ``expected_traffic`` on the Experiment. In pre-joined mode each table is its
        own population, so each table gets its own tests.
        """
        if not self.expected_traffic:
            raise ValueError("Pass expected_traffic={flight: share} to Experiment to test for sample ratio mismatch.")
        population = self.population[self.population["granularity"] == granularity]
        keys = ["metric_table", "user_segment", "user_segment_value"] + (["day"] if granularity == DAILY else [])
        columns = keys + ["control", "treatment", "n_control", "n_treatment", "expected_treatment_share",
                          "observed_treatment_share", "p_value"]
        control, treatments = self.flights[0], self.flights[1:]

        def counts(flight: str, name: str) -> pd.DataFrame:
            rows = population[population["flight"] == flight]
            return rows.groupby(keys, as_index=False)["n"].sum().rename(columns={"n": name})

        frames = []
        for order, treatment in enumerate(treatments):
            pair = counts(control, "n_control").merge(counts(treatment, "n_treatment"), on=keys, how="outer")
            pair = pair.fillna({"n_control": 0.0, "n_treatment": 0.0})
            weight_c, weight_t = self.expected_traffic[control], self.expected_traffic[treatment]
            with np.errstate(divide="ignore", invalid="ignore"):
                observed = pair["n_treatment"] / (pair["n_control"] + pair["n_treatment"])
            frames.append(pair.assign(
                control=control,
                treatment=treatment,
                expected_treatment_share=weight_t / (weight_c + weight_t),
                observed_treatment_share=observed,
                p_value=stats.srm_p_value(pair["n_control"], pair["n_treatment"], weight_c, weight_t),
                _order=order,
            ))
        if not frames:
            return pd.DataFrame(columns=columns)
        out = pd.concat(frames, ignore_index=True)
        out["_overall"] = out["user_segment"] != OVERALL
        sort = ["metric_table", "_overall", "user_segment", "user_segment_value"] + (["day"] if granularity == DAILY else [])
        return out.sort_values(sort + ["_order"], kind="mergesort")[columns].reset_index(drop=True)

    def _segments(self, column: str, segment: SegmentArg, granularity: str, other: str, other_value: str) -> pd.DataFrame:
        t = self._table
        rows = (t["granularity"] == granularity) & (t[column] != OVERALL) & (t[other] == other_value)
        name = None
        if segment is not None:
            name = segment if isinstance(segment, str) else SEPARATOR.join(segment)
            rows &= t[column] == name
        out = t[rows].reset_index(drop=True)
        if name is None or out.empty:
            return out
        dims = name.split(SEPARATOR)
        values = out[f"{column}_value"]
        if len(dims) == 1:
            parts = values.to_frame(0)
        else:
            parts = values.str.split(SEPARATOR, n=len(dims) - 1, expand=True, regex=False)
        position = out.columns.get_loc(f"{column}_value") + 1
        for i, dim in enumerate(dims):
            if dim not in out.columns:
                out.insert(position + i, dim, parts[i])
        return out

    # ---------------------------------------------------------------------------------- display

    def to_html(self, frame: Optional[pd.DataFrame] = None, max_rows: int = 500) -> str:
        """An HTML table of ``frame`` (default: ``overall()``), rows coloured by outcome."""
        frame = self.overall() if frame is None else frame
        segmented = bool(len(frame)) and bool(
            ((frame["user_segment"] != OVERALL) | (frame["metric_segment"] != OVERALL)).any()
        )
        dated = bool(len(frame)) and bool(frame["day"].notna().any())
        headers = ["Metric"] + (["Day"] if dated else []) + (["Segment"] if segmented else [])
        headers += ["Treatment", "Control value", "Treatment value", "Δ % [CI]", "p-value", "Units (C / T)"]
        cell = 'style="padding:2px 8px;text-align:{align}"'
        out = ['<table class="breakitdown" style="border-collapse:collapse">', "<thead><tr>"]
        out += [f'<th {cell.format(align="left")}>{html.escape(h)}</th>' for h in headers]
        out.append("</tr></thead><tbody>")
        for row in frame.head(max_rows).itertuples(index=False):
            color = _ROW_COLORS.get(row.outcome)
            style = f' style="background:{color};color:#1a1a1a"' if color else ""
            metric = row.label + ("" if _missing(row.quantile) else f" p{row.quantile * 100:g}")
            values = [metric]
            if dated:
                values.append("" if pd.isna(row.day) else f"{row.day:%Y-%m-%d}")
            if segmented:
                values.append(_segment_text(row))
            values += [
                row.treatment,
                _number(row.value_control),
                _number(row.value_treatment),
                _pct(row.pct_diff, row.pct_diff_ci_lower, row.pct_diff_ci_upper),
                _p_value(row.p_value),
                f"{_count(row.n_control)} / {_count(row.n_treatment)}",
            ]
            aligns = ["left"] * (len(headers) - 5) + ["right"] * 5
            out.append(f"<tr{style}>")
            out += [f'<td {cell.format(align=a)}>{html.escape(str(v))}</td>' for v, a in zip(values, aligns)]
            out.append("</tr>")
        out.append("</tbody></table>")
        if len(frame) > max_rows:
            out.append(f"<p>{len(frame) - max_rows} more rows not shown.</p>")
        return "\n".join(out)

    def _repr_html_(self) -> str:
        return self.to_html()

    def __repr__(self) -> str:
        metrics = self._table["metric"].nunique() if len(self._table) else 0
        return (f"<ExperimentResults: {len(self._table)} comparisons, {metrics} metrics, "
                f"flights {', '.join(self.flights)}>")


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _number(value: Any) -> str:
    return "–" if _missing(value) else f"{value:,.4g}"


def _count(value: Any) -> str:
    return "–" if _missing(value) else f"{value:,.0f}"


def _pct(value: Any, low: Any, high: Any) -> str:
    if _missing(value):
        return "–"
    interval = "" if _missing(low) or _missing(high) else f" [{low:+.2f}, {high:+.2f}]"
    return f"{value:+.2f}%{interval}"


def _p_value(value: Any) -> str:
    if _missing(value):
        return "–"
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


def _segment_text(row: Any) -> str:
    parts = []
    if row.user_segment != OVERALL:
        parts.append(f"{row.user_segment}={row.user_segment_value}")
    if row.metric_segment != OVERALL:
        parts.append(f"{row.metric_segment}={row.metric_segment_value}")
    return "; ".join(parts) or OVERALL
