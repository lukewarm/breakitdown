"""Pandas side: per-flight sufficient statistics -> one row per (metric, segment, treatment) comparison.

Also adds the cross-row results: Benjamini-Hochberg adjusted p-values, heterogeneous treatment
effect (HTE) tests across user segments, and the share/contribution of each metric segment.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import stats
from ._common import NULL_LABEL, OVERALL
from .aggregate import CUMULATIVE, DAILY
from .metrics import Metric

GROUP_KEYS = ["period", "granularity", "day", "user_segment", "user_segment_value", "metric_segment", "metric_segment_value"]
USER_KEYS = ["period", "granularity", "day", "user_segment", "user_segment_value"]
INFERENCE = ["diff_ci_lower", "diff_ci_upper", "pct_diff_ci_lower", "pct_diff_ci_upper", "p_value", "std_error", "pct_std_error"]

RESULT_COLUMNS = [
    "metric_table", "metric", "label", "kind", "direction", "quantile",
    "period", "granularity", "day",
    "user_segment", "user_segment_value", "metric_segment", "metric_segment_value",
    "control", "treatment", "n_control", "n_treatment",
    "value_control", "value_treatment",
    "diff", "diff_ci_lower", "diff_ci_upper",
    "pct_diff", "pct_diff_ci_lower", "pct_diff_ci_upper",
    "p_value", "p_value_bh", "significant", "outcome", "sufficient_sample", "srm_p_value",
    "std_error", "pct_std_error", "var_control", "var_treatment",
    "hte_tested", "hte_p_value", "hte_detected", "hte_pattern",
    "segment_share_control", "segment_contribution",
]

BH_FAMILY = ["period", "granularity", "day", "user_segment", "user_segment_value", "metric_segment",
             "metric_segment_value", "treatment"]
HTE_FAMILY = ["period", "granularity", "day", "metric_table", "metric", "quantile", "treatment", "user_segment",
              "metric_segment", "metric_segment_value"]
CONTRIBUTION_FAMILY = ["period", "granularity", "day", "metric_table", "metric", "treatment", "user_segment",
                       "user_segment_value", "metric_segment"]


def compare_table(
    table_name: str,
    keyed_metrics: Sequence[Tuple[str, Metric]],
    sufficient_stats: pd.DataFrame,
    counts: pd.DataFrame,
    flights: Sequence[str],
    *,
    alpha: float,
    enforce_min_sample: bool,
    expected_traffic: Optional[Mapping[str, float]],
) -> pd.DataFrame:
    """Compare every treatment with control for the metrics of one sufficient-statistics frame."""
    if sufficient_stats.empty:
        return pd.DataFrame()
    sufficient_stats = sufficient_stats.assign(day=pd.to_datetime(sufficient_stats["day"]))
    frames: List[pd.DataFrame] = []
    for granularity in (CUMULATIVE, DAILY):
        part = sufficient_stats[sufficient_stats["granularity"] == granularity]
        if part.empty:
            continue
        keys = GROUP_KEYS if granularity == DAILY else [k for k in GROUP_KEYS if k != "day"]
        user_keys = USER_KEYS if granularity == DAILY else [k for k in USER_KEYS if k != "day"]
        grid, groups = _complete_grid(part, counts[counts["granularity"] == granularity], keys, user_keys, flights)
        if "day" not in groups.columns:
            groups = groups.assign(day=pd.NaT)
        frames += _compare_grid(
            table_name, keyed_metrics, grid, groups, flights, alpha, enforce_min_sample, expected_traffic
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _complete_grid(
    part: pd.DataFrame,
    counts: pd.DataFrame,
    keys: List[str],
    user_keys: List[str],
    flights: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One row per group x flight in a fixed order, so per-flight arrays line up by reshaping.

    A flight without rows in a group still gets a row: its sums are NaN (read as zero) and its
    units come from the population.
    """
    groups = part[keys].drop_duplicates().sort_values(keys, kind="mergesort").reset_index(drop=True)
    grid = groups.loc[groups.index.repeat(len(flights))].reset_index(drop=True)
    grid["flight"] = np.tile(np.asarray(flights, dtype=object), len(groups))
    stat_cols = [c for c in part.columns if c not in GROUP_KEYS and c != "flight"]
    grid = grid.merge(part[keys + ["flight"] + stat_cols], on=keys + ["flight"], how="left", validate="one_to_one")
    grid = grid.merge(counts[user_keys + ["flight", "n"]], on=user_keys + ["flight"], how="left", validate="many_to_one")
    return grid, groups


def _compare_grid(
    table_name: str,
    keyed_metrics: Sequence[Tuple[str, Metric]],
    grid: pd.DataFrame,
    groups: pd.DataFrame,
    flights: Sequence[str],
    alpha: float,
    enforce_min_sample: bool,
    expected_traffic: Optional[Mapping[str, float]],
) -> List[pd.DataFrame]:
    k = len(flights)
    n = grid["n"].fillna(0.0).to_numpy(dtype=float)
    frames = []
    for key, metric in keyed_metrics:
        for output in metric.outputs(key):

            def stat(name: str, fill: float = 0.0, _prefix: str = output.prefix) -> np.ndarray:
                values = grid[f"{_prefix}__{name}"].to_numpy(dtype=float)
                return np.where(np.isnan(values), fill, values)

            estimate = metric.estimate(stat, n)
            control = estimate.take(slice(0, None, k))
            for j in range(1, k):
                treatment = estimate.take(slice(j, None, k))
                result = stats.compare(control, treatment, alpha)
                frame = groups.copy()
                frame["metric_table"] = table_name
                frame["metric"] = metric.name
                frame["label"] = metric.display_name
                frame["kind"] = metric.kind
                frame["direction"] = metric.direction.value
                frame["quantile"] = np.nan if output.quantile is None else output.quantile
                frame["control"] = flights[0]
                frame["treatment"] = flights[j]
                frame["n_control"] = control.n
                frame["n_treatment"] = treatment.n
                frame["value_control"] = control.point
                frame["value_treatment"] = treatment.point
                frame["var_control"] = control.variance
                frame["var_treatment"] = treatment.variance
                for name, values in result.items():
                    frame[name] = values
                with np.errstate(invalid="ignore"):
                    sufficient = np.asarray(metric.has_min_sample(control, treatment), dtype=bool)
                frame["sufficient_sample"] = sufficient
                if enforce_min_sample:
                    frame.loc[~sufficient, INFERENCE] = np.nan
                if expected_traffic:
                    frame["srm_p_value"] = stats.srm_p_value(
                        control.n, treatment.n, expected_traffic[flights[0]], expected_traffic[flights[j]]
                    )
                else:
                    frame["srm_p_value"] = np.nan
                frames.append(frame)
    return frames


def finalize(
    comparisons: pd.DataFrame,
    *,
    alpha: float,
    hte_alpha: float,
    hte_min_units: float,
    hte_scale: str,
    metric_order: Mapping[Tuple[str, str], int],
    user_segment_order: Mapping[str, int],
    metric_segment_order: Mapping[str, int],
    treatment_order: Mapping[str, int],
) -> pd.DataFrame:
    """Cross-row results (BH, outcome, HTE, contributions), then a stable row and column order."""
    if comparisons.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = comparisons.reset_index(drop=True)
    df["day"] = pd.to_datetime(df["day"])
    df["p_value_bh"] = df.groupby(BH_FAMILY, dropna=False, sort=False)["p_value"].transform(stats.bh_adjust)
    df["significant"] = (df["p_value"] < alpha).fillna(False).astype(bool)
    df["outcome"] = _outcome(df)
    _add_hte(df, hte_alpha, hte_min_units, hte_scale)
    _add_contributions(df)
    order = pd.DataFrame({
        "_granularity": (df["granularity"] != CUMULATIVE).astype(int),
        "_metric": [metric_order.get((t, m), 0) for t, m in zip(df["metric_table"], df["metric"])],
        "_user": df["user_segment"].map(lambda s: -1 if s == OVERALL else user_segment_order.get(s, 10**6)),
        "_metric_segment": df["metric_segment"].map(lambda s: -1 if s == OVERALL else metric_segment_order.get(s, 10**6)),
        "_treatment": df["treatment"].map(treatment_order),
    })
    df = pd.concat([df, order], axis=1).sort_values(
        ["_granularity", "day", "_metric", "quantile", "_user", "user_segment_value", "_metric_segment",
         "metric_segment_value", "_treatment"],
        kind="mergesort",
        na_position="first",
    )
    return df[RESULT_COLUMNS].reset_index(drop=True)


def _outcome(df: pd.DataFrame) -> np.ndarray:
    up = df["direction"] == "up"
    down = df["direction"] == "down"
    pos = df["diff"] > 0
    neg = df["diff"] < 0
    return np.select(
        [df["p_value"].isna(), ~df["significant"], (up & pos) | (down & neg), (up & neg) | (down & pos)],
        ["n/a", "flat", "better", "worse"],
        default="changed",
    )


def _add_hte(df: pd.DataFrame, hte_alpha: float, min_units: float, scale: str, pattern_alpha: float = 0.01) -> None:
    """Cochran's Q test of equal treatment effects across the values of each user segment."""
    df["hte_tested"] = False
    df["hte_p_value"] = np.nan
    df["hte_detected"] = False
    df["hte_pattern"] = None
    segmented = df["user_segment"] != OVERALL
    if not segmented.any():
        return
    if scale == "absolute":
        effect, se = df["diff"], df["std_error"]
    elif scale == "relative":
        effect, se = df["pct_diff"], df["pct_std_error"]
    else:
        raise ValueError(f"hte_scale must be 'absolute' or 'relative', got {scale!r}.")
    var = se * se
    eligible = (
        segmented
        & (df["n_control"] >= min_units)
        & (df["n_treatment"] >= min_units)
        & np.isfinite(effect)
        & np.isfinite(var)
        & (var > 0)
        & ~df["user_segment_value"].str.contains(NULL_LABEL, regex=False)
    )
    for _, family in df[segmented].groupby(HTE_FAMILY, dropna=False, sort=False):
        index = family.index
        used = index[eligible.loc[index].to_numpy()]
        if len(used) < 2:
            continue
        _, _, p_value = stats.heterogeneity_test(effect.loc[used].to_numpy(), var.loc[used].to_numpy())
        df.loc[used, "hte_tested"] = True
        df.loc[index, "hte_p_value"] = p_value
        if p_value < hte_alpha:
            df.loc[index, "hte_detected"] = True
            df.loc[index, "hte_pattern"] = _hte_pattern(effect.loc[used], df.loc[used, "p_value"], pattern_alpha)


def _hte_pattern(effect: pd.Series, p_value: pd.Series, alpha: float) -> str:
    significant = p_value < alpha
    up = int((significant & (effect > 0)).sum())
    down = int((significant & (effect < 0)).sum())
    if up and down:
        return "opposite directions"
    if up == len(effect) or down == len(effect):
        return "same direction, different sizes"
    if up or down:
        return "only some segments affected"
    return "no single segment significant"


def _add_contributions(df: pd.DataFrame) -> None:
    """For metric segments of Count/Binary metrics: share of the control value and of the change."""
    df["segment_share_control"] = np.nan
    df["segment_contribution"] = np.nan
    rows = (df["metric_segment"] != OVERALL) & df["kind"].isin(["count", "binary"])
    if not rows.any():
        return
    grouped = df[rows].groupby(CONTRIBUTION_FAMILY, dropna=False, sort=False)
    total_control = grouped["value_control"].transform("sum")
    total_diff = grouped["diff"].transform("sum")
    with np.errstate(divide="ignore", invalid="ignore"):
        df.loc[rows, "segment_share_control"] = df.loc[rows, "value_control"] / total_control
        df.loc[rows, "segment_contribution"] = df.loc[rows, "diff"] / total_diff.where(total_diff != 0)
