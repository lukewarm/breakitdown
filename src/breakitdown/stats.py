"""Statistical tests on sufficient statistics.

Pure numpy/scipy and vectorised: every function takes arrays with one element per group
(flight x segment x day), so a whole scorecard is a handful of array operations. Nothing here
knows about Spark, which keeps the statistics easy to test and to reuse.

An ``Estimate`` is a point estimate per group plus the variance of that estimate. Each metric kind
has an ``*_estimate`` constructor, and ``compare`` turns a control and a treatment estimate into
differences, confidence intervals and p-values. A variance-reduction method such as CUPED
(roadmap) slots in as another estimator that adjusts the point and the variance before ``compare``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as _st

__all__ = [
    "Estimate",
    "mean_estimate",
    "ratio_estimate",
    "quantile_estimate",
    "compare",
    "srm_p_value",
    "bh_adjust",
    "heterogeneity_test",
]


@dataclass(frozen=True)
class Estimate:
    """Point estimates per group and the variance of those estimates."""

    point: np.ndarray
    variance: np.ndarray
    n: np.ndarray
    """Randomization units in the group, including units without rows (they count as zeros)."""
    events: Optional[np.ndarray] = None
    """Binary: units with a positive value. Quantile: rows (events). Used by sample-size rules."""
    welch: bool = False
    """Mean metrics use a t reference distribution with Welch-Satterthwaite degrees of freedom."""

    def take(self, index) -> "Estimate":
        events = None if self.events is None else self.events[index]
        return Estimate(self.point[index], self.variance[index], self.n[index], events, self.welch)


def _f(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values, np.nan)


def mean_estimate(n, total, total_sq, events=None) -> Estimate:
    """Mean of unit values from the unit count, the sum and the sum of squares.

    Used for Count metrics (unit totals) and Binary metrics (0/1 indicators, where the sum of
    squares equals the sum).
    """
    n, total, total_sq = _f(n), _f(total), _f(total_sq)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = total / n
        unit_var = np.maximum(total_sq - n * mean * mean, 0.0) / (n - 1)
        variance = unit_var / n
    return Estimate(mean, variance, n, None if events is None else _f(events), welch=True)


def ratio_estimate(n, num, num_sq, den, den_sq, num_den) -> Estimate:
    """sum(numerator) / sum(denominator) with the delta-method variance.

    Units are the independent observations: ``num`` and ``den`` are sums of unit totals, the
    ``*_sq`` arguments sums of squared unit totals and ``num_den`` the sum of their products.
    The variance is ``(s_y^2 - 2 R s_xy + R^2 s_x^2) / (n x_bar^2)``, the usual delta-method
    formula written so that it cannot go negative through rounding.
    """
    n, num, num_sq, den, den_sq, num_den = (_f(a) for a in (n, num, num_sq, den, den_sq, num_den))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den
        den_mean = den / n
        num_var = (num_sq - num * num / n) / (n - 1)
        den_var = (den_sq - den * den / n) / (n - 1)
        cov = (num_den - num * den / n) / (n - 1)
        variance = np.maximum(num_var - 2 * ratio * cov + ratio * ratio * den_var, 0.0) / (n * den_mean * den_mean)
    return Estimate(ratio, variance, n)


def quantile_estimate(q, density, events, events_sq, above, above_sq, above_events, n) -> Estimate:
    """A quantile of the pooled rows with a unit-clustered variance.

    For quantile ``q`` let ``p_u`` be unit u's row count and ``z_u`` its rows above ``q``. The share
    of rows above ``q``, ``r = sum(z) / sum(p)``, is a ratio estimator with delta-method variance
    ``sum((z_u - r p_u)^2) / sum(p)^2``. Dividing by the squared density at ``q`` maps it to the
    quantile scale (Bahadur representation). Units without rows do not change it, so zero-filled
    units only change ``n``.
    """
    q, density = _f(q), _f(density)
    events, events_sq, above, above_sq, above_events, n = (
        _f(a) for a in (events, events_sq, above, above_sq, above_events, n)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        share = above / events
        share_var = np.maximum(above_sq - 2 * share * above_events + share * share * events_sq, 0.0) / (events * events)
        variance = share_var / (density * density)
    return Estimate(q, variance, n, events=events)


def compare(control: Estimate, treatment: Estimate, alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Treatment vs control: absolute and relative differences, confidence intervals, p-values.

    Mean metrics (Count, Binary) use Welch's t-test; ratio and quantile metrics a z-test. The
    relative difference ``100 * (T / C - 1)`` gets a delta-method interval for every metric kind.
    Non-finite results (empty groups, zero variance) come back as NaN.
    """
    c, t = control, treatment
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        diff = t.point - c.point
        var = t.variance + c.variance
        se = np.sqrt(var)
        z = diff / se
        if c.welch and t.welch:
            dof = var * var / (t.variance ** 2 / (t.n - 1) + c.variance ** 2 / (c.n - 1))
            p_value = 2 * _st.t.sf(np.abs(z), dof)
            crit = _st.t.ppf(1 - alpha / 2, dof)
        else:
            p_value = 2 * _st.norm.sf(np.abs(z))
            crit = np.full_like(diff, _st.norm.ppf(1 - alpha / 2))
        rel = t.point / c.point - 1
        rel_se = np.sqrt(t.variance / c.point ** 2 + t.point ** 2 * c.variance / c.point ** 4)
    out = {
        "diff": diff,
        "diff_ci_lower": diff - crit * se,
        "diff_ci_upper": diff + crit * se,
        "pct_diff": 100 * rel,
        "pct_diff_ci_lower": 100 * (rel - crit * rel_se),
        "pct_diff_ci_upper": 100 * (rel + crit * rel_se),
        "p_value": p_value,
        "std_error": se,
        "pct_std_error": 100 * rel_se,
    }
    return {name: _finite(values) for name, values in out.items()}


def srm_p_value(n_control, n_treatment, weight_control: float, weight_treatment: float) -> np.ndarray:
    """Sample-ratio-mismatch p-value of one control/treatment pair.

    Chi-square goodness of fit of the pair's unit counts against the expected split (1 dof), which
    equals a two-sided z-test of the treatment's share with the expected share in the variance.
    Each treatment is tested against control on its own, never all flights at once.
    """
    nc, nt = _f(n_control), _f(n_treatment)
    share = weight_treatment / (weight_control + weight_treatment)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_t = (nc + nt) * share
        expected_c = (nc + nt) - expected_t
        chi2 = (nt - expected_t) ** 2 / expected_t + (nc - expected_c) ** 2 / expected_c
        p_value = _st.chi2.sf(chi2, 1)
    return _finite(p_value)


def bh_adjust(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values. NaNs are left out of the family and stay NaN."""
    p = _f(p_values)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok], kind="mergesort")
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    adjusted = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    values = np.empty(m)
    values[order] = adjusted
    out[ok] = values
    return out


def heterogeneity_test(effects, variances) -> Tuple[float, int, float]:
    """Cochran's Q test that independent per-segment effects are all equal.

    Equivalent to the Wald chi-square test of the k - 1 contrasts ``d_1 - d_j``, without the
    matrix inverse. Returns (Q, degrees of freedom, p-value).
    """
    e, v = _f(effects), _f(variances)
    if len(e) < 2:
        return float("nan"), 0, float("nan")
    w = 1.0 / v
    mean = (w * e).sum() / w.sum()
    q = float((w * (e - mean) ** 2).sum())
    dof = len(e) - 1
    return q, dof, float(_st.chi2.sf(q, dof))
