"""The statistics layer, checked against independent computations on raw unit-level data."""

import numpy as np
import pytest
from scipy import stats as st

from breakitdown import stats


def _mean(x):
    return stats.mean_estimate([len(x)], [x.sum()], [(x * x).sum()])


def test_count_matches_welch_t_test_on_raw_values():
    rng = np.random.default_rng(0)
    xc, xt = rng.poisson(3.0, 800).astype(float), rng.poisson(3.3, 700).astype(float)
    out = stats.compare(_mean(xc), _mean(xt))
    ref = st.ttest_ind(xt, xc, equal_var=False)
    low, high = ref.confidence_interval(0.95)
    assert out["diff"][0] == pytest.approx(xt.mean() - xc.mean(), rel=1e-12)
    assert out["p_value"][0] == pytest.approx(ref.pvalue, rel=1e-10)
    assert out["diff_ci_lower"][0] == pytest.approx(low, rel=1e-10)
    assert out["diff_ci_upper"][0] == pytest.approx(high, rel=1e-10)


def test_binary_matches_welch_t_test_on_indicators():
    rng = np.random.default_rng(1)
    xc, xt = (rng.random(900) < 0.30).astype(float), (rng.random(950) < 0.34).astype(float)
    control = stats.mean_estimate([len(xc)], [xc.sum()], [xc.sum()], events=[xc.sum()])
    treatment = stats.mean_estimate([len(xt)], [xt.sum()], [xt.sum()], events=[xt.sum()])
    out = stats.compare(control, treatment)
    assert out["p_value"][0] == pytest.approx(st.ttest_ind(xt, xc, equal_var=False).pvalue, rel=1e-10)
    assert control.variance[0] == pytest.approx(xc.mean() * (1 - xc.mean()) / (len(xc) - 1), rel=1e-12)


def test_ratio_matches_the_linearised_variance():
    rng = np.random.default_rng(2)

    def unit_values(scale, n=3000):
        den = rng.poisson(5, n) + 1.0
        return rng.poisson(den * scale).astype(float), den

    def estimate(y, x):
        return stats.ratio_estimate(len(y), y.sum(), (y * y).sum(), x.sum(), (x * x).sum(), (x * y).sum())

    def linearised(y, x):
        r = y.sum() / x.sum()
        return np.var(y - r * x, ddof=1) / (len(y) * x.mean() ** 2)

    (yc, xc), (yt, xt) = unit_values(0.30), unit_values(0.32)
    control, treatment = estimate(yc, xc), estimate(yt, xt)
    assert float(control.variance) == pytest.approx(linearised(yc, xc), rel=1e-9)
    out = stats.compare(control, treatment)
    z = (yt.sum() / xt.sum() - yc.sum() / xc.sum()) / np.sqrt(linearised(yc, xc) + linearised(yt, xt))
    assert float(out["p_value"]) == pytest.approx(2 * st.norm.sf(abs(z)), rel=1e-9)


def test_quantile_variance_is_the_unit_linearisation_and_ignores_empty_units():
    rng = np.random.default_rng(3)
    units = 2000
    rows = rng.poisson(3, units) + 1
    values = [rng.poisson(40, k) for k in rows]
    q, density = float(np.quantile(np.concatenate(values), 0.5)), 0.05
    above = np.array([(v > q).sum() for v in values], dtype=float)
    m = rows.astype(float)

    def estimate(n_units):
        return stats.quantile_estimate(q, density, m.sum(), (m * m).sum(), above.sum(), (above * above).sum(),
                                       (above * m).sum(), n_units)

    r = above.sum() / m.sum()
    expected = ((above - r * m) ** 2).sum() / m.sum() ** 2 / density ** 2
    assert float(estimate(units).variance) == pytest.approx(expected, rel=1e-9)
    # Units without rows change only n, never the variance.
    assert float(estimate(units + 5000).variance) == pytest.approx(expected, rel=1e-9)


def test_relative_ci_uses_the_delta_method():
    control = stats.Estimate(np.array([10.0]), np.array([0.04]), np.array([1e6]))
    treatment = stats.Estimate(np.array([10.5]), np.array([0.05]), np.array([1e6]))
    out = stats.compare(control, treatment)
    se = np.sqrt(0.05 / 10.0 ** 2 + 10.5 ** 2 * 0.04 / 10.0 ** 4)
    z = st.norm.ppf(0.975)
    assert out["pct_diff"][0] == pytest.approx(5.0)
    assert out["pct_diff_ci_lower"][0] == pytest.approx(100 * (0.05 - z * se))
    assert out["pct_diff_ci_upper"][0] == pytest.approx(100 * (0.05 + z * se))


def test_bh_adjust():
    p = [0.001, 0.002, 0.002, 0.002, 0.3, 0.3, 0.9, 0.9, 0.99]
    assert stats.bh_adjust(p) == pytest.approx([0.0045, 0.0045, 0.0045, 0.0045, 0.45, 0.45, 0.99, 0.99, 0.99])
    # Against the definition: the adjusted p_(i) is min over j >= i of p_(j) * m / j, capped at 1.
    rng = np.random.default_rng(4)
    p = rng.uniform(size=30) ** 3
    p[[4, 11]] = np.nan
    ok = ~np.isnan(p)
    ranked, order = np.sort(p[ok]), np.argsort(p[ok])
    m = len(ranked)
    expected = np.empty(m)
    expected[order] = [min(min(ranked[j] * m / (j + 1) for j in range(i, m)), 1.0) for i in range(m)]
    out = stats.bh_adjust(p)
    assert np.isnan(out[[4, 11]]).all()
    np.testing.assert_allclose(out[ok], expected, rtol=1e-12)


def test_heterogeneity_equals_the_wald_test_of_contrasts():
    rng = np.random.default_rng(5)
    effects, variances = rng.normal(size=5), rng.uniform(0.5, 2.0, size=5)
    k = len(effects)
    contrasts = np.hstack([np.ones((k - 1, 1)), -np.eye(k - 1)])  # d_1 - d_j
    d = contrasts @ effects
    wald = float(d @ np.linalg.inv(contrasts @ np.diag(variances) @ contrasts.T) @ d)
    q, dof, p = stats.heterogeneity_test(effects, variances)
    assert (q, dof) == (pytest.approx(wald, rel=1e-9), k - 1)
    assert p == pytest.approx(st.chi2.sf(wald, k - 1), rel=1e-9)


def test_srm_is_pairwise():
    # control 1000 vs a treatment with 2100 units in a 1:2 design, and vs one with 1060 units in 1:1.
    p_12 = stats.srm_p_value([1000], [2100], 1, 2)[0]
    p_11 = stats.srm_p_value([1000], [1060], 1, 1)[0]
    assert p_12 == pytest.approx(st.chisquare([1000, 2100], [3100 / 3, 6200 / 3]).pvalue, rel=1e-9)
    assert p_11 == pytest.approx(st.chisquare([1000, 1060]).pvalue, rel=1e-9)
    # Same as a two-sided z-test of the treatment share with the expected share in the variance.
    share, n = 2 / 3, 3100
    z = (2100 / n - share) / np.sqrt(share * (1 - share) / n)
    assert p_12 == pytest.approx(2 * st.norm.sf(abs(z)), rel=1e-9)


def test_compare_degenerate_groups_give_nan():
    empty = stats.mean_estimate([0], [0], [0])
    some = stats.mean_estimate([500], [250], [400])
    out = stats.compare(empty, some)
    assert np.isnan(out["p_value"][0]) and np.isnan(out["pct_diff"][0])
    constant = stats.mean_estimate([500], [500], [500])  # every unit has value 1: zero variance
    out = stats.compare(constant, constant)
    assert out["diff"][0] == 0 and np.isnan(out["p_value"][0])
