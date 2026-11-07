"""Estimators, including the size of the contrast test on a clustered null."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from trajclaims import stats


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5])
    assert stats.holm(p) == pytest.approx(multipletests(p, method="holm")[1])


def test_holm_nan_stays_nan_and_leaves_the_family():
    nan = np.nan
    assert stats.holm([0.01, nan, 0.02]) == pytest.approx([0.02, nan, 0.02], nan_ok=True)
    assert stats.holm([nan, 0.01, 0.02]) == pytest.approx([nan, 0.02, 0.02], nan_ok=True)
    assert stats.holm([nan, nan]) == pytest.approx([nan, nan], nan_ok=True)


def test_rogan_gladen_recovers_a_known_prevalence():
    truth, se, sp = 0.7, 0.9, 0.8
    observed = truth * se + (1 - truth) * (1 - sp)
    assert stats.rogan_gladen(observed, se, sp) == pytest.approx(truth)


def test_calibration_recovers_a_known_prevalence():
    truth, se, sp = 0.7, 0.9, 0.8
    said_yes = truth * se + (1 - truth) * (1 - sp)
    ppv = truth * se / said_yes
    npv = (1 - truth) * sp / (1 - said_yes)
    assert stats.calibrated(said_yes, ppv, npv) == pytest.approx(truth)


def test_kappa_known_values():
    assert stats.cohen_kappa(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    assert stats.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)


def test_weighted_auroc_matches_expansion():
    y = np.array([0, 0, 1, 1, 1])
    score = np.array([0.1, 0.4, 0.4, 0.8, 0.2])
    weight = np.array([2.0, 1.0, 1.0, 3.0, 2.0])
    expanded = np.repeat(np.arange(5), weight.astype(int))
    pos = score[expanded][y[expanded] == 1]
    neg = score[expanded][y[expanded] == 0]
    direct = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert stats.weighted_auroc(y, score, weight) == pytest.approx(direct)


def test_logit_and_ols_match_statsmodels_cluster_errors():
    rng = np.random.default_rng(1)
    n, groups = 3000, rng.integers(0, 40, 3000)
    shared = rng.normal(0, 0.5, 40)[groups]
    x = pd.DataFrame({"const": 1.0, "a": rng.normal(size=n), "b": rng.integers(0, 2, n) * 1.0})
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.2 + 0.5 * x["a"] + shared)))).astype(float).to_numpy()

    mine = stats.logit_cluster(x, y, groups)
    ref = sm.Logit(y, x).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    assert mine.beta == pytest.approx(ref.params.to_numpy(), rel=1e-5)
    assert mine.se == pytest.approx(ref.bse.to_numpy(), rel=1e-2)

    mine = stats.ols_cluster(x, y, groups)
    ref = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    assert mine.beta == pytest.approx(ref.params.to_numpy(), rel=1e-8)
    assert mine.se == pytest.approx(ref.bse.to_numpy(), rel=1e-6)


def test_shapley_splits_an_additive_table():
    rows = np.array(["a", "a", "b", "b"])
    cols = np.array(["x", "y", "x", "y"])
    y = np.array([0.0, 2.0, 1.0, 3.0])
    shares = stats.shapley_two_way(y, np.ones(4), rows, cols)
    assert shares["row"] == pytest.approx(0.2)
    assert shares["col"] == pytest.approx(0.8)
    assert shares["rest"] == pytest.approx(0.0, abs=1e-12)


def _null_tables(rng, tasks=80, cells=8):
    """Equal claim rate in every cell; tasks shift it, and vary by cell, as real tasks do."""
    base = rng.normal(0.4, 1.0, tasks)
    trials = rng.integers(5, 30, (tasks, cells))
    prob = 1 / (1 + np.exp(-(base[:, None] + rng.normal(0, 0.3, (tasks, cells)))))
    return rng.binomial(trials, prob).astype(float), trials.astype(float)


@pytest.mark.slow
def test_contrast_size_and_holm_control_on_a_clustered_null():
    rng = np.random.default_rng(7)
    sims, rejected, any_holm, cover, covered = 400, 0, 0, 0, 0
    total = 0
    for sim in range(sims):
        num, den = _null_tables(rng)
        weights = stats.task_weights(num.shape[0], 500, sim)
        draws = stats.ratio_draws(num, den, weights)
        point = num.sum(0) / den.sum(0)
        table = stats.contrasts(point, draws)
        rejected += int((table["p"] < 0.05).sum())
        total += len(table)
        any_holm += int((table["p_holm"] < 0.05).any())
        low, high = stats.percentile(draws[:, 0:1] - draws[:, 1:2])
        cover += int(low[0] <= 0 <= high[0])
        covered += 1
    size = rejected / total
    mc_se = np.sqrt(0.05 * 0.95 / sims)  # contrasts within a simulation are correlated
    assert size < 0.05 + 3 * mc_se
    assert any_holm / sims < 0.05 + 3 * mc_se
    assert cover / covered > 0.95 - 3 * mc_se


def test_percentile_of_an_impossible_group_is_nan():
    low, high = stats.percentile(np.full((10, 1), np.nan))
    assert np.isnan(low[0]) and np.isnan(high[0])


def test_a_bias_corrected_interval_recentres_on_the_estimate():
    draws = np.random.default_rng(1).normal(0.6, 0.05, 20000)
    plain = np.quantile(draws, [0.025, 0.975])
    low, high = stats.bias_corrected(0.65, draws)
    assert low > plain[0] + 0.05 and high > plain[1] + 0.05
    centred = stats.bias_corrected(0.6, draws)
    assert centred == pytest.approx(tuple(plain), abs=0.005)


def test_a_range_interval_reaches_zero_for_equal_groups_and_not_for_separated_ones():
    rng = np.random.default_rng(2)
    equal = np.array([0.5, 0.5, 0.5])
    assert stats.range_interval(equal, rng.normal(equal, 0.05, (5000, 3)))[0] == 0.0
    apart = np.array([0.2, 0.5, 0.8])
    low, high = stats.range_interval(apart, rng.normal(apart, 0.02, (5000, 3)))
    assert 0.5 < low < 0.6 < high < 0.7
