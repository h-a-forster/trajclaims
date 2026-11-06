"""Task-clustered estimators. Every stochastic function takes a seed."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

Z95 = float(sps.norm.ppf(0.975))


def task_weights(n_tasks: int, reps: int, seed: int) -> np.ndarray:
    """Bootstrap counts over tasks: `reps` x `n_tasks`, each row summing to `n_tasks`."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_tasks, np.full(n_tasks, 1 / n_tasks), size=reps).astype(np.float64)


def ratio_draws(num: np.ndarray, den: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Bootstrap draws of pooled ratios; `num`, `den` are tasks x groups. NaN if a draw has no
    denominator in a group."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (weights @ num) / (weights @ den)


def percentile(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval per column; a group with no denominator in any draw gives NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanquantile(draws, 0.025, axis=0), np.nanquantile(draws, 0.975, axis=0)


def bias_corrected(point: float, draws: np.ndarray) -> tuple[float, float]:
    """Efron's bias-corrected percentile interval.

    For a statistic whose bootstrap distribution sits to one side of the estimate. Variance
    shares are one: resampling adds noise to the cell means, which moves replicates towards the
    residual, so a plain percentile interval is off-centre.
    """
    below = (np.mean(draws < point) + np.mean(draws <= point)) / 2
    z0 = sps.norm.ppf(np.clip(below, 1 / (2 * len(draws)), 1 - 1 / (2 * len(draws))))
    low, high = sps.norm.cdf([2 * z0 - Z95, 2 * z0 + Z95])
    return float(np.quantile(draws, low)), float(np.quantile(draws, high))


def range_interval(point: np.ndarray, draws: np.ndarray) -> tuple[float, float]:
    """Interval for max - min of group means, from a simultaneous band on their differences.

    A percentile interval of bootstrapped ranges cannot reach zero, because a resampled range
    is never below the gap noise opens between equal groups. Here every pairwise difference
    gets a studentised max-|t| band, so the interval reaches zero when no pair is separated.
    """
    left, right = np.triu_indices(len(point), k=1)
    diff = point[left] - point[right]
    spread = draws[:, left] - draws[:, right]
    se = np.nanstd(spread, axis=0, ddof=1)
    t = np.nanmax(np.abs(spread - diff) / np.where(se > 0, se, np.inf), axis=1)
    critical = np.nanquantile(t, 0.95)
    return (
        float(max(0.0, np.max(np.abs(diff) - critical * se))),
        float(np.max(np.abs(diff) + critical * se)),
    )


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, in input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    scaled = (len(p) - np.arange(len(p))) * p[order]
    adjusted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    out = np.empty(len(p))
    out[order] = adjusted
    return out


def contrasts(point: np.ndarray, draws: np.ndarray) -> pd.DataFrame:
    """All pairwise differences of groups, as Wald tests with bootstrap SEs, Holm-adjusted.

    Bootstrap tail p-values cannot be used here: with R draws they stop at 2/R, which is above
    Holm's first threshold once there are more than about R/40 contrasts.
    """
    left, right = np.triu_indices(len(point), k=1)
    diff = point[left] - point[right]
    se = np.nanstd(draws[:, left] - draws[:, right], axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, diff / se, 0.0)
    p = 2 * sps.norm.sf(np.abs(z))
    return pd.DataFrame(
        {
            "left": left,
            "right": right,
            "diff": diff,
            "se": se,
            "z": z,
            "p": p,
            "p_holm": holm(p),
        }
    )


def rogan_gladen(p: np.ndarray, se: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """Prevalence corrected for a classifier with sensitivity `se` and specificity `sp`."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (p + sp - 1) / (se + sp - 1)


def calibrated(p: np.ndarray, ppv: np.ndarray, npv: np.ndarray) -> np.ndarray:
    """Prevalence from the rule rate and its predictive values (inverse calibration).

    Uses the same validation counts as Rogan-Gladen, read the other way. It does not depend on
    the validation sample's own rule rate matching the population's, and stays within [0, 1].
    """
    return p * ppv + (1 - p) * (1 - npv)


def beta_draws(hits: int, total: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Jeffreys posterior draws of a proportion, to carry validation-sample noise."""
    return rng.beta(hits + 0.5, total - hits + 0.5, size=reps)


def wilson(hits: float, total: float) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p, z2 = hits / total, Z95**2
    centre = (p + z2 / (2 * total)) / (1 + z2 / total)
    half = Z95 * np.sqrt(p * (1 - p) / total + z2 / (4 * total**2)) / (1 + z2 / total)
    return float(centre - half), float(centre + half)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Unweighted kappa of two label lists."""
    x, y = pd.Series(a), pd.Series(b)
    observed = float((x == y).mean())
    px, py = x.value_counts(normalize=True), y.value_counts(normalize=True)
    expected = float(px.mul(py, fill_value=0).sum())
    return (observed - expected) / (1 - expected)


def weighted_auroc(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    """AUROC with case weights, ties counted one half. `y` is 0/1."""
    order = np.argsort(score, kind="stable")
    s, t, w = score[order], y[order], weight[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    pos = np.add.reduceat(w * t, starts)
    neg = np.add.reduceat(w * (1 - t), starts)
    below = np.r_[0.0, np.cumsum(neg)[:-1]]
    total = pos.sum() * neg.sum()
    return float((pos * (below + neg / 2)).sum() / total) if total > 0 else float("nan")


def auroc_draws(
    y: np.ndarray, score: np.ndarray, tasks: np.ndarray, reps: int, seed: int
) -> np.ndarray:
    """Task-clustered bootstrap draws of AUROC."""
    codes, _ = pd.factorize(tasks)
    weights = task_weights(codes.max() + 1, reps, seed)
    return np.array([weighted_auroc(y, score, row[codes]) for row in weights])


@dataclass(frozen=True, slots=True)
class Fit:
    names: list[str]
    beta: np.ndarray
    se: np.ndarray


def _crv1(x: np.ndarray, scores: np.ndarray, bread: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cluster-robust (CR1) covariance from per-row scores and the inverse Hessian."""
    codes, uniques = pd.factorize(groups)
    n, k, g = x.shape[0], x.shape[1], len(uniques)
    summed = np.zeros((g, k))
    np.add.at(summed, codes, scores)
    meat = summed.T @ summed
    scale = g / (g - 1) * (n - 1) / (n - k)
    return scale * bread @ meat @ bread


def design(frame: pd.DataFrame, numeric: list[str], factors: list[str]) -> pd.DataFrame:
    """Intercept, numeric columns and dummies for each factor, first level dropped."""
    parts = [pd.Series(1.0, index=frame.index, name="const"), frame[numeric].astype(float)]
    for factor in factors:
        parts.append(
            pd.get_dummies(
                frame[factor], prefix=factor, prefix_sep="=", drop_first=True, dtype=float
            )
        )
    return pd.concat(parts, axis=1)


def logit_cluster(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, ridge: float = 0.0) -> Fit:
    """Maximum-likelihood logit by Newton steps, with task-clustered CR1 standard errors."""
    matrix = x.to_numpy(dtype=float)
    beta = np.zeros(matrix.shape[1])
    penalty = ridge * np.eye(matrix.shape[1])
    for _ in range(100):
        eta = matrix @ beta
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-10, None)
        hessian = (matrix * w[:, None]).T @ matrix + penalty
        step = np.linalg.solve(hessian, matrix.T @ (y - p) - penalty @ beta)
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    else:
        raise ArithmeticError("logit did not converge")
    p = 1 / (1 + np.exp(-(matrix @ beta)))
    bread = np.linalg.inv((matrix * (p * (1 - p))[:, None]).T @ matrix + penalty)
    scores = matrix * (y - p)[:, None]
    cov = _crv1(matrix, scores, bread, groups)
    return Fit(list(x.columns), beta, np.sqrt(np.diag(cov)))


def ols_cluster(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> Fit:
    """Least squares with task-clustered CR1 standard errors."""
    matrix = x.to_numpy(dtype=float)
    bread = np.linalg.pinv(matrix.T @ matrix)
    beta = bread @ matrix.T @ y
    scores = matrix * (y - matrix @ beta)[:, None]
    cov = _crv1(matrix, scores, bread, groups)
    return Fit(list(x.columns), beta, np.sqrt(np.diag(cov)))


def _weighted_r2(y: np.ndarray, w: np.ndarray, x: np.ndarray) -> float:
    root = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)
    fitted = x @ beta
    mean = np.average(y, weights=w)
    return float(1 - np.sum(w * (y - fitted) ** 2) / np.sum(w * (y - mean) ** 2))


def shapley_two_way(
    y: np.ndarray, w: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> dict[str, float]:
    """Shares of between-cell variance: row, column and the rest (interaction plus noise).

    The two main effects are not orthogonal in an unbalanced table, so each gets its Shapley
    value: the mean of its gain in weighted R^2 over the two orders of entry.
    """
    a = pd.get_dummies(rows, dtype=float).to_numpy()
    b = pd.get_dummies(cols, dtype=float).to_numpy()
    r2_a, r2_b = _weighted_r2(y, w, a), _weighted_r2(y, w, b)
    r2_ab = _weighted_r2(y, w, np.hstack([a, b]))
    return {
        "row": (r2_a + r2_ab - r2_b) / 2,
        "col": (r2_b + r2_ab - r2_a) / 2,
        "rest": 1 - r2_ab,
    }
