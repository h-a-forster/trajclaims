"""Estimands 3 and 4: verification against claims, and model against scaffold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trajclaims import stats
from trajclaims.rules import LABELS

CLAIM = LABELS[0]
MIN_SCAFFOLDS = 3
REPS = 2000
LEVELS = ("none", "adhoc-check", "test-runner", "task-check")


def _odds(fit_frame: pd.DataFrame, numeric: list[str], factors: list[str]) -> dict[str, Any]:
    x = stats.design(fit_frame, numeric, factors)
    fit = stats.logit_cluster(x, fit_frame["y"].to_numpy(), fit_frame["task"].to_numpy(), 1e-6)
    out: dict[str, Any] = {}
    for name in numeric:
        i = fit.names.index(name)
        out[name] = {
            "odds_ratio": float(np.exp(fit.beta[i])),
            "ci": [
                float(np.exp(fit.beta[i] - stats.Z95 * fit.se[i])),
                float(np.exp(fit.beta[i] + stats.Z95 * fit.se[i])),
            ],
        }
    return out


def verification_tables(labelled: pd.DataFrame) -> dict[str, Any]:
    """What failed trials ran, and its association with claiming completion.

    Task fixed effects, not the random intercept first planned: harder tasks plausibly draw
    more checking and more failure, which a random intercept assumes away, and with hundreds of
    failed trials per task the incidental-parameter bias of fixed effects is negligible.
    """
    failed = labelled[labelled["reward"] == 0]
    by_scaffold: dict[str, Any] = {}
    for scaffold, group in failed.groupby("scaffold", sort=True):
        shell = group["shell_commands"].sum()
        counts = group["verification"].value_counts()
        by_scaffold[str(scaffold)] = {
            "failed": len(group),
            **{name: int(counts.get(name, 0)) for name in LEVELS},
            "after_last_edit": int(group["verified_after_edit"].sum()),
            "opaque_share": float(group["opaque_commands"].sum() / shell) if shell else 0.0,
            "blank_share": float(group["blank_commands"].sum() / shell) if shell else 0.0,
        }
    shipped = sorted(failed.loc[failed["verification"] == "task-check", "task"].unique())
    own_check = {
        task: {
            "failed": int((failed["task"] == task).sum()),
            "ran_own_check": int(
                ((failed["task"] == task) & (failed["verification"] == "task-check")).sum()
            ),
        }
        for task in shipped
    }
    # Blank commands hide what ran, so those trials cannot be classified.
    usable = failed[
        (failed["status"] == "readable")
        & (failed["shell_commands"] > 0)
        & (failed["blank_commands"] <= failed["shell_commands"] / 2)
    ].copy()
    usable["y"] = (usable["label"] == CLAIM).astype(float)
    usable["adhoc_check"] = (usable["verification"] == "adhoc-check").astype(float)
    usable["test_run"] = usable["verification"].isin(("test-runner", "task-check")).astype(float)
    usable["after_edit"] = usable["verified_after_edit"].astype(float)
    constant = usable.groupby("task")["y"].transform("nunique") == 1
    fit_frame = usable[~constant]
    level_counts = {
        name: {
            "n": int((usable["verification"] == name).sum()),
            "claims_done": int(usable.loc[usable["verification"] == name, "y"].sum()),
        }
        for name in LEVELS
    }
    checks = ["adhoc_check", "test_run"]
    return {
        "by_scaffold": by_scaffold,
        "own_check_tasks": own_check,
        "regression_n": len(fit_frame),
        "regression_tasks": int(fit_frame["task"].nunique()),
        "regression_tasks_dropped_constant": int(usable.loc[constant, "task"].nunique()),
        "claim_rate_by_level": level_counts,
        "odds_unadjusted": _odds(fit_frame, checks, []),
        "odds_scaffold_model": _odds(fit_frame, checks, ["scaffold", "model"]),
        "odds_adjusted": _odds(fit_frame, checks, ["scaffold", "model", "task"]),
        "odds_after_edit": _odds(fit_frame, ["after_edit"], ["scaffold", "model", "task"]),
    }


def _shares(
    num: np.ndarray,
    den: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    """Shapley shares of between-cell variance, with task-bootstrap intervals."""
    w_point = den.sum(0)
    y_point = num.sum(0) / w_point
    point = stats.shapley_two_way(y_point, w_point, rows, cols)
    draws = stats.ratio_draws(num, den, weights)
    w_draws = weights @ den
    replicates = [
        stats.shapley_two_way(d, w, rows, cols)
        for d, w in zip(draws, w_draws, strict=True)
        if np.isfinite(d).all() and (w > 0).all()
    ]
    between = np.average((y_point - np.average(y_point, weights=w_point)) ** 2, weights=w_point)
    noise = np.average(np.nanvar(draws, axis=0, ddof=1), weights=w_point) / between
    return {
        "model_share": point["row"],
        "scaffold_share": point["col"],
        "rest_share": point["rest"],
        "model_share_ci": np.quantile([r["row"] for r in replicates], [0.025, 0.975]).tolist(),
        "scaffold_share_ci": np.quantile([r["col"] for r in replicates], [0.025, 0.975]).tolist(),
        "noise_over_between": float(noise),
        "replicates": len(replicates),
    }


def _fixed(
    names: np.ndarray, others: np.ndarray, fcr: np.ndarray, draws: np.ndarray, key: str
) -> list[dict[str, Any]]:
    """Range of FCR across the other factor, for each level of one factor."""
    out = []
    for name in sorted(set(names)):
        index = np.flatnonzero(names == name)
        if len(index) < 2:
            continue
        values = fcr[index]
        ranges = np.nanmax(draws[:, index], axis=1) - np.nanmin(draws[:, index], axis=1)
        out.append(
            {
                key: str(name),
                "cells": {str(o): float(v) for o, v in zip(others[index], values, strict=True)},
                "range": float(values.max() - values.min()),
                "range_ci": np.quantile(ranges, [0.025, 0.975]).tolist(),
            }
        )
    return out


def model_vs_scaffold(labelled: pd.DataFrame, min_cell: int, seed: int) -> dict[str, Any]:
    """Two-way decomposition on models seen under at least three scaffolds with enough failures.

    Clustered fixed-effects coefficients answer "which levels differ"; Shapley shares of the
    cell means answer "how much of the variation each factor carries". Cell means are shown raw
    and with task fixed effects removed, so a scaffold is not credited for its task mix.
    """
    known = labelled[(labelled["reward"] == 0) & (labelled["status"] != "lost")].copy()
    sizes = known.groupby(["model", "scaffold"]).size()
    sizes = sizes[sizes >= min_cell]
    seen = sizes.groupby("model").size()
    models = sorted(seen[seen >= MIN_SCAFFOLDS].index)
    if not models:
        return {"models": [], "cells": 0}
    keep = sizes[sizes.index.get_level_values("model").isin(models)]
    data = known.set_index(["model", "scaffold"]).loc[keep.index].reset_index()
    data["y"] = (data["label"] == CLAIM).astype(float)
    data["cell"] = data["scaffold"] + "|" + data["model"]

    saturated = stats.ols_cluster(
        stats.design(data, [], ["cell", "task"]), data["y"].to_numpy(), data["task"].to_numpy()
    )
    effects = {
        n[5:]: b for n, b in zip(saturated.names, saturated.beta, strict=True) if n[:5] == "task="
    }
    data["y_adj"] = data["y"] - data["task"].map(effects).fillna(0.0)

    main = stats.ols_cluster(
        stats.design(data, [], ["scaffold", "model", "task"]),
        data["y"].to_numpy(),
        data["task"].to_numpy(),
    )
    terms = {
        n: {"coef": float(b), "se": float(s)}
        for n, b, s in zip(main.names, main.beta, main.se, strict=True)
        if n.startswith(("scaffold=", "model="))
    }

    cells = pd.Index(sorted(data["cell"].unique()))
    tasks = pd.Index(sorted(labelled["task"].unique()))
    t, c = tasks.get_indexer(data["task"]), cells.get_indexer(data["cell"])
    den, raw, adj = (np.zeros((len(tasks), len(cells))) for _ in range(3))
    np.add.at(den, (t, c), 1.0)
    np.add.at(raw, (t, c), data["y"].to_numpy())
    np.add.at(adj, (t, c), data["y_adj"].to_numpy())
    rows = np.array([s.split("|")[1] for s in cells])
    cols = np.array([s.split("|")[0] for s in cells])
    weights = stats.task_weights(len(tasks), REPS, seed)
    draws = stats.ratio_draws(raw, den, weights)
    fcr = raw.sum(0) / den.sum(0)
    return {
        "models": models,
        "scaffolds": sorted(set(cols.tolist())),
        "cells": len(cells),
        "failed_trials": len(data),
        "raw": _shares(raw, den, rows, cols, weights),
        "task_adjusted": _shares(adj, den, rows, cols, weights),
        "main_effects_with_task_fe": terms,
        "scaffold_spread_for_fixed_model": _fixed(rows, cols, fcr, draws, "model"),
        "model_spread_for_fixed_scaffold": _fixed(cols, rows, fcr, draws, "scaffold"),
    }
