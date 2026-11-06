"""Detect false success from text, after arXiv:2606.09863, with held-out tasks.

Among trials whose closing statement claims completion, predict that the verifier failed them.
The label is the verifier's, so it does not depend on text; the paper's false-success label is a
regex over the closing message. Each fold is a file under `unit_dir`, in a folder named for a hash
of the inputs, and is skipped on rerun with the same inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

from trajclaims.stats import auroc_draws, percentile, weighted_auroc

SEED = 20261106
FOLDS = 5
REPS = 2000
MIN_CLASS = 20
MIN_CELL_CLASS = 5
DETECTOR_SCHEMA = 1


def _tfidf_scores(train: pd.Series, y: np.ndarray, test: pd.Series) -> np.ndarray:
    """The paper's TF-IDF + logistic settings (its Appendix C)."""
    vectoriser = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42
    )
    model.fit(vectoriser.fit_transform(train), y)
    return model.predict_proba(vectoriser.transform(test))[:, 1]


def _id_scores(train: pd.DataFrame, y: np.ndarray, test: pd.DataFrame) -> np.ndarray:
    """Baseline with no text: which scaffold and model produced the trial."""
    encoder = OneHotEncoder(handle_unknown="ignore")
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear")
    model.fit(encoder.fit_transform(train), y)
    return model.predict_proba(encoder.transform(test))[:, 1]


def _cached(unit: Path, compute: Callable[[], np.ndarray]) -> np.ndarray:
    """Scores from `unit`, computing them if absent. Rounding happens first, on both paths:
    rounded scores tie where exact ones do not, and ties change AUROC."""
    if unit.exists():
        return np.array(json.loads(unit.read_text(encoding="utf-8")))
    scores = [round(float(s), 6) for s in compute()]
    part = unit.with_suffix(".part")
    part.write_text(json.dumps(scores), encoding="utf-8")
    os.replace(part, unit)
    return np.array(scores)


def _fold_scores(
    frame: pd.DataFrame, feature: str, fold: int, test_mask: np.ndarray, unit_dir: Path
) -> np.ndarray:
    def compute() -> np.ndarray:
        train, test = frame[~test_mask], frame[test_mask]
        y = train["failed"].to_numpy()
        if feature == "ids":
            columns = ["scaffold", "model"]
            return _id_scores(train[columns], y, test[columns])
        if feature == "closing+tail":
            return _tfidf_scores(
                train["closing"] + "\n" + train["tail"], y, test["closing"] + "\n" + test["tail"]
            )
        return _tfidf_scores(train[feature], y, test[feature])

    return _cached(unit_dir / f"s{DETECTOR_SCHEMA}-{feature}-{fold}.json", compute)


def _pair_scores(
    frame: pd.DataFrame, name: str, train: np.ndarray, test: np.ndarray, unit_dir: Path
) -> np.ndarray:
    """Closing-text scores for an explicit train and test split, cached as a unit."""

    def compute() -> np.ndarray:
        y = frame.loc[train, "failed"].to_numpy()
        return _tfidf_scores(frame.loc[train, "closing"], y, frame.loc[test, "closing"])

    return _cached(unit_dir / f"s{DETECTOR_SCHEMA}-closing-{name}.json", compute)


def _pooled_folds(y: np.ndarray, scores: np.ndarray, folds: np.ndarray) -> float:
    """Trial-weighted AUROC within folds, so scores of different models are not compared."""
    total = weight = 0.0
    for fold in np.unique(folds):
        ys = y[folds == fold]
        if 0 < ys.sum() < len(ys):
            total += len(ys) * weighted_auroc(ys, scores[folds == fold], np.ones(len(ys)))
            weight += len(ys)
    return total / weight if weight else float("nan")


def out_of_fold(frame: pd.DataFrame, feature: str, folds: np.ndarray, unit_dir: Path) -> np.ndarray:
    scores = np.zeros(len(frame))
    for fold in np.unique(folds):
        mask = folds == fold
        scores[mask] = _fold_scores(frame, feature, str(fold), mask, unit_dir)
    return scores


def _within_cells(frame: pd.DataFrame, scores: np.ndarray, folds: np.ndarray) -> tuple[float, int]:
    """Mean AUROC inside scaffold x model x fold, weighted by trials.

    Scaffold and model style cannot help inside a cell, and scores from different folds' models
    are never compared.
    """
    total = weight = 0.0
    used = set()
    y = frame["failed"].to_numpy()
    for key, index in frame.groupby(["scaffold", "model", folds]).indices.items():
        ys = y[index]
        if min(ys.sum(), len(ys) - ys.sum()) < MIN_CELL_CLASS:
            continue
        total += len(ys) * weighted_auroc(ys, scores[index], np.ones(len(ys)))
        weight += len(ys)
        used.add(key[:2])
    return (total / weight if weight else float("nan")), len(used)


def run(
    frame: pd.DataFrame, unit_dir: Path, log: Callable[[str], None] = print
) -> dict[str, object]:
    """`frame` has task, scaffold, model, failed, closing, tail. Returns the summary block."""
    frame = frame.reset_index(drop=True)
    inputs = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    unit_dir = unit_dir / hashlib.sha256(inputs).hexdigest()[:16]
    unit_dir.mkdir(parents=True, exist_ok=True)
    y = frame["failed"].to_numpy()
    tasks = frame["task"].to_numpy()
    folds = np.zeros(len(frame), dtype=int)
    for fold, (_, test) in enumerate(GroupKFold(FOLDS).split(frame, y, tasks)):
        folds[test] = fold
    results: dict[str, object] = {
        "n": len(frame),
        "n_false_success": int(y.sum()),
        "tasks": int(frame["task"].nunique()),
    }
    for feature in ("ids", "closing", "tail", "closing+tail"):
        scores = out_of_fold(frame, feature, folds, unit_dir)
        draws = auroc_draws(y, scores, tasks, REPS, SEED)
        low, high = percentile(draws)
        within, cells = _within_cells(frame, scores, folds)
        results[feature] = {
            "auroc": weighted_auroc(y, scores, np.ones(len(y))),
            "ci": [float(low), float(high)],
            "auroc_within_cell": within,
            "within_cells": cells,
        }
        log(f"detector {feature}: {results[feature]}")
    # Leave one scaffold out. Holding out tasks as well matters: the same tasks in training lets
    # the detector recognise hard tasks by their vocabulary, a leak task-disjoint folds prevent.
    held: dict[str, dict[str, float]] = {}
    codes = pd.factorize(frame["scaffold"], sort=True)[0]
    names = sorted(frame["scaffold"].unique())
    for code, name in enumerate(names):
        mask = codes == code
        ys = y[mask]
        if len(names) < 2 or min(ys.sum(), len(ys) - ys.sum()) < MIN_CLASS:
            continue
        leaky = _fold_scores(frame, "closing", f"loso-{name}", mask, unit_dir)
        clean = np.zeros(len(frame))
        for fold in np.unique(folds):
            test = mask & (folds == fold)
            train = ~mask & (folds != fold)
            clean[test] = _pair_scores(frame, f"lost-{name}-{fold}", train, test, unit_dir)
        held[name] = {
            "tasks_shared": weighted_auroc(ys, leaky, np.ones(len(ys))),
            "tasks_held_out": _pooled_folds(ys, clean[mask], folds[mask]),
        }
    results["leave_one_scaffold_out"] = held
    return results
