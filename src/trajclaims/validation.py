"""The validation sample: drawn after the rules were frozen, from trials they never saw."""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from pathlib import Path
from typing import Any

import pandas as pd

from trajclaims.rules import LABELS

SEED = 20261106
PER_FAILED = 60
PER_PASSED = 20
KEY = ["task", "scaffold", "model_raw", "trial"]
TOKEN = re.compile(
    r"AKIA[0-9A-Z]{12,}|gh[pousr]_[A-Za-z0-9]{10,}|hf_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9-]{16,}"
)


def is_dev(frame: pd.DataFrame) -> pd.Series:
    """One trial in five, by key hash: the only trials read while writing the rules."""
    keys = zip(*(frame[c] for c in KEY), strict=True)
    return pd.Series(
        [zlib.crc32("|".join(key).encode("utf-8")) % 5 == 0 for key in keys], index=frame.index
    )


def item_id(row: Any) -> str:
    key = "|".join(str(getattr(row, c)) for c in KEY)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def draw(labelled: pd.DataFrame) -> pd.DataFrame:
    """Random messages per scaffold x reward stratum, with each stratum's population size.

    The population is closing statements a rule can read, outside the development trials.
    """
    population = labelled[(labelled["status"] == "readable") & ~labelled["dev"]]
    parts = []
    for (scaffold, reward), group in population.groupby(["scaffold", "reward"], sort=True):
        size = PER_FAILED if reward == 0 else PER_PASSED
        taken = group.sample(min(size, len(group)), random_state=SEED).copy()
        taken["stratum_n"] = len(group)
        taken["stratum"] = f"{scaffold}|{reward}"
        parts.append(taken)
    sample = pd.concat(parts, ignore_index=True)
    sample["id"] = [item_id(row) for row in sample.itertuples()]
    return sample


def redact(text: str) -> str:
    """Hide token-shaped strings. Tasks plant fake credentials, and closings can quote them."""
    return TOKEN.sub("[redacted-token]", text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def disagreements(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, str]:
    """Item id -> both labels, sorted so the hint does not reveal which labeller gave which."""
    left = {row["id"]: row["label"] for row in a}
    right = {row["id"]: row["label"] for row in b}
    if set(left) != set(right):
        raise ValueError("the two labellers did not label the same items")
    return {i: " / ".join(sorted((left[i], right[i]))) for i in left if left[i] != right[i]}


def _pr(truth: pd.Series, pred: pd.Series, weight: pd.Series) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in LABELS:
        hit = float(weight[(truth == name) & (pred == name)].sum())
        predicted = float(weight[pred == name].sum())
        actual = float(weight[truth == name].sum())
        out[name] = {
            "precision": hit / predicted if predicted else float("nan"),
            "recall": hit / actual if actual else float("nan"),
        }
    return out


def rule_accuracy(sample: pd.DataFrame) -> dict[str, Any]:
    """Per-class precision and recall of the rules, on the sample and on the population.

    Strata were sampled at different rates, so population figures weight each message by its
    stratum's size over its sample count.
    """
    weight = sample["stratum_n"] / sample.groupby("stratum")["stratum"].transform("size")
    truth, pred = sample["adjudicated"], sample["rule_label"]
    return {
        "sample": _pr(truth, pred, pd.Series(1.0, index=sample.index)),
        "population": _pr(truth, pred, weight),
        "accuracy_population": float(weight[truth == pred].sum() / weight.sum()),
    }


def error_rates(sample: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per scaffold, among failed trials: counts behind the sensitivity and specificity of
    `claims-done`, against adjudicated labels. Within a stratum the sample is random, so these
    need no weights."""
    failed = sample[sample["reward"] == 0]
    out: dict[str, dict[str, int]] = {}
    for scaffold, group in failed.groupby("scaffold", sort=True):
        truth = group["adjudicated"] == "claims-done"
        pred = group["rule_label"] == "claims-done"
        out[str(scaffold)] = {
            "true_pos": int((truth & pred).sum()),
            "pos": int(truth.sum()),
            "true_neg": int((~truth & ~pred).sum()),
            "neg": int((~truth).sum()),
        }
    return out
