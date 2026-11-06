"""The study: labels, rates with task-clustered intervals, and the written outputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trajclaims import detector, models_effects, stats, validation
from trajclaims.data import DATASET, REVISION, SHARDS
from trajclaims.rules import LABELS, RULES_VERSION, label
from trajclaims.validation import is_dev, read_jsonl

SEED = 20261106
REPS = 10000
MIN_CELL = 40
MIN_RATE_N = 5
CLAIM, HEDGE, FAILURE, SILENT = LABELS
RULES_FILE = Path(__file__).resolve().parent / "rules.py"
LABEL_COLUMNS = [
    "task",
    "scaffold",
    "model",
    "model_raw",
    "trial",
    "reward",
    "ending",
    "status",
    "signal",
    "label",
    "rule",
    "verification",
    "verified_after_edit",
    "timed_out",
]


def rules_sha() -> str:
    return hashlib.sha256(RULES_FILE.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def assign_labels(trials: pd.DataFrame) -> pd.DataFrame:
    """Trials with a trajectory, labelled. `lost` closing statements stay unlabelled.

    A cut-off or an empty closing says nothing, so it is `no-statement` by construction.
    """
    frame = trials[trials["n_steps"] > 0].copy()
    labels, rules = [], []
    for status, text in zip(frame["status"], frame["closing_text"], strict=True):
        if status == "readable":
            name, rule = label(text)
        elif status == "lost":
            name, rule = "", "lost"
        else:
            name, rule = SILENT, status
        labels.append(name)
        rules.append(rule)
    frame["label"], frame["rule"] = labels, rules
    frame["dev"] = is_dev(frame)
    return frame


def labels_jsonl(frame: pd.DataFrame) -> bytes:
    """Labels as canonical bytes; the regeneration oracle compares these."""
    lines = []
    for row in frame[LABEL_COLUMNS].to_dict("records"):
        row["verified_after_edit"] = bool(row["verified_after_edit"])
        row["timed_out"] = bool(row["timed_out"])
        lines.append(json.dumps(row, sort_keys=True, ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def coverage(trials: pd.DataFrame) -> pd.DataFrame:
    """Per scaffold: how many trials exist, have steps, and have a usable closing statement."""
    rows = []
    for scaffold, group in trials.groupby("scaffold", sort=True):
        failed = group[group["reward"] == 0]
        passed = group[group["reward"] == 1]
        steps = group[group["n_steps"] > 0]
        counts = steps["status"].value_counts()
        rows.append(
            {
                "scaffold": scaffold,
                "trials": len(group),
                "failed": len(failed),
                "with_steps": len(steps),
                "steps_rate_failed": float((failed["n_steps"] > 0).mean()) if len(failed) else 0.0,
                "steps_rate_passed": float((passed["n_steps"] > 0).mean()) if len(passed) else 0.0,
                **{s: int(counts.get(s, 0)) for s in ("readable", "lost", "no-text", "cutoff")},
            }
        )
    return pd.DataFrame(rows)


def task_matrix(
    frame: pd.DataFrame, mask: np.ndarray, tasks: pd.Index, group: pd.Series, groups: pd.Index
) -> np.ndarray:
    """Tasks x groups counts of the rows in `mask`."""
    out = np.zeros((len(tasks), len(groups)))
    t = tasks.get_indexer(frame.loc[mask, "task"])
    g = groups.get_indexer(group[mask])
    np.add.at(out, (t, g), 1.0)
    return out


def claim_rates(errors: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    """Sensitivity and specificity counts per scaffold, pooled where a scaffold has too few."""
    pooled = {k: sum(e[k] for e in errors.values()) for k in ("true_pos", "pos", "true_neg", "neg")}
    out: dict[str, dict[str, Any]] = {"*": {**pooled, "source": "pooled"}}
    for scaffold, counts in errors.items():
        enough = min(counts["pos"], counts["neg"]) >= MIN_RATE_N
        out[scaffold] = (
            {**counts, "source": "scaffold"} if enough else {**pooled, "source": "pooled"}
        )
    return out


def _corrected_cells(
    frame: pd.DataFrame, rates: dict[str, dict[str, Any]], weights: np.ndarray, seed: int
) -> dict[str, Any]:
    """Rogan-Gladen claim counts per scaffold x model cell, among readable failed closings.

    The validation sample covers readable closings only, so the correction applies there.
    Sensitivity and specificity are scaffold-level; draws carry their sampling noise.
    """
    readable = frame[(frame["reward"] == 0) & (frame["status"] == "readable")]
    cell = readable["scaffold"] + "|" + readable["model"]
    cells = pd.Index(sorted(cell.unique()))
    tasks = pd.Index(sorted(frame["task"].unique()))
    num = task_matrix(readable, (readable["label"] == CLAIM).to_numpy(), tasks, cell, cells)
    den = task_matrix(readable, np.ones(len(readable), dtype=bool), tasks, cell, cells)
    scaffolds = [c.split("|")[0] for c in cells]
    rng = np.random.default_rng(seed)
    reps = weights.shape[0]
    se, sp, se_point, sp_point = {}, {}, {}, {}
    for name in sorted(set(scaffolds)):
        counts = rates.get(name, rates["*"])
        se[name] = stats.beta_draws(counts["true_pos"], counts["pos"], reps, rng)
        sp[name] = stats.beta_draws(counts["true_neg"], counts["neg"], reps, rng)
        se_point[name] = counts["true_pos"] / counts["pos"]
        sp_point[name] = counts["true_neg"] / counts["neg"]
    se_draws = np.column_stack([se[s] for s in scaffolds])
    sp_draws = np.column_stack([sp[s] for s in scaffolds])
    seen_draws = weights @ den
    corrected = stats.rogan_gladen(stats.ratio_draws(num, den, weights), se_draws, sp_draws)
    point = stats.rogan_gladen(
        num.sum(0) / den.sum(0),
        np.array([se_point[s] for s in scaffolds]),
        np.array([sp_point[s] for s in scaffolds]),
    )
    return {
        "cells": cells,
        "seen": den.sum(0),
        "seen_draws": seen_draws,
        "count": np.clip(point, 0.0, 1.0) * den.sum(0),
        "count_draws": np.clip(corrected, 0.0, 1.0) * seen_draws,
    }


def rate_table(
    frame: pd.DataFrame,
    level: str,
    weights: np.ndarray,
    rates: dict[str, dict[str, Any]] | None,
    bridge_rate: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Rates per group at one level, plus the FCR draws for contrasts."""
    group = frame["scaffold"] + "|" + frame["model"] if level == "cell" else frame[level]
    groups = pd.Index(sorted(group.unique()))
    tasks = pd.Index(sorted(frame["task"].unique()))
    is_failed = (frame["reward"] == 0).to_numpy()
    is_lost = (frame["status"] == "lost").to_numpy()
    is_read = (frame["status"] == "readable").to_numpy()
    lab = frame["label"].to_numpy()

    def count(mask: np.ndarray) -> np.ndarray:
        return task_matrix(frame, mask, tasks, group, groups)

    claims = count(is_failed & (lab == CLAIM))
    hedged = count(is_failed & (lab == HEDGE))
    fails = count(is_failed & (lab == FAILURE))
    silent = count(is_failed & (lab == SILENT))
    lost = count(is_failed & is_lost)
    read = count(is_failed & is_read)
    known = claims + hedged + fails + silent
    fcr_draws = stats.ratio_draws(claims, known, weights)
    fcr_low, fcr_high = stats.percentile(fcr_draws)
    read_low, read_high = stats.percentile(stats.ratio_draws(claims, read, weights))
    honest_low, honest_high = stats.percentile(stats.ratio_draws(fails, known, weights))
    k, n, l_, r = claims.sum(0), known.sum(0), lost.sum(0), read.sum(0)
    declared = count(is_failed & (frame["ending"] == "declared").to_numpy()).sum(0)
    pass_claims = count(~is_failed & (lab == CLAIM)).sum(0)
    pass_read = count(~is_failed & is_read).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        table = pd.DataFrame(
            {
                "level": level,
                "group": groups,
                "failed_with_steps": n + l_,
                "failed_known": n,
                "failed_readable": r,
                "failed_lost": l_,
                "claims_done": k,
                "hedged": hedged.sum(0),
                "reports_failure": fails.sum(0),
                "no_statement": silent.sum(0),
                "fcr": k / n,
                "fcr_low": fcr_low,
                "fcr_high": fcr_high,
                "fcr_if_lost_silent": k / (n + l_),
                "fcr_if_lost_claim": (k + l_) / (n + l_),
                "fcr_bridged": (k + bridge_rate * l_) / (n + l_),
                "fcr_readable": k / r,
                "fcr_readable_low": read_low,
                "fcr_readable_high": read_high,
                "honest": fails.sum(0) / n,
                "honest_low": honest_low,
                "honest_high": honest_high,
                "declared_rate_failed": declared / (n + l_),
                "claim_rate_passed_readable": pass_claims / pass_read,
                "discrimination_readable": pass_claims / pass_read - k / r,
                "false_claim_share": k / (k + pass_claims),
            }
        )
    table["reportable"] = table["failed_known"] >= MIN_CELL
    if rates is not None:
        _add_corrected(table, frame, level, rates, weights)
    return table, fcr_draws


def _add_corrected(
    table: pd.DataFrame,
    frame: pd.DataFrame,
    level: str,
    rates: dict[str, dict[str, Any]],
    weights: np.ndarray,
) -> None:
    """Corrected FCR over readable closings, aggregated from scaffold x model cells."""
    block = _corrected_cells(frame, rates, weights, SEED + 1)
    cells = block["cells"]
    key = {
        "scaffold": [c.split("|")[0] for c in cells],
        "model": [c.split("|")[1] for c in cells],
        "cell": list(cells),
    }[level]
    rows = np.zeros((len(cells), len(table)))
    rows[np.arange(len(cells)), pd.Index(table["group"]).get_indexer(key)] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = (rows.T @ block["count"]) / (rows.T @ block["seen"])
        draws = (block["count_draws"] @ rows) / (block["seen_draws"] @ rows)
    low, high = stats.percentile(draws)
    table["fcr_readable_corrected"] = corrected
    table["fcr_readable_corrected_low"] = low
    table["fcr_readable_corrected_high"] = high
    table["fcr_corrected"] = corrected * table["failed_readable"] / table["failed_known"]


def bridge(labelled: pd.DataFrame) -> dict[str, Any]:
    """What the partial scrape loses, from trials that both scrapes list."""
    twins = labelled[labelled["partial_status"].notna()]
    rows: list[dict[str, Any]] = []
    for (scaffold, reward, status), group in twins.groupby(
        ["scaffold", "reward", "partial_status"], sort=True
    ):
        readable = group[group["status"] == "readable"]
        rows.append(
            {
                "scaffold": scaffold,
                "reward": int(reward),
                "partial_status": status,
                "n": len(group),
                "readable_in_full": len(readable),
                "claims_done": int((readable["label"] == CLAIM).sum()),
            }
        )

    def failed_readable(status: str) -> pd.DataFrame:
        chosen = (twins["partial_status"] == status) & (twins["reward"] == 0)
        return twins[chosen & (twins["status"] == "readable")]

    lost, kept = failed_readable("lost"), failed_readable("readable")
    lost_claims = int((lost["label"] == CLAIM).sum())
    kept_claims = int((kept["label"] == CLAIM).sum())
    paired = twins[(twins["partial_status"] == "readable") & (twins["status"] == "readable")]
    relabelled = [label(text)[0] for text in paired["partial_text"]]
    agreement = float(np.mean(np.array(relabelled) == paired["label"].to_numpy()))
    return {
        "pairs": len(twins),
        "rows": rows,
        "failed_lost_readable_in_full": len(lost),
        "failed_lost_claims_done": lost_claims,
        "claim_rate_lost": lost_claims / len(lost) if len(lost) else 0.0,
        "claim_rate_lost_ci": list(stats.wilson(lost_claims, len(lost))),
        "failed_kept": len(kept),
        "failed_kept_claims_done": kept_claims,
        "claim_rate_kept": kept_claims / len(kept) if len(kept) else 0.0,
        "paired_text_pairs": len(paired),
        "paired_label_agreement": agreement,
    }


def sensitivity(labelled: pd.DataFrame, trials: pd.DataFrame) -> dict[str, Any]:
    """FCR per scaffold on alternative populations, and the template stratum."""
    means = trials.groupby("task")["reward"].mean()
    never = sorted(means[means == 0].index)
    rare = sorted(means[means < 0.02].index)
    known = labelled[(labelled["reward"] == 0) & (labelled["status"] != "lost")]

    def by(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
        out = {}
        for scaffold, group in frame.groupby("scaffold", sort=True):
            out[str(scaffold)] = {
                "n": len(group),
                "fcr": float((group["label"] == CLAIM).mean()),
            }
        return out

    texts = labelled[labelled["status"] == "readable"]
    repeats = texts.groupby(["scaffold", "closing_text"])["closing_text"].transform("size")
    template = repeats >= 10
    phrase = texts["closing_text"].str.contains(r"(?i)task (?:is |has been )?(?:now )?complete")
    return {
        "never_solved_tasks": never,
        "rarely_solved_tasks": rare,
        "excluding_rarely_solved": by(known[~known["task"].isin(rare)]),
        "excluding_cutoffs": by(known[known["ending"] != "cutoff"]),
        "excluding_templates": by(known[~known.index.isin(texts.index[template])]),
        "template_share_of_readable": {
            str(k): float(v) for k, v in template.groupby(texts["scaffold"]).mean().items()
        },
        "task_complete_phrase_share": {
            str(k): float(v) for k, v in phrase.groupby(texts["scaffold"]).mean().items()
        },
        "ending_by_scaffold_failed": {
            str(k): {str(e): int(n) for e, n in v.value_counts().sort_index().items()}
            for k, v in known.groupby("scaffold", sort=True)["ending"]
        },
    }


def validation_summary(rows: list[dict[str, Any]], current_sha: str) -> dict[str, Any]:
    """Agreement and rule accuracy from results/v1/validation.jsonl."""
    sample = pd.DataFrame(rows)
    if set(sample["rules_sha"]) != {current_sha}:
        raise ValueError("rules.py changed after the validation sample was drawn: draw a new one")
    a, b = sample["label_a"].tolist(), sample["label_b"].tolist()
    rng = np.random.default_rng(SEED + 3)
    n = len(sample)
    draws = []
    for _ in range(2000):
        pick = rng.integers(0, n, n)
        draws.append(stats.cohen_kappa([a[i] for i in pick], [b[i] for i in pick]))
    rule_claim = (sample["rule_label"] == CLAIM).astype(str).tolist()
    truth_claim = (sample["adjudicated"] == CLAIM).astype(str).tolist()
    return {
        "n": n,
        "strata": {k: int(v) for k, v in sample["stratum"].value_counts().sort_index().items()},
        "labeller_agreement": float(np.mean(sample["label_a"] == sample["label_b"])),
        "kappa_labellers": stats.cohen_kappa(a, b),
        "kappa_labellers_ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "kappa_rules_vs_adjudicated": stats.cohen_kappa(
            sample["rule_label"].tolist(), sample["adjudicated"].tolist()
        ),
        "kappa_rules_vs_adjudicated_claim_binary": stats.cohen_kappa(rule_claim, truth_claim),
        "disagreements": int((sample["label_a"] != sample["label_b"]).sum()),
        "adjudicated_counts": {
            k: int(v) for k, v in sample["adjudicated"].value_counts().sort_index().items()
        },
        "rule_accuracy": validation.rule_accuracy(sample),
        "claim_errors_failed": validation.error_rates(sample),
        "direct_fcr_readable": _direct(sample),
        "labeller_models": sorted({m for r in rows for m in r["models"]}),
        "prompt_shas": sorted({s for r in rows for s in r["prompt_shas"]}),
    }


def _direct(sample: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Design-based FCR: adjudicated labels only, so free of rule error but noisier."""
    out: dict[str, dict[str, Any]] = {}
    failed = sample[sample["reward"] == 0]
    for scaffold, group in failed.groupby("scaffold", sort=True):
        hits = int((group["adjudicated"] == CLAIM).sum())
        out[str(scaffold)] = {
            "n": len(group),
            "fcr": hits / len(group),
            "ci": list(stats.wilson(hits, len(group))),
        }
    return out


def clean(value: Any) -> Any:
    """JSON-safe, rounded, with no NaN, so reruns are byte-identical."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if isinstance(value, str | bool | np.bool_) or value is None:
        return value if not isinstance(value, np.bool_) else bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    number = float(value)
    return None if not np.isfinite(number) else round(number, 6)


def _rates_and_contrasts(
    labelled: pd.DataFrame,
    rates: dict[str, dict[str, Any]] | None,
    bridge_rate: float,
    out: Path,
    summary: dict[str, Any],
) -> None:
    weights = stats.task_weights(labelled["task"].nunique(), REPS, SEED)
    tables = []
    for level in ("scaffold", "model", "cell"):
        table, draws = rate_table(labelled, level, weights, rates, bridge_rate)
        tables.append(table)
        if level != "cell":
            continue
        keep = table["reportable"].to_numpy()
        pairs = stats.contrasts(table.loc[keep, "fcr"].to_numpy(), draws[:, keep])
        names = table.loc[keep, "group"].to_numpy()
        pairs["left"] = names[pairs["left"]]
        pairs["right"] = names[pairs["right"]]
        pairs.round(6).to_csv(out / "contrasts.csv", index=False, lineterminator="\n")
        summary["contrasts"] = {
            "cells": int(keep.sum()),
            "pairs": len(pairs),
            "holm_significant_05": int((pairs["p_holm"] < 0.05).sum()),
        }
    fcr = pd.concat(tables, ignore_index=True)
    fcr.round(6).to_csv(out / "fcr.csv", index=False, lineterminator="\n")


def run_study(
    trials: pd.DataFrame,
    counts: dict[str, int],
    out: Path,
    unit_dir: Path,
    with_validation: bool = True,
    with_detector: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Regenerate every output under `out` from extracted trials and validation.jsonl."""
    out.mkdir(parents=True, exist_ok=True)
    labelled = assign_labels(trials)
    payload = labels_jsonl(labelled)
    (out / "labels.jsonl").write_bytes(payload)
    log(f"labels: {len(labelled)} trials")

    twins = bridge(labelled)
    summary: dict[str, Any] = {
        "dataset": {
            "name": DATASET,
            "revision": REVISION,
            "shards_sha256": SHARDS,
            **counts,
            "scaffolds_listed": int(trials["scaffold"].nunique()),
            "scaffolds_with_steps": int(labelled["scaffold"].nunique()),
            "models_raw_listed": int(trials["model_raw"].nunique()),
            "models_listed": int(trials["model"].nunique()),
            "models_with_steps": int(labelled["model"].nunique()),
            "tasks": int(trials["task"].nunique()),
            "trials_with_steps": len(labelled),
            "failed_with_steps": int((labelled["reward"] == 0).sum()),
        },
        "rules": {"version": RULES_VERSION, "sha256": rules_sha()},
        "labels_sha256": hashlib.sha256(payload).hexdigest(),
        "bridge": twins,
        "bootstrap": {"reps": REPS, "seed": SEED, "cluster": "task"},
        "min_cell_failed": MIN_CELL,
    }
    coverage(trials).to_csv(out / "coverage.csv", index=False, lineterminator="\n")

    rates = None
    validation_path = out / "validation.jsonl"
    if with_validation and validation_path.exists():
        block = validation_summary(read_jsonl(validation_path), rules_sha())
        rates = claim_rates(block["claim_errors_failed"])
        block["rates_used"] = rates
        summary["validation"] = block

    _rates_and_contrasts(labelled, rates, twins["claim_rate_lost"], out, summary)
    log("rates and contrasts done")
    summary["verification"] = models_effects.verification_tables(labelled)
    log("verification done")
    summary["model_vs_scaffold"] = models_effects.model_vs_scaffold(labelled, MIN_CELL, SEED + 2)
    log("decomposition done")
    summary["sensitivity"] = sensitivity(labelled, trials)
    if with_detector:
        claimed = labelled[labelled["label"] == CLAIM]
        frame = pd.DataFrame(
            {
                "task": claimed["task"],
                "scaffold": claimed["scaffold"],
                "model": claimed["model"],
                "failed": (claimed["reward"] == 0).astype(int),
                "closing": claimed["closing_text"],
                "tail": claimed["tail"],
            }
        )
        summary["detector"] = detector.run(frame, unit_dir, log)
    cleaned = clean(summary)
    text = json.dumps(cleaned, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    (out / "summary.json").write_text(text, encoding="utf-8", newline="\n")
    return cleaned
