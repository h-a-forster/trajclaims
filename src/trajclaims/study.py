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
    "with_command",
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
    frame["closing_group"] = validation.closing_group(frame)
    return frame


def labels_jsonl(frame: pd.DataFrame) -> bytes:
    """Labels as canonical bytes; the regeneration oracle compares these."""
    lines = []
    for row in frame[LABEL_COLUMNS].to_dict("records"):
        row["with_command"] = bool(row["with_command"])
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
    """Rogan-Gladen claim counts per closing group x model cell, among readable failed closings.

    The validation sample covers readable closings only, so the correction applies there.
    Sensitivity and specificity are group-level; draws carry their sampling noise. Counts stay
    unclipped: the estimator is linear in a group, so clipping a cell would bias the sum.
    """
    readable = frame[(frame["reward"] == 0) & (frame["status"] == "readable")]
    cell = readable["closing_group"] + "|" + readable["model"]
    cells = pd.Index(sorted(cell.unique()))
    tasks = pd.Index(sorted(frame["task"].unique()))
    num = task_matrix(readable, (readable["label"] == CLAIM).to_numpy(), tasks, cell, cells)
    den = task_matrix(readable, np.ones(len(readable), dtype=bool), tasks, cell, cells)
    scaffolds = [c.split("|")[0] for c in cells]
    rng = np.random.default_rng(seed)
    reps = weights.shape[0]
    se, sp, se_point, sp_point = {}, {}, {}, {}
    ppv, npv, ppv_point, npv_point = {}, {}, {}, {}
    for name in sorted(set(scaffolds)):
        counts = rates.get(name, rates["*"])
        se[name] = stats.beta_draws(counts["true_pos"], counts["pos"], reps, rng)
        sp[name] = stats.beta_draws(counts["true_neg"], counts["neg"], reps, rng)
        se_point[name] = counts["true_pos"] / counts["pos"]
        sp_point[name] = counts["true_neg"] / counts["neg"]
        false_pos, false_neg = (
            counts["neg"] - counts["true_neg"],
            counts["pos"] - counts["true_pos"],
        )
        said_yes, said_no = counts["true_pos"] + false_pos, counts["true_neg"] + false_neg
        ppv[name] = stats.beta_draws(counts["true_pos"], said_yes, reps, rng)
        npv[name] = stats.beta_draws(counts["true_neg"], said_no, reps, rng)
        ppv_point[name] = counts["true_pos"] / said_yes
        npv_point[name] = counts["true_neg"] / said_no
    se_draws = np.column_stack([se[s] for s in scaffolds])
    sp_draws = np.column_stack([sp[s] for s in scaffolds])
    seen_draws = weights @ den
    rate_draws = stats.ratio_draws(num, den, weights)
    corrected = stats.rogan_gladen(rate_draws, se_draws, sp_draws)
    calibrated = stats.calibrated(
        rate_draws,
        np.column_stack([ppv[s] for s in scaffolds]),
        np.column_stack([npv[s] for s in scaffolds]),
    )
    calibrated_point = stats.calibrated(
        num.sum(0) / den.sum(0),
        np.array([ppv_point[s] for s in scaffolds]),
        np.array([npv_point[s] for s in scaffolds]),
    )
    point = stats.rogan_gladen(
        num.sum(0) / den.sum(0),
        np.array([se_point[s] for s in scaffolds]),
        np.array([sp_point[s] for s in scaffolds]),
    )
    return {
        "cells": cells,
        "seen": den.sum(0),
        "seen_draws": seen_draws,
        "count": point * den.sum(0),
        "count_draws": corrected * seen_draws,
        "calibrated": calibrated_point * den.sum(0),
        "calibrated_draws": calibrated * seen_draws,
    }


def rate_table(
    frame: pd.DataFrame,
    level: str,
    weights: np.ndarray,
    rates: dict[str, dict[str, Any]] | None,
    bridge_rates: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Rates per group at one level, plus the draws of each rate that gets contrasts."""
    if level == "all":
        group = pd.Series("all", index=frame.index)
    elif level == "cell":
        group = frame["scaffold"] + "|" + frame["model"]
    else:
        group = frame[level]
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
    read_draws = stats.ratio_draws(claims, read, weights)
    fcr_low, fcr_high = stats.percentile(fcr_draws)
    read_low, read_high = stats.percentile(read_draws)
    honest_low, honest_high = stats.percentile(stats.ratio_draws(fails, known, weights))
    k, n, l_, r = claims.sum(0), known.sum(0), lost.sum(0), read.sum(0)
    lost_rows = is_failed & is_lost
    rate = frame.loc[lost_rows, "scaffold"].map(bridge_rates).fillna(bridge_rates["*"])
    expected = rate.groupby(group[lost_rows].to_numpy()).sum().reindex(groups, fill_value=0.0)
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
                "fcr_bridged": (k + expected.to_numpy()) / (n + l_),
                "fcr_readable": k / r,
                "fcr_readable_low": read_low,
                "fcr_readable_high": read_high,
                "honest": fails.sum(0) / n,
                "honest_low": honest_low,
                "honest_high": honest_high,
                "honest_readable": fails.sum(0) / r,
                "declared_rate_failed": declared / (n + l_),
                "claim_rate_passed_readable": pass_claims / pass_read,
                "discrimination_readable": pass_claims / pass_read - k / r,
                "false_claim_share": k / (k + pass_claims),
            }
        )
    table["reportable"] = table["failed_known"] >= MIN_CELL
    table["reportable_readable"] = table["failed_readable"] >= MIN_CELL
    if rates is not None:
        _add_corrected(table, frame, level, rates, weights)
    return table, {"fcr": fcr_draws, "fcr_readable": read_draws}


def _add_corrected(
    table: pd.DataFrame,
    frame: pd.DataFrame,
    level: str,
    rates: dict[str, dict[str, Any]],
    weights: np.ndarray,
) -> None:
    """Corrected FCR over readable closings, aggregated from closing group x model cells."""
    block = _corrected_cells(frame, rates, weights, SEED + 1)
    cells = [c.replace(validation.WITH_COMMAND, "") for c in block["cells"]]
    key = {
        "all": ["all"] * len(cells),
        "scaffold": [c.split("|")[0] for c in cells],
        "model": [c.split("|")[1] for c in cells],
        "cell": cells,
    }[level]
    rows = np.zeros((len(cells), len(table)))
    rows[np.arange(len(cells)), pd.Index(table["group"]).get_indexer(key)] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = np.clip((rows.T @ block["count"]) / (rows.T @ block["seen"]), 0.0, 1.0)
        draws = np.clip((block["count_draws"] @ rows) / (block["seen_draws"] @ rows), 0.0, 1.0)
    low, high = stats.percentile(draws)
    table["fcr_readable_corrected"] = corrected
    table["fcr_readable_corrected_low"] = low
    table["fcr_readable_corrected_high"] = high
    with np.errstate(divide="ignore", invalid="ignore"):
        table["fcr_readable_calibrated"] = (rows.T @ block["calibrated"]) / (rows.T @ block["seen"])
        draws = (block["calibrated_draws"] @ rows) / (block["seen_draws"] @ rows)
    low, high = stats.percentile(draws)
    table["fcr_readable_calibrated_low"] = low
    table["fcr_readable_calibrated_high"] = high
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
    same = np.array(relabelled, dtype=str) == paired["label"].to_numpy(dtype=str)
    agreement = float(same.mean()) if len(same) else 0.0
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
        "lost_claim_rate_used": _bridge_rates(rows),
    }


def _bridge_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Claim rate among failed messages the partial scrape lost, per bridged scaffold.

    Scaffolds outside the bridge get `*`, the rate over the bridged scaffolds whose closings are
    prose: mini-swe-agent's long closings are mostly code, unlike a long claude-code summary.
    """
    lost = [r for r in rows if r["partial_status"] == "lost" and r["reward"] == 0]
    rates = {r["scaffold"]: r["claims_done"] / r["readable_in_full"] for r in lost}
    prose = [r for r in lost if r["scaffold"] != "mini-swe-agent"]
    claims, readable = (
        sum(r["claims_done"] for r in prose),
        sum(r["readable_in_full"] for r in prose),
    )
    rates["*"] = claims / readable if readable else 0.0
    return rates


def sensitivity(labelled: pd.DataFrame, trials: pd.DataFrame) -> dict[str, Any]:
    """FCR per scaffold on alternative populations, and the template stratum."""
    means = trials.groupby("task")["reward"].mean()
    failed_read = labelled[(labelled["reward"] == 0) & (labelled["status"] == "readable")]
    sizes = failed_read.groupby("scaffold").size()
    main = sizes[sizes >= MIN_CELL]
    claimed = failed_read[failed_read["label"] == CLAIM].groupby("scaffold").size()
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
    has_steps = labelled[labelled["ending"] != "none"]
    ending_check = {
        str(ending): {
            "n": len(group),
            "fail_rate": float((group["reward"] == 0).mean()),
            "at_timeout": float(group["timed_out"].mean()),
        }
        for ending, group in has_steps.groupby("ending", sort=True)
    }
    return {
        "composition_failed_readable": {
            "share": {str(k): float(v / sizes.sum()) for k, v in sizes.items()},
            "unweighted_mean_fcr": float((claimed.reindex(main.index, fill_value=0) / main).mean()),
            "scaffolds_in_mean": len(main),
        },
        "fcr_readable_by_closing_group": {
            str(k): {"n": len(v), "fcr": float((v["label"] == CLAIM).mean())}
            for k, v in failed_read.groupby("closing_group", sort=True)
        },
        "ending_check": ending_check,
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


def validation_summary(
    rows: list[dict[str, Any]], current_sha: str, labelled: pd.DataFrame
) -> dict[str, Any]:
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
        "direct_fcr_readable": _direct(sample, labelled),
        "labeller_models": sorted({m for r in rows for m in r["models"]}),
        "prompt_shas": sorted({s for r in rows for s in r["prompt_shas"]}),
    }


def _direct(sample: pd.DataFrame, labelled: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Design-based rates: adjudicated labels only, so free of rule error but noisier.

    Each message counts for its stratum's size over its sample count. Wilson intervals use the
    Kish effective sample size. `rule_rate_*` show how far the sample's rule rate sits from its
    population's; the calibrated estimate removes that gap, this one inherits it.
    """
    failed = sample[sample["reward"] == 0].copy()
    failed["weight"] = failed["stratum_n"] / failed.groupby("stratum")["stratum"].transform("size")
    drawn_from = labelled[
        (labelled["reward"] == 0) & (labelled["status"] == "readable") & ~labelled["dev"]
    ]

    def estimate(group: pd.DataFrame, population: pd.DataFrame) -> dict[str, Any]:
        weight = group["weight"]
        effective = weight.sum() ** 2 / (weight**2).sum()

        def share(mask: pd.Series) -> float:
            return float(weight[mask].sum() / weight.sum())

        claim, honest = share(group["adjudicated"] == CLAIM), share(group["adjudicated"] == FAILURE)
        return {
            "n": len(group),
            "effective_n": float(effective),
            "fcr": claim,
            "ci": list(stats.wilson(claim * effective, effective)),
            "honest": honest,
            "honest_ci": list(stats.wilson(honest * effective, effective)),
            "rule_rate_sample": share(group["rule_label"] == CLAIM),
            "rule_rate_population": float((population["label"] == CLAIM).mean()),
        }

    out = {"all": estimate(failed, drawn_from)}
    for scaffold, group in failed.groupby("scaffold", sort=True):
        out[str(scaffold)] = estimate(group, drawn_from[drawn_from["scaffold"] == scaffold])
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
    return None if not np.isfinite(number) else round(number, 6) + 0.0


def _rates_and_contrasts(
    labelled: pd.DataFrame,
    rates: dict[str, dict[str, Any]] | None,
    bridge_rates: dict[str, float],
    out: Path,
    summary: dict[str, Any],
) -> None:
    weights = stats.task_weights(labelled["task"].nunique(), REPS, SEED)
    tables = []
    for level in ("all", "scaffold", "model", "cell"):
        table, draws = rate_table(labelled, level, weights, rates, bridge_rates)
        tables.append(table)
    frames = []
    counts: dict[str, Any] = {}
    for metric, flag in (("fcr", "reportable"), ("fcr_readable", "reportable_readable")):
        keep = table[flag].to_numpy()
        pairs = stats.contrasts(table.loc[keep, metric].to_numpy(), draws[metric][:, keep])
        names = table.loc[keep, "group"].to_numpy()
        pairs["left"], pairs["right"] = names[pairs["left"]], names[pairs["right"]]
        pairs.insert(0, "metric", metric)
        frames.append(pairs)
        counts[metric] = {
            "cells": int(keep.sum()),
            "pairs": len(pairs),
            "holm_significant_05": int((pairs["p_holm"] < 0.05).sum()),
        }
    summary["contrasts"] = counts
    all_pairs = pd.concat(frames, ignore_index=True)
    all_pairs.round(6).to_csv(out / "contrasts.csv", index=False, lineterminator="\n")
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
    labelled = assign_labels(trials)
    payload = labels_jsonl(labelled)
    block = None
    validation_path = out / "validation.jsonl"
    if with_validation and validation_path.exists():
        rows = read_jsonl(validation_path)
        block = validation_summary(rows, rules_sha(), labelled)
        validation.require_current(rows, validation.sample_rows(labelled, rules_sha()))
    out.mkdir(parents=True, exist_ok=True)
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
    if block is not None:
        rates = claim_rates(block["claim_errors_failed"])
        block["rates_used"] = rates
        summary["validation"] = block

    _rates_and_contrasts(labelled, rates, twins["lost_claim_rate_used"], out, summary)
    log("rates and contrasts done")
    summary["verification"] = models_effects.verification_tables(labelled)
    log("verification done")
    summary["model_vs_scaffold"] = {
        "readable": models_effects.model_vs_scaffold(labelled, MIN_CELL, SEED + 2, readable=True),
        "known": models_effects.model_vs_scaffold(labelled, MIN_CELL, SEED + 2, readable=False),
    }
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
