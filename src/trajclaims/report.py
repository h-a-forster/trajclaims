"""results/v1/tables.md and numbers.json: generated from committed results, never edited."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CELL_NOTE = (
    "A cell inherits its scaffold's error rates, so its Rogan-Gladen figure is unstable and is "
    "clipped to 0 to 100 only for display; 100.0 means the correction hit the boundary. Prefer "
    "the calibrated column here."
)


def _pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{100 * value:.1f}"


def _coverage_rows(table: pd.DataFrame) -> list[str]:
    columns = list(table.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in table.itertuples(index=False):
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _span(value: float, low: float, high: float) -> str:
    return f"{_pct(value)} ({_pct(low)} to {_pct(high)})"


def _rate_rows(table: pd.DataFrame) -> list[str]:
    head = (
        "| group | failed readable | FCR readable % (95% CI) | calibrated % (95% CI) "
        "| Rogan-Gladen % (95% CI) | claim given pass % | discrimination | false-claim share % "
        "| honest of readable % | failed known | lost | FCR % (95% CI) | if lost silent / claim "
        "| bridged |"
    )
    lines = [head, "|" + "---|" * 14]
    calibrated = "fcr_readable_calibrated" in table
    for r in table.itertuples():
        adjusted, corrected = "n/a", "n/a"
        if calibrated and not pd.isna(r.fcr_readable_calibrated):
            adjusted = _span(
                r.fcr_readable_calibrated,
                r.fcr_readable_calibrated_low,
                r.fcr_readable_calibrated_high,
            )
            corrected = _span(
                r.fcr_readable_corrected,
                r.fcr_readable_corrected_low,
                r.fcr_readable_corrected_high,
            )
        lines.append(
            f"| {r.group} | {int(r.failed_readable)} "
            f"| {_span(r.fcr_readable, r.fcr_readable_low, r.fcr_readable_high)} "
            f"| {adjusted} | {corrected} | {_pct(r.claim_rate_passed_readable)} "
            f"| {_pct(r.discrimination_readable)} | {_pct(r.false_claim_share)} "
            f"| {_pct(r.honest_readable)} | {int(r.failed_known)} | {int(r.failed_lost)} "
            f"| {_span(r.fcr, r.fcr_low, r.fcr_high)} "
            f"| {_pct(r.fcr_if_lost_silent)} / {_pct(r.fcr_if_lost_claim)} "
            f"| {_pct(r.fcr_bridged)} |"
        )
    return lines


def write_tables(out: Path) -> None:
    fcr = pd.read_csv(out / "fcr.csv")
    coverage = pd.read_csv(out / "coverage.csv")
    lines = ["# Generated tables", "", "Written by `trajclaims study`. Do not edit.", ""]
    lines += ["## Coverage", "", *_coverage_rows(coverage), ""]
    levels = (
        ("all", "All scaffolds"),
        ("scaffold", "Per scaffold"),
        ("model", "Per model"),
        ("cell", "Cells"),
    )
    for level, title in levels:
        table = fcr[(fcr["level"] == level) & fcr["reportable_readable"]]
        lines += [f"## {title}", "", *_rate_rows(table), ""]
        if level == "cell":
            lines += [CELL_NOTE, ""]
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _count(value: float) -> str:
    return f"{int(value):,}"


def _two(value: float) -> str:
    return f"{value:.2f}"


def _put(out: dict[str, str], key: str, value: float, low: float, high: float, fmt=_pct) -> None:
    out[key], out[f"{key}_lo"], out[f"{key}_hi"] = fmt(value), fmt(low), fmt(high)


def numbers(out: Path) -> dict[str, str]:
    """Every figure the README and REPORT quote, formatted as they quote it.

    Built only from committed files in `out`, so CI can rebuild it without the dataset.
    """
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    fcr = pd.read_csv(out / "fcr.csv")
    coverage = pd.read_csv(out / "coverage.csv")
    sample = [
        json.loads(line)
        for line in (out / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    n: dict[str, str] = {}

    ds = summary["dataset"]
    for key in (
        "rows",
        "duplicates",
        "trials",
        "trials_with_steps",
        "failed_with_steps",
        "scaffolds_listed",
        "scaffolds_with_steps",
        "models_raw_listed",
        "models_listed",
        "models_with_steps",
        "tasks",
    ):
        n[f"data.{key}"] = _count(ds[key])
    n["data.scaffolds_without_steps"] = _count(ds["scaffolds_listed"] - ds["scaffolds_with_steps"])
    n["bootstrap.reps"] = _count(summary["bootstrap"]["reps"])
    n["bootstrap.seed"] = str(summary["bootstrap"]["seed"])

    lost, readable = coverage["lost"].sum(), coverage["readable"].sum()
    n["coverage.lost_share"] = _pct(lost / (lost + readable))
    for row in coverage.to_dict("records"):
        if row["readable"] + row["lost"]:
            key = f"coverage.{row['scaffold']}"
            n[f"{key}.lost"], n[f"{key}.readable"] = _count(row["lost"]), _count(row["readable"])
            n[f"{key}.lost_share"] = _pct(row["lost"] / (row["lost"] + row["readable"]))
            n[f"{key}.no_text"] = _count(row["no-text"])

    for row in fcr[fcr["level"].isin(["all", "scaffold"])].itertuples(index=False):
        key = f"fcr.{row.group}"
        for column in ("failed_readable", "failed_known", "failed_lost", "claims_done"):
            n[f"{key}.{column}"] = _count(getattr(row, column))
            n[f"{key}.{column}_raw"] = str(int(getattr(row, column)))
        for column in (
            "fcr_readable",
            "fcr_readable_calibrated",
            "fcr_readable_corrected",
            "fcr",
        ):
            if not pd.isna(getattr(row, column)):
                _put(
                    n,
                    f"{key}.{column}",
                    getattr(row, column),
                    getattr(row, f"{column}_low"),
                    getattr(row, f"{column}_high"),
                )
        for column in (
            "claim_rate_passed_readable",
            "false_claim_share",
            "honest_readable",
            "fcr_if_lost_silent",
            "fcr_if_lost_claim",
            "fcr_bridged",
        ):
            n[f"{key}.{column}"] = _pct(getattr(row, column))
    large = ["claude-code", "codex", "gemini-cli", "mini-swe-agent", "openhands"]
    scaffolds = fcr[(fcr["level"] == "scaffold") & fcr["reportable_readable"]].set_index("group")
    calibrated = scaffolds.loc[large, "fcr_readable_calibrated"]
    raw = scaffolds.loc[large, "fcr_readable"]
    n["fcr.large.calibrated_min"], n["fcr.large.calibrated_max"] = (
        _pct(calibrated.min()),
        _pct(calibrated.max()),
    )
    n["fcr.large.raw_min"], n["fcr.large.raw_max"] = _pct(raw.min()), _pct(raw.max())
    honest = scaffolds["honest_readable"]
    n["fcr.scaffold.honest_min"], n["fcr.scaffold.honest_max"] = (
        _pct(honest.min()),
        _pct(honest.max()),
    )

    sens = summary["sensitivity"]
    comp = sens["composition_failed_readable"]
    for scaffold, share in comp["share"].items():
        n[f"pool.{scaffold}"] = _pct(share)
    n["pool.scaffolds_in_mean"] = str(comp["scaffolds_in_mean"])
    n["pool.unweighted_mean"] = _pct(comp["unweighted_mean_fcr"])
    for group, cell in sens["fcr_readable_by_closing_group"].items():
        n[f"closing.{group}.fcr"], n[f"closing.{group}.n"] = _pct(cell["fcr"]), _count(cell["n"])

    v = summary["validation"]
    n["val.n"], n["val.disagreements"] = _count(v["n"]), _count(v["disagreements"])
    _put(n, "val.kappa", v["kappa_labellers"], *v["kappa_labellers_ci"], fmt=_two)
    n["val.agreement"] = _pct(v["labeller_agreement"])
    n["val.kappa_rules"] = _two(v["kappa_rules_vs_adjudicated"])
    n["val.kappa_rules_claim"] = _two(v["kappa_rules_vs_adjudicated_claim_binary"])
    for label, count in v["adjudicated_counts"].items():
        n[f"val.adjudicated.{label}"] = _count(count)
    for scope in ("population", "population_failed"):
        for label, cell in v["rule_accuracy"][scope].items():
            n[f"val.{scope}.{label}.precision"] = _two(cell["precision"])
            n[f"val.{scope}.{label}.recall"] = _two(cell["recall"])
    n["val.reports_failure_recall_pct"] = (
        f"{100 * v['rule_accuracy']['population_failed']['reports-failure']['recall']:.0f}"
    )
    direct = v["direct_fcr_readable"]["all"]
    _put(n, "val.direct", direct["fcr"], *direct["ci"])
    _put(n, "val.honest", direct["honest"], *direct["honest_ci"])
    n["val.direct_n"] = _count(direct["n"])
    n["val.rule_rate_sample"] = _pct(direct["rule_rate_sample"])
    n["val.rule_rate_population"] = _pct(direct["rule_rate_population"])
    n["val.rule_rate_gap"] = _pct(direct["rule_rate_population"] - direct["rule_rate_sample"])
    for stratum, size in v["strata"].items():
        n[f"val.strata.{stratum}"] = _count(size)
    command = [r for r in sample if r["stratum"] == "terminus-2+command|0"]
    claimed = [r for r in command if r["rule_label"] == "claims-done"]
    n["val.command.rule_claims"] = _count(len(claimed))
    n["val.command.rule_claims_held"] = _count(
        sum(r["adjudicated"] == "claims-done" for r in claimed)
    )
    n["val.deepagent-harbor.negatives"] = str(v["claim_errors_failed"]["deepagent-harbor"]["neg"])

    for population, block in summary["model_vs_scaffold"].items():
        key = f"split.{population}"
        n[f"{key}.models"], n[f"{key}.cells"] = str(len(block["models"])), str(block["cells"])
        for kind, shares in (("raw", block["raw"]), ("adj", block["task_adjusted"])):
            for name in ("scaffold_share", "model_share", "scaffold_minus_model"):
                _put(n, f"{key}.{kind}.{name}", shares[name], *shares[f"{name}_ci"])
            n[f"{key}.replicates"] = _count(shares["replicates"])
        for spread in block["scaffold_spread_for_fixed_model"]:
            _put(n, f"{key}.spread_model.{spread['model']}", spread["range"], *spread["range_ci"])
        for spread in block["model_spread_for_fixed_scaffold"]:
            _put(
                n,
                f"{key}.spread_scaffold.{spread['scaffold']}",
                spread["range"],
                *spread["range_ci"],
            )
    contrasts = summary["contrasts"]["fcr_readable"]
    n["contrasts.pairs"], n["contrasts.cells"] = (
        _count(contrasts["pairs"]),
        _count(contrasts["cells"]),
    )
    n["contrasts.significant"] = _count(contrasts["holm_significant_05"])

    ver = summary["verification"]
    totals = ver["totals_failed"]
    n["ver.failed"], n["ver.none"] = _count(totals["failed"]), _count(totals["none"])
    n["ver.none_share"] = _pct(totals["none"] / totals["failed"])
    for level, cell in ver["claim_rate_by_level"].items():
        n[f"ver.{level}.claims"], n[f"ver.{level}.n"] = (
            _count(cell["claims_done"]),
            _count(cell["n"]),
        )
    for model in ("odds_unadjusted", "odds_scaffold_model", "odds_adjusted"):
        for term in ("adhoc_check", "test_run"):
            cell = ver[model][term]
            _put(n, f"ver.{model}.{term}", cell["odds_ratio"], *cell["ci"], fmt=_two)
    cell = ver["odds_after_edit"]["after_edit"]
    _put(n, "ver.after_edit", cell["odds_ratio"], *cell["ci"], fmt=_two)
    large_tasks = ver["large_tasks"]
    cell = large_tasks["odds_after_edit"]["after_edit"]
    _put(n, "ver.large.after_edit", cell["odds_ratio"], *cell["ci"], fmt=_two)
    n["ver.large.tasks"] = _count(large_tasks["tasks"])
    n["ver.large.min_trials"] = _count(large_tasks["min_trials"])
    n["ver.regression_n"] = _count(ver["regression_n"])
    n["ver.regression_tasks"] = _count(ver["regression_tasks"])

    det = summary["detector"]
    n["det.n"] = _count(det["n"])
    for features in ("closing", "tail", "closing+tail", "ids"):
        cell = det[features]
        _put(n, f"det.{features}", cell["auroc"], *cell["ci"], fmt=_two)
        n[f"det.{features}.within"] = _two(cell["auroc_within_cell"])
    loso = det["leave_one_scaffold_out"].values()
    for split in ("tasks_shared", "tasks_held_out"):
        values = [cell[split] for cell in loso]
        n[f"det.loso.{split}_min"], n[f"det.loso.{split}_max"] = (
            _two(min(values)),
            _two(max(values)),
        )

    twins = summary["bridge"]
    _put(n, "bridge.lost", twins["claim_rate_lost"], *twins["claim_rate_lost_ci"])
    n["bridge.kept"] = _pct(twins["claim_rate_kept"])
    n["bridge.agreement"] = _pct(twins["paired_label_agreement"])
    n["bridge.text_pairs"] = _count(twins["paired_text_pairs"])
    for row in twins["rows"]:
        if row["reward"] == 0 and row["partial_status"] in ("lost", "readable"):
            key = f"bridge.{row['scaffold']}.{row['partial_status']}"
            n[f"{key}.claims"], n[f"{key}.n"] = _count(row["claims_done"]), _count(row["n"])
            n[f"{key}.pct"] = f"{100 * row['claims_done'] / row['n']:.0f}"
    return dict(sorted(n.items()))


def write_numbers(out: Path) -> None:
    text = json.dumps(numbers(out), indent=1, ensure_ascii=False) + "\n"
    (out / "numbers.json").write_text(text, encoding="utf-8", newline="\n")
