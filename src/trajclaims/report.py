"""results/v1/tables.md: tables of the study, generated, never edited by hand."""

from __future__ import annotations

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
