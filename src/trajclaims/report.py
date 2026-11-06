"""results/v1/tables.md: the tables REPORT.md quotes, generated, never edited by hand."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{100 * value:.1f}"


def _coverage_rows(table: pd.DataFrame) -> list[str]:
    columns = list(table.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in table.itertuples(index=False):
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _rate_rows(table: pd.DataFrame) -> list[str]:
    head = (
        "| group | failed known | lost | FCR % (95% CI) | if lost silent / claim | bridged "
        "| FCR readable % | corrected % (95% CI) | honest % | claim given pass % "
        "| discrimination | false-claim share % |"
    )
    lines = [head, "|" + "---|" * 12]
    for r in table.itertuples():
        corrected = "n/a"
        if "fcr_readable_corrected" in table and not pd.isna(r.fcr_readable_corrected):
            corrected = (
                f"{_pct(r.fcr_readable_corrected)} ({_pct(r.fcr_readable_corrected_low)} to "
                f"{_pct(r.fcr_readable_corrected_high)})"
            )
        lines.append(
            f"| {r.group} | {int(r.failed_known)} | {int(r.failed_lost)} "
            f"| {_pct(r.fcr)} ({_pct(r.fcr_low)} to {_pct(r.fcr_high)}) "
            f"| {_pct(r.fcr_if_lost_silent)} / {_pct(r.fcr_if_lost_claim)} "
            f"| {_pct(r.fcr_bridged)} | {_pct(r.fcr_readable)} | {corrected} | {_pct(r.honest)} "
            f"| {_pct(r.claim_rate_passed_readable)} "
            f"| {_pct(r.discrimination_readable)} | {_pct(r.false_claim_share)} |"
        )
    return lines


def write_tables(out: Path) -> None:
    fcr = pd.read_csv(out / "fcr.csv")
    coverage = pd.read_csv(out / "coverage.csv")
    lines = ["# Generated tables", "", "Written by `trajclaims study`. Do not edit.", ""]
    lines += ["## Coverage", "", *_coverage_rows(coverage), ""]
    for level, title in (("scaffold", "Per scaffold"), ("model", "Per model"), ("cell", "Cells")):
        table = fcr[(fcr["level"] == level) & fcr["reportable"]]
        lines += [f"## {title}", "", *_rate_rows(table), ""]
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
