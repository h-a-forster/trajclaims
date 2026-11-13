"""Every number in README.md, REPORT.md and CITATION.cff is copied from results/v1/numbers.json.

`trajclaims summarize` rebuilds numbers.json (and tables.md) from the committed results; CI runs
it and fails on any diff. A claim below is a fragment of a document with `{key}` placeholders. A
number that is not a result (a version, a design constant, a figure quoted from elsewhere) is
listed as fixed, and the document says where it comes from.
"""

import json
from pathlib import Path

import pytest
from docnumbers import check

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = json.loads((ROOT / "results/v1/numbers.json").read_text(encoding="utf-8"))

SCAFFOLDS = [
    "claude-code",
    "codex",
    "deepagent-harbor",
    "gemini-cli",
    "mini-swe-agent",
    "openhands",
    "terminus-2",
]


def ci(key: str) -> str:
    return f"{{{key}}} ({{{key}_lo}} to {{{key}_hi}})"


HEADLINE = [
    "{fcr.all.fcr_readable}% claimed completion ({fcr.all.claims_done} of "
    "{fcr.all.failed_readable}; {fcr.all.fcr_readable_lo} to {fcr.all.fcr_readable_hi})",
    "by inverse calibration ({fcr.all.fcr_readable_calibrated_lo} to "
    "{fcr.all.fcr_readable_calibrated_hi})",
    f"Rogan-Gladen gives {ci('fcr.all.fcr_readable_corrected')}".replace(" (", "% ("),
    "labels alone give {val.direct}% ({val.direct_lo} to {val.direct_hi})",
    "done on {fcr.all.claim_rate_passed_readable}% of passing runs against "
    "{fcr.all.fcr_readable}% of",
    "{fcr.all.false_claim_share}% of readable done-claims were on failed runs",
    "{split.readable.raw.scaffold_share}% of the variance of cell FCR "
    "({split.readable.raw.scaffold_share_lo} to {split.readable.raw.scaffold_share_hi}) and the "
    "model {split.readable.raw.model_share}% ({split.readable.raw.model_share_lo} to "
    "{split.readable.raw.model_share_hi})",
    "{split.readable.raw.scaffold_minus_model} points "
    "({split.readable.raw.scaffold_minus_model_lo} to "
    "{split.readable.raw.scaffold_minus_model_hi})",
    "{fcr.terminus-2.fcr_readable}%",
    "{fcr.terminus-2.fcr_readable_calibrated}%",
    "{fcr.large.calibrated_min} to {fcr.large.calibrated_max}%",
    "odds ratio of {ver.odds_adjusted.test_run} for claiming done "
    "({ver.odds_adjusted.test_run_lo} to {ver.odds_adjusted.test_run_hi})",
    f"check after the last edit has {ci('ver.after_edit')}",
    "labels ({val.honest_lo} to {val.honest_hi})",
    "AUROC is {det.closing} ({det.closing_lo} to {det.closing_hi})",
]

README = HEADLINE + [
    "For {split.readable.models} models seen",
    "estimate is {fcr.all.fcr_readable_calibrated}% by inverse",
    "{fcr.all.honest_readable}% of readable failed closings report failure by rule count, and "
    "{val.honest}% by adjudicated labels",
    "Rates use {bootstrap.reps} resamples of the {data.tasks} tasks",
    "on a sample whose rule rate ran {val.rule_rate_gap} points below",
    "The pool is {pool.terminus-2}% terminus-2 and {pool.mini-swe-agent}% mini-swe-agent",
    "the unweighted mean of the {pool.scaffolds_in_mean} main scaffolds is {pool.unweighted_mean}%",
    "the scaffold leads for {split.known.models} models, {split.known.raw.scaffold_share}% against "
    "{split.known.raw.model_share}%, a difference of {split.known.raw.scaffold_minus_model} points "
    "({split.known.raw.scaffold_minus_model_lo} to {split.known.raw.scaffold_minus_model_hi})",
    "at {fcr.terminus-2.fcr_readable}%, against {fcr.large.raw_min} to {fcr.large.raw_max}% for",
    "That gap mostly closes after calibration: {fcr.terminus-2.fcr_readable_calibrated}% against "
    "{fcr.large.calibrated_min} to {fcr.large.calibrated_max}%",
    "on both scales ({fcr.deepagent-harbor.fcr_readable}% and "
    "{fcr.deepagent-harbor.fcr_readable_calibrated}%), on only "
    "{fcr.deepagent-harbor.failed_readable} failed runs",
    "{ver.none_share}% of failed runs show no check",
    "({ver.odds_adjusted.test_run_lo} to {ver.odds_adjusted.test_run_hi}); a check after the last "
    "edit has {ver.after_edit} ({ver.after_edit_lo} to {ver.after_edit_hi}), or "
    "{ver.large.after_edit} ({ver.large.after_edit_lo} to {ver.large.after_edit_hi}) on the "
    "{ver.large.tasks} tasks with at least {ver.large.min_trials} trials",
    "The rules find only {val.reports_failure_recall_pct}% of these, so the per-scaffold rule "
    "rates, {fcr.scaffold.honest_min} to {fcr.scaffold.honest_max}%,",
    "({det.closing_lo} to {det.closing_hi}) from the closing message",
    "names alone reach {det.ids}",
    "inflates it to {det.loso.tasks_shared_min} to {det.loso.tasks_shared_max} on a held-out "
    "scaffold; holding tasks out gives {det.loso.tasks_held_out_min} to "
    "{det.loso.tasks_held_out_max}",
    "Validation: {val.n} messages, two blind labellers, kappa {val.kappa} ({val.kappa_lo} to "
    "{val.kappa_hi}), {val.disagreements} disagreements",
    "population-weighted precision {val.population_failed.claims-done.precision} and recall "
    "{val.population_failed.claims-done.recall} for claims-done "
    "({val.population.claims-done.precision} and {val.population.claims-done.recall} with "
    "passing runs included), but only {val.population_failed.hedged.precision} and "
    "{val.population_failed.hedged.recall} for hedged",
    "{val.command.rule_claims_held} of {val.command.rule_claims} rule claims held up",
    "{data.duplicates} rows are second listings of a trial, leaving {data.trials_with_steps} "
    "trials with steps from **{data.scaffolds_with_steps}** scaffolds and "
    "{data.models_with_steps} canonical models",
    "so {coverage.lost_share}% of closing statements are lost: "
    "{coverage.claude-code.lost_share}% for claude-code and {coverage.codex.lost_share}% for codex",
    "({bridge.openhands.lost.pct}% against {bridge.openhands.readable.pct}% on openhands, "
    "{bridge.terminus-2.lost.pct}% against {bridge.terminus-2.readable.pct}% on terminus-2)",
]

TABLE_SCAFFOLD = [
    f"| {s} | {{fcr.{s}.failed_readable_raw}} | {ci(f'fcr.{s}.fcr_readable')} "
    f"| {ci(f'fcr.{s}.fcr_readable_calibrated')} | {{fcr.{s}.claim_rate_passed_readable}} "
    f"| {{fcr.{s}.honest_readable}} | {{fcr.{s}.failed_lost_raw}} |"
    for s in SCAFFOLDS
]


def split_row(label: str, population: str, kind: str) -> str:
    key = f"split.{population}"
    return (
        f"| {label} | {{{key}.models}} | {{{key}.cells}} "
        f"| {ci(f'{key}.{kind}.scaffold_share').replace(' (', '% (')} "
        f"| {ci(f'{key}.{kind}.model_share').replace(' (', '% (')} "
        f"| {ci(f'{key}.{kind}.scaffold_minus_model')} |"
    )


REPORT = HEADLINE + [
    "estimate is **{fcr.all.fcr_readable_calibrated}%**",
    "For the {split.readable.models} models seen",
    "{val.honest}% of readable failed closings report failure by adjudicated labels",
    "and {fcr.all.honest_readable}% by rule count",
    "against {det.ids} from scaffold and model names alone",
    "For rates they come from {bootstrap.reps} resamples of the {data.tasks} tasks (seed "
    "{bootstrap.seed})",
    "at revision `04e8940f`, {data.rows} rows",
    "{data.duplicates} rows are a second listing of the same trial, which leaves {data.trials} "
    "trials",
    "Only {data.scaffolds_with_steps} of the {data.scaffolds_listed} listed scaffolds have steps: "
    "{data.trials_with_steps} trials, {data.failed_with_steps} of them failed. The "
    "{data.models_raw_listed} model strings are {data.models_listed} models, "
    "{data.models_with_steps} of them with steps",
    "The {data.scaffolds_without_steps} scaffolds without steps cannot be measured",
    "As a result, {coverage.lost_share}% of closings are lost. claude-code has "
    "{coverage.claude-code.lost} lost against {coverage.claude-code.readable} readable "
    "({coverage.claude-code.lost_share}%), and codex has {coverage.codex.lost} lost against "
    "{coverage.codex.readable} readable ({coverage.codex.lost_share}%)",
    "The {data.duplicates} duplicate pairs form a bridge sample",
    "**Validation.** {val.n} messages",
    "A third run adjudicated the {val.disagreements} disagreements. Kappa between the labellers "
    "is {val.kappa} ({val.kappa_lo} to {val.kappa_hi}), and raw agreement is {val.agreement}%",
    "Computed raw and task-adjusted, with {split.readable.replicates} task resamples",
    "{ver.regression_n} trials on {ver.regression_tasks} tasks",
    "among {det.n} claims-done trials",
    f"| Failed readable, rules | {ci('fcr.all.fcr_readable').replace(' (', '% (')} |",
    "| Failed readable, calibrated | "
    f"{ci('fcr.all.fcr_readable_calibrated').replace(' (', '% (')} |",
    "| Failed readable, Rogan-Gladen | "
    f"{ci('fcr.all.fcr_readable_corrected').replace(' (', '% (')} |",
    "| Failed readable, adjudicated labels only ({val.direct_n} messages) | "
    f"{ci('val.direct').replace(' (', '% (')} |",
    "| All failed with a known closing ({fcr.all.failed_known}), rules | "
    f"{ci('fcr.all.fcr').replace(' (', '% (')} |",
    "| Same, if every lost closing were silent / a claim | {fcr.all.fcr_if_lost_silent}% / "
    "{fcr.all.fcr_if_lost_claim}% |",
    "| Same, lost closings at the bridge's claim rates | {fcr.all.fcr_bridged}% |",
    "ran {val.rule_rate_gap} points below that of the closings it was drawn from "
    "({val.rule_rate_sample}% against {val.rule_rate_population}%)",
    "The pool is {pool.terminus-2}% terminus-2 and {pool.mini-swe-agent}% mini-swe-agent. The "
    "unweighted mean of the {pool.scaffolds_in_mean} scaffolds in it is {pool.unweighted_mean}%",
    "{coverage.terminus-3-3.no_text} terminus-3-3 declared ends carry no prose",
    *TABLE_SCAFFOLD,
    "{closing.terminus-2.fcr}% of {closing.terminus-2.n} claim done, against "
    "{closing.terminus-2+command.fcr}% of the {closing.terminus-2+command.n} closings",
    "rests on {fcr.deepagent-harbor.failed_readable} failed runs and sits exactly at the "
    "{val.deepagent-harbor.negatives}-negative limit",
    "the rules find only {val.reports_failure_recall_pct}% of failure reports",
    "Of {contrasts.pairs} pairwise contrasts among {contrasts.cells} cells on the readable rate, "
    "{contrasts.significant} are significant",
    split_row("Readable, raw", "readable", "raw"),
    split_row("Readable, task-adjusted", "readable", "adj"),
    split_row("All known, raw", "known", "raw"),
    split_row("All known, task-adjusted", "known", "adj"),
    "spans {split.readable.spread_model.gpt-5-nano} points across scaffolds for gpt-5-nano "
    "({split.readable.spread_model.gpt-5-nano_lo} to {split.readable.spread_model.gpt-5-nano_hi})",
    "spans {split.readable.spread_scaffold.codex} points across models under codex "
    "({split.readable.spread_scaffold.codex_lo} to {split.readable.spread_scaffold.codex_hi})",
    "{ver.none} of {ver.failed} failed runs show no check",
    "none: {ver.none.claims} of {ver.none.n}",
    "ad hoc check: {ver.adhoc-check.claims} of {ver.adhoc-check.n}",
    "test runner: {ver.test-runner.claims} of {ver.test-runner.n}",
    "the task's own check: {ver.task-check.claims} of {ver.task-check.n}",
    f"| Unadjusted | {ci('ver.odds_unadjusted.adhoc_check')} "
    f"| {ci('ver.odds_unadjusted.test_run')} |",
    f"| Scaffold and model | {ci('ver.odds_scaffold_model.adhoc_check')} "
    f"| {ci('ver.odds_scaffold_model.test_run')} |",
    f"| Scaffold, model and task | {ci('ver.odds_adjusted.adhoc_check')} "
    f"| {ci('ver.odds_adjusted.test_run')} |",
    f"A check after the last edit has an odds ratio of {ci('ver.after_edit')}, and "
    f"{ci('ver.large.after_edit')} on the {{ver.large.tasks}} tasks with at least "
    "{ver.large.min_trials} trials",
    f"| Closing message | {ci('det.closing')} | {{det.closing.within}} |",
    f"| Closing message and trajectory tail | {ci('det.closing+tail')} "
    "| {det.closing+tail.within} |",
    f"| Scaffold and model names only | {ci('det.ids')} | {{det.ids.within}} |",
    "Leaving one scaffold out gives {det.loso.tasks_shared_min} to {det.loso.tasks_shared_max} "
    "when tasks are shared between training and test, and {det.loso.tasks_held_out_min} to "
    "{det.loso.tasks_held_out_max} when tasks are held out",
    "claims-done {val.adjudicated.claims-done}, hedged {val.adjudicated.hedged}, no-statement "
    "{val.adjudicated.no-statement}, reports-failure {val.adjudicated.reports-failure}",
    "the rules have precision {val.population_failed.claims-done.precision} and recall "
    "{val.population_failed.claims-done.recall}",
    "**hedged:** precision {val.population_failed.hedged.precision} and recall "
    "{val.population_failed.hedged.recall}",
    "{val.kappa_rules} over four labels, and {val.kappa_rules_claim} for claim",
    "where {val.command.rule_claims_held} of {val.command.rule_claims} rule claims held up",
    "claim done at {bridge.lost}% ({bridge.lost_lo} to {bridge.lost_hi}), against "
    "{bridge.kept}% for kept ones",
    "agree on {bridge.agreement}% of the {bridge.text_pairs} pairs",
    "openhands: {bridge.openhands.lost.claims} of {bridge.openhands.lost.n} lost against "
    "{bridge.openhands.readable.claims} of {bridge.openhands.readable.n} kept",
    "terminus-2: {bridge.terminus-2.lost.claims} of {bridge.terminus-2.lost.n} lost against "
    "{bridge.terminus-2.readable.claims} of {bridge.terminus-2.readable.n} kept",
    "claude-code's all-failures rate is {fcr.claude-code.fcr}%, with bounds of "
    "{fcr.claude-code.fcr_if_lost_silent} and {fcr.claude-code.fcr_if_lost_claim}. codex's is "
    "{fcr.codex.fcr}%, with bounds of {fcr.codex.fcr_if_lost_silent} and "
    "{fcr.codex.fcr_if_lost_claim}",
    "**{data.scaffolds_with_steps} of {data.scaffolds_listed} scaffolds.** The "
    "{data.scaffolds_without_steps} without steps",
    "recall of {val.population_failed.hedged.recall} and {val.population_failed.reports-failure"
    ".recall}",
    "the calibrated ({fcr.all.fcr_readable_calibrated}%) and direct ({val.direct}%) estimates",
    "depends on {val.strata.terminus-2+command|0} adjudicated failed messages in that group",
    "**Trajectories for the {data.scaffolds_without_steps} scaffolds without steps.**",
]

FIXED = {
    "README.md": [
        "Python 3.11+",
        "Intervals are 95%",
        "Terminal-Bench 2",
        "against 0.83 to 0.95 in arXiv:2606.09863",  # the paper's figures, quoted
        "The dataset card's 34,462 trajectories, 26 scaffolds and 49 models",  # the card's
        "every string of 1024 or more characters",  # the scrape's limit
        "within ten points",
        "Apache-2.0",
    ],
    "REPORT.md": [
        "Terminal-Bench 2",
        "Results v1, as released in 0.2.0",
        "Intervals are 95%",
        "(95% CI)",
        "`reward = 0`",
        "`harbor-framework/terminal-bench-2`",
        "every string of 1024 or more characters",
        "arXiv:2606.09863",
        "arXiv:2606.09863 reports 0.83 to 0.95",  # the paper's figures, quoted
        "Holm 0.05",
    ],
    "CITATION.cff": [
        "cff-version: 1.2.0",
        "Terminal-Bench 2",
        "version: 0.2.0",
        "year: 2026",
        "notes: arXiv:2606.09863",
        "terminal-bench",
    ],
}

CLAIMS = {"README.md": README, "REPORT.md": REPORT, "CITATION.cff": []}


@pytest.mark.parametrize("name", sorted(CLAIMS))
def test_every_number_is_copied_from_the_generated_summary(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    result = check(text, CLAIMS[name], FIXED[name], NUMBERS)
    problems = [f"claim not found: {c}" for c in result.missing_claims]
    problems += [f"fixed text not found: {c}" for c in result.missing_fixed]
    problems += [f"number not from numbers.json: {c}" for c in result.unchecked]
    assert not problems, "\n".join(problems)


def test_the_check_catches_a_changed_or_unlisted_number():
    text = "Rates: 71.2% (7,606 of 10,677). See `gpt-5-nano`, 2026-11-06, v0.2.0."
    claims = ["{a}% ({b} of {c})"]
    values = {"a": "71.2", "b": "7,606", "c": "10,677"}
    clean = check(text, claims, [], values)
    assert not (clean.missing_claims or clean.unchecked)
    assert check(text, claims, [], {**values, "b": "7,607"}).missing_claims
    assert check(text + " Also 12 more.", claims, [], values).unchecked
