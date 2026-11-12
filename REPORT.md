# trajclaims report: what failed coding agents say at the end of Terminal-Bench 2 runs

Results v1, as released in 0.2.0. Every number comes from `results/v1`: rates from `fcr.csv` and
`tables.md`, everything else from `summary.json`. Intervals are 95%. For rates they come from
10,000 resamples of the 89 tasks (seed 20261106). Labels, estimands and the critique of the original
plan: [docs/DESIGN.md](docs/DESIGN.md).

## Summary

- **Failed runs mostly say done.** Of failed runs that ended themselves with a readable closing
  statement, 71.2% claimed completion (7,606 of 10,677; 68.2 to 74.1). Corrected for label error,
  the preferred estimate is **65.9%** by inverse calibration (60.7 to 70.0). Rogan-Gladen gives
  70.3% (60.9 to 80.8), and the adjudicated labels alone give 61.5% (55.2 to 67.5).
- **The claim carries little information.** Agents claimed done on 90.5% of passing runs against
  71.2% of failing ones, and 54.7% of readable done-claims were on failed runs.
- **Scaffold and model cannot be ranked.** For the 9 models seen under at least three scaffolds,
  the scaffold has 39.9% of the variance of cell FCR (30.5 to 55.8) and the model 34.3% (23.0 to
  48.3). The difference, 5.5 points (-14.9 to 30.8), includes zero.
- **No scaffold is a clear outlier after calibration.** terminus-2 has the highest raw rate among
  the large scaffolds, 78.7%. Calibrated, it is 68.1%, inside the 57.5 to 71.7% of claude-code,
  codex, gemini-cli, mini-swe-agent and openhands.
- **Checking does not make agents more candid.** Running a test tool has an adjusted odds ratio of
  0.84 for claiming done (0.64 to 1.10). A check after the last edit has 1.30 (1.04 to 1.63).
- **Honest failure is uncommon.** 10.4% of readable failed closings report failure by adjudicated
  labels (7.1 to 15.0), and 6.1% by rule count.
- **A TF-IDF detector of false claims is weaker than reported elsewhere.** Task-disjoint AUROC is
  0.75 (0.71 to 0.79), against 0.70 from scaffold and model names alone.

## Question

When a coding agent's work fails an independent verifier, does the agent say it is done? And does
that depend on the model or on the scaffold around it? The study reads the closing statements in the
public Terminal-Bench 2 trajectory dataset against the verifier's all-or-nothing reward. The only
model calls are for labelling a validation sample, so it costs nothing to rerun.

## Data

Source: `yoonholee/terminalbench-trajectories` at revision `04e8940f`, 52,104 rows, with shard hashes
in `summary.json`. Task metadata comes from `harbor-framework/terminal-bench-2` at the commit in
`configs/tasks.json`.

- 4,952 rows are a second listing of the same trial, which leaves 47,152 trials.
- Only 9 of the 26 listed scaffolds have steps: 29,510 trials, 19,495 of them failed. The 49 model
  strings are 36 models, 31 of them with steps.
- The 17 scaffolds without steps cannot be measured. They include Factory Droid, Goose, mux and
  Junie.

The original scrape replaced every string of 1024 or more characters with a placeholder, and it
blanked OpenHands' `finish` argument. As a result, 23.4% of closings are lost. claude-code has 1,201
lost against 959 readable (55.6%), and codex has 799 lost against 483 readable (62.3%) (`tables.md`,
Coverage). The 4,952 duplicate pairs form a bridge sample: one row of each pair comes from a scrape
that kept the full strings.

## Method

- **Endings.** `closing.py` models how each scaffold ends a run (declared signal, stopped or cut off)
  and takes the closing statement from that step. When terminus-2 sends its completion signal
  beside commands, that closing becomes its own group, `terminus-2+command`.
- **Labels.** Frozen deterministic rules (`rules-1`) assign `claims-done`, `hedged`,
  `reports-failure` or `no-statement`. The rules were written on a development fifth held out by
  hash, and committed before the validation sample was drawn.
- **Validation.** 653 messages, stratified by scaffold x reward, were labelled blind by two model
  labellers with different prompts. A third run adjudicated the 70 disagreements. Kappa between the
  labellers is 0.78 (0.74 to 0.83), and raw agreement is 89.3%. All labellers are Claude models;
  there was no human adjudication.
- **Headline estimand.** `fcr_readable`: claims-done over failed, self-ended trials with a readable
  closing. The planned denominator, all failed trials with a known closing, is kept as `fcr`.
- **Correction.** Per closing group, on failed trials, three estimates: inverse calibration from
  the adjudicated predictive values (preferred), Rogan-Gladen from sensitivity and specificity,
  and the direct adjudicated rate.
- **Model against scaffold.** Shapley shares of the weighted variance of cell means, over models
  seen under at least three scaffolds. Computed raw and task-adjusted, with 2,000 task resamples.
- **Verification.** A logit of claims-done on failed readable trials, with scaffold, model and task
  fixed effects and task-clustered CR1 errors: 10,067 trials on 89 tasks.
- **Detector.** TF-IDF plus logistic regression with the settings of arXiv:2606.09863, predicting
  `reward = 0` among 13,914 claims-done trials, in five folds grouped by task.

## Results

### False completion rate

| Population | Rate (95% CI) |
|---|---|
| Failed readable, rules | 71.2% (68.2 to 74.1) |
| Failed readable, calibrated | 65.9% (60.7 to 70.0) |
| Failed readable, Rogan-Gladen | 70.3% (60.9 to 80.8) |
| Failed readable, adjudicated labels only (461 messages) | 61.5% (55.2 to 67.5) |
| All failed with a known closing (16,296), rules | 46.7% (41.8 to 51.6) |
| Same, if every lost closing were silent / a claim | 39.0% / 55.4% |
| Same, lost closings at the bridge's claim rates | 50.5% |

The adjudicated rate sits below the calibrated one because the sample's rule rate ran 5.5 points
below that of the closings it was drawn from (65.2% against 70.7%). The pool is 47.1% terminus-2
and 30.0% mini-swe-agent. The unweighted mean of the 7 scaffolds in it is 68.6%.

The all-failures rate counts cut-off runs and wordless completion signals as non-claims, and the
scaffold decides those. For example, 562 terminus-3-3 declared ends carry no prose.

### By scaffold

Failed trials only (`tables.md`).

| Scaffold | Failed readable | Rules % (95% CI) | Calibrated % (95% CI) | Claim given pass % | Reports failure % | Failed, lost |
|---|---|---|---|---|---|---|
| claude-code | 456 | 65.1 (56.3 to 73.2) | 57.9 (45.5 to 68.6) | 87.7 | 17.1 | 654 |
| codex | 265 | 60.4 (50.0 to 69.4) | 62.8 (51.0 to 73.1) | 80.3 | 12.1 | 554 |
| deepagent-harbor | 50 | 78.0 (63.6 to 88.7) | 89.2 (76.7 to 95.3) | 90.0 | 2.0 | 7 |
| gemini-cli | 384 | 68.5 (60.5 to 75.9) | 71.7 (61.8 to 81.0) | 91.4 | 16.7 | 107 |
| mini-swe-agent | 3205 | 63.9 (60.4 to 67.3) | 65.9 (55.7 to 74.1) | 90.4 | 4.8 | 260 |
| openhands | 1287 | 65.3 (61.1 to 69.6) | 57.5 (46.7 to 68.0) | 81.2 | 10.3 | 1013 |
| terminus-2 | 5027 | 78.7 (75.8 to 81.5) | 68.1 (60.3 to 74.2) | 92.7 | 3.7 | 587 |

- **terminus-2.** Its raw lead comes from its plain closings: 86.8% of 3,975 claim done, against
  48.1% of the 1,052 closings sent beside commands. Calibrating with each group's own error rates
  closes most of the gap.
- **deepagent-harbor** is higher on both scales, but it rests on 50 failed runs and sits exactly at
  the 5-negative limit for its own correction rates.
- **Failure reports.** The per-scaffold rates are low because the rules find only 56% of failure
  reports.

Per-model and per-cell rates are in `tables.md`. Of 1,891 pairwise contrasts among 62 cells on the
readable rate, 320 are significant at Holm 0.05 (`contrasts.csv`).

### Scaffold against model

| Closings | Models | Cells | Scaffold share | Model share | Scaffold minus model |
|---|---|---|---|---|---|
| Readable, raw | 9 | 32 | 39.9% (30.5 to 55.8) | 34.3% (23.0 to 48.3) | 5.5 (-14.9 to 30.8) |
| Readable, task-adjusted | 9 | 32 | 39.6% (29.4 to 55.4) | 34.2% (22.9 to 48.5) | 5.5 (-16.4 to 29.4) |
| All known, raw | 12 | 44 | 49.0% (42.8 to 57.2) | 30.1% (24.2 to 36.9) | 19.0 (7.3 to 31.5) |
| All known, task-adjusted | 12 | 44 | 46.4% (39.9 to 55.0) | 33.3% (27.3 to 40.2) | 13.1 (1.4 to 26.6) |

On all known closings the scaffold leads. That population, though, counts cut-offs and silent
signals, which the scaffold decides. On readable closings, the comparable population, neither
leads. Both move a cell by tens of points:

- With the model fixed, the readable rate spans 40.4 points across scaffolds for gpt-5-nano
  (21.7 to 59.0).
- With the scaffold fixed, it spans 44.9 points across models under codex (18.2 to 71.7).

### Verification

Verification levels are a lower bound, because placeholders hide commands: 15,358 of 19,495 failed
runs show no check. Claims-done on failed readable closings, by level:

- none: 5,699 of 7,920
- ad hoc check: 829 of 1,036
- test runner: 521 of 698
- the task's own check: 286 of 413

| Odds ratio for claiming done | Ad hoc check | Test runner |
|---|---|---|
| Unadjusted | 1.56 (1.14 to 2.13) | 1.03 (0.75 to 1.42) |
| Scaffold and model | 1.36 (0.97 to 1.91) | 0.93 (0.67 to 1.29) |
| Scaffold, model and task | 1.19 (0.91 to 1.55) | 0.84 (0.64 to 1.10) |

A check after the last edit has an odds ratio of 1.30 (1.04 to 1.63), and 1.27 (1.00 to 1.60) on the
76 tasks with at least 50 trials. No form of checking is shown to lower the claim rate.

### Detector

| Features | AUROC (95% CI) | Inside scaffold x model cells |
|---|---|---|
| Closing message | 0.75 (0.71 to 0.79) | 0.70 |
| Closing message and trajectory tail | 0.77 (0.73 to 0.81) | 0.70 |
| Scaffold and model names only | 0.70 (0.67 to 0.72) | 0.50 |

arXiv:2606.09863 reports 0.83 to 0.95, with a regex over the same closing message as its label.
Leaving one scaffold out gives 0.83 to 0.95 when tasks are shared between training and test, and
0.56 to 0.75 when tasks are held out. Much of a high AUROC here is task identity.

### Label quality

Adjudicated labels: claims-done 460, hedged 78, no-statement 65, reports-failure 50.

- **claims-done:** on failed runs, weighted to the population, the rules have precision 0.85 and
  recall 0.90.
- **hedged:** precision 0.62 and recall 0.47.
- **Kappa between rules and adjudicated labels:** 0.50 over four labels, and 0.63 for claim against
  the rest.
- **Worst group:** the rules do worst on `terminus-2+command`, where 14 of 29 rule claims held up.
  That group is corrected with its own rates.

### Lost closings

In the bridge sample, failed lost closings read from the full scrape claim done at 63.8% (59.9 to
67.6), against 69.0% for kept ones. That interval is Wilson on messages, not clustered. The two
scrapes' labels agree on 95.9% of the 3,330 pairs with text in both.

By scaffold, lost prose closings claim at rates near those of kept ones:

- openhands: 229 of 293 lost against 145 of 212 kept
- terminus-2: 142 of 195 lost against 1,331 of 1,710 kept

The bridge has no claude-code or codex pairs, so their bounds stay wide. claude-code's all-failures
rate is 38.6%, with bounds of 20.9 and 66.8. codex's is 42.2%, with bounds of 17.1 and 76.5.

## Limitations

- **Associations only.** Cells ran on different dates, by different submitters, with different
  settings. Nothing here shows that a scaffold causes what it correlates with.
- **9 of 26 scaffolds.** The 17 without steps cannot be compared.
- **Lost text is not missing at random.** Whether a closing is lost depends on its length.
  claude-code and codex lose over half their closings, and the bridge cannot cover them.
- **Model labellers from one vendor.** Their errors may correlate, and they may read `claims-done`
  the way a model would write it. Agreement measures consistency, not truth.
- **The rules are weak beyond claims-done.** Hedged and failure-report counts rest on rules with
  recall of 0.47 and 0.56. Prefer the adjudicated honest rate.
- **Verifier semantics.** Reward is all-or-nothing. Infrastructure errors that the timeout check
  misses stay in the failed set, so some "false" claims may be on nearly passing or broken runs.
- **Labels read what an agent wrote**, not what it believed.
- **Some intervals are slightly narrow.** Task-adjusted share intervals hold task effects at their
  full-sample estimates, and the bridge's Wilson interval ignores clustering.

## What would change the conclusion

- **Human adjudication** that disagrees with the model labellers on what counts as a claim. If
  human and model labels diverge by more than the calibrated (65.9%) and direct (61.5%) estimates
  do, the labels, not the agents, would be setting the rate.
- **Recovered closings for claude-code and codex.** If their lost closings claim at a rate far from
  their kept ones, their rates and the scaffold comparison move. The bounds above show how far.
- **Rules for closings sent beside commands**, built under a new `results/v2` with a new sample.
  terminus-2's calibrated rate depends on 60 adjudicated failed messages in that group.
- **A crossed design** that runs the same models under the same scaffolds on one date. Only that
  could rank scaffold against model; the observational shares here cannot.
- **Trajectories for the 17 scaffolds without steps.** Any of them could be the outlier this study
  does not find.

## Reproduce

```sh
uv sync --group dev
uv run trajclaims fetch      # 221 MB, pinned revision, hashes checked
uv run trajclaims extract    # about 3 minutes
uv run trajclaims study      # regenerates results/v1 from the committed validation.jsonl
uv run trajclaims check      # labels byte-identical to the committed file
```
