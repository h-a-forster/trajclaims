# trajclaims

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

When a coding agent's run fails the verifier, does the agent say it is done? And is that down to
the model or to the scaffold around it?

trajclaims audits the closing statements in the public
[Terminal-Bench 2 trajectory dataset](https://huggingface.co/datasets/yoonholee/terminalbench-trajectories)
against the verifier's verdicts, with no API cost. It uses frozen rule labels validated on blind
double labels, task-clustered intervals and a misclassification correction.

## Findings

All figures are in [results/v1](results/v1): rates in `fcr.csv` and `tables.md`, everything else
in `summary.json`. Intervals are 95%. Rates use 10,000 resamples of the 89 tasks;
[docs/DESIGN.md](docs/DESIGN.md) lists the other intervals.

- **Failed runs mostly say done.** Among failed runs that ended themselves with a readable closing
  statement, 71.2% claimed completion (7,606 of 10,677; 68.2 to 74.1). Adjusted for label error,
  the preferred estimate is 65.9% by inverse calibration (60.7 to 70.0). Rogan-Gladen gives 70.3%
  (60.9 to 80.8). Adjudicated labels alone give 61.5% (55.2 to 67.5), on a sample whose rule rate
  ran 5.5 points below its population's. The pool is 47.1% terminus-2 and 30.0% mini-swe-agent;
  the unweighted mean of the 7 main scaffolds is 68.6%.
- **The claim says little.** Agents claimed done on 90.5% of passing runs against 71.2% of failing
  ones, and 54.7% of readable done-claims were on failed runs.
- **Scaffold and model cannot be ranked on what agents say.** For 9 models seen under at least
  three scaffolds, the scaffold has 39.9% of the variance of cell FCR (30.5 to 55.8) and the model
  34.3% (23.0 to 48.3); the difference is 5.5 points (-14.9 to 30.8). Over all known closings,
  including silent ones, the scaffold leads for 12 models, 49.0% against 30.1%, a difference of
  19.0 points (7.3 to 31.5), but that counts cut-offs and silent signals, which scaffolds impose.
- **No scaffold is a clear outlier.** terminus-2 has the highest raw rate among large scaffolds at
  78.7%, against 60.4 to 68.5% for claude-code, codex, gemini-cli, mini-swe-agent and openhands.
  That gap mostly closes after calibration: 68.1% against 57.5 to 71.7%. deepagent-harbor is
  higher on both scales (78.0% and 89.2%), on only 50 failed runs.
- **Checking does not make agents more candid.** 78.8% of failed runs show no check (a lower
  bound on checking). Running a test tool has an adjusted odds ratio of 0.84 for claiming done
  (0.64 to 1.10); a check after the last edit has 1.30 (1.04 to 1.63), or 1.27 (1.00 to 1.60) on
  the 76 tasks with at least 50 trials.
- **Honest failure is uncommon.** 6.1% of readable failed closings report failure by rule count,
  and 10.4% by adjudicated labels (7.1 to 15.0). The rules find only 56% of these, so the
  per-scaffold rule rates, 2.0 to 17.1%, are rough and low.
- **TF-IDF detection is weaker here than reported elsewhere.** Task-disjoint AUROC is 0.75 (0.71
  to 0.79) from the closing message, against 0.83 to 0.95 in arXiv:2606.09863, and scaffold and
  model names alone reach 0.70. Sharing tasks between training and test inflates it to 0.83 to
  0.95 on a held-out scaffold; holding tasks out gives 0.56 to 0.75.

Validation: 653 messages, two blind labellers, kappa 0.78 (0.74 to 0.83), 70 disagreements
adjudicated by a third run. All are Claude models; there was no human adjudication. On failed
runs, against adjudicated labels, the rules have population-weighted precision 0.85 and recall
0.90 for claims-done (0.91 and 0.94 with passing runs included), but only 0.62 and 0.47 for
hedged. They do worst on terminus-2 closings sent beside commands, which often read as plans: 14
of 29 rule claims held up. Those closings are corrected with their own error rates.

## What the data can and cannot say

The dataset card's 34,462 trajectories, 26 scaffolds and 49 models shrink on inspection: 4,952
rows are second listings of a trial, leaving 29,510 trials with steps from **9** scaffolds and 31
canonical models. The original scrape replaced every string of 1024 or more characters with a
placeholder, and blanked OpenHands' `finish` argument, so 23.4% of closing statements are lost:
55.6% for claude-code and 62.3% for codex (of readable or lost closings). Those two carry the most
doubt. Trials listed by both scrapes form a bridge sample: lost prose closings claim done at
rates near kept ones, within ten points either way (78% against 68% on openhands, 73% against 78%
on terminus-2). It cannot cover claude-code or codex.

Associations only: cells ran on different dates, by different submitters. See
[docs/DESIGN.md](docs/DESIGN.md) for the critique of the original plan and each design decision.

## Reproduce

Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --group dev
uv run trajclaims fetch      # 221 MB, pinned revision, hashes checked
uv run trajclaims extract    # about 3 minutes
uv run trajclaims study      # regenerates results/v1 from the committed validation.jsonl
uv run trajclaims check      # labels byte-identical to the committed file
```

`labels.jsonl`, `summary.json`, `fcr.csv` and `contrasts.csv` regenerate byte for byte, whether or
not the detector's cached scores exist. Only labelling used a model (`trajclaims label`).

## Licence

MIT. The source dataset is Apache-2.0; `results/v1/validation.jsonl` quotes messages from it.
