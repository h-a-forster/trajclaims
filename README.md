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
in `summary.json`. Intervals are 95%, from 10,000 resamples of the 89 tasks.

- **Failed runs mostly say done.** Among failed runs that ended themselves with a readable closing
  statement, 73.8% claimed completion (7,100 of 9,625; 70.6 to 76.7). Adjusted for label error it
  is 70.1% by inverse calibration (64.4 to 74.6) and 74.8% by Rogan-Gladen.
- **The claim says little.** Agents claimed done on 91.5% of passing runs against 73.8% of failing
  ones, and 53.9% of readable done-claims were on failed runs.
- **The scaffold carries more than the model.** For 7 models seen under at least three scaffolds,
  the scaffold has 66.5% of the variance of cell FCR (50.1 to 74.1) and the model 12.8% (6.9 to
  24.7). Over all known closings, including silent ones, it is 42.7% against 31.7%. terminus-2 is
  the outlier: 86.8% of failed runs claim done, against 60.4 to 68.5% for claude-code, codex,
  gemini-cli, mini-swe-agent and openhands.
- **Checking does not make agents more candid.** 78.8% of failed runs show no check (a lower
  bound on checking). Running a test tool has an adjusted odds ratio of 0.84 for claiming done
  (0.63 to 1.13); a check after the last edit has 1.35 (1.05 to 1.73).
- **Honest failure is rare.** 2.0 to 17.1% of readable failed closings per main scaffold say so.
- **TF-IDF detection is weaker here than reported elsewhere.** Task-disjoint AUROC is 0.74 (0.69
  to 0.78) from the closing message, against 0.83 to 0.95 in arXiv:2606.09863, and scaffold and
  model names alone reach 0.69. Sharing tasks between training and test inflates it to 0.82 to
  0.95 on a held-out scaffold; holding tasks out gives 0.63 to 0.76.

Validation: 573 messages, two blind labellers, kappa 0.79 (0.74 to 0.85), 52 disagreements
adjudicated by a third run. All are Claude models; there was no human adjudication. Against
adjudicated labels the rules have population-weighted precision 0.93 and recall 0.94 for
claims-done, but only 0.61 and 0.45 for hedged.

## What the data can and cannot say

The dataset card's 34,462 trajectories, 26 scaffolds and 49 models shrink on inspection: 4,952
rows are second listings of a trial, leaving 29,510 trials with steps from **9** scaffolds and 31
canonical models. The original scrape replaced every string of 1024 or more characters with a
placeholder, and blanked OpenHands' `finish` argument, so 23.5% of closing statements are lost:
55.6% for claude-code and 62.3% for codex (of readable or lost closings). Those two carry the most
doubt. Trials listed by both scrapes form a bridge sample: lost prose closings claim done about as often
as kept ones, or more (78% against 68% on openhands, 84% against 86% on terminus-2). It cannot
cover claude-code or codex.

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

`labels.jsonl`, `summary.json`, `fcr.csv` and `contrasts.csv` regenerate byte for byte. Only
labelling used a model (`trajclaims sample` and `trajclaims label`, run once).

## Licence

MIT. The source dataset is Apache-2.0; `results/v1/validation.jsonl` quotes messages from it.
