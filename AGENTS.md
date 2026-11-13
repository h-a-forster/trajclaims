# AGENTS.md

Guide for coding agents (and humans) working on trajclaims. Read `docs/DESIGN.md` before changing
any statistics, labels or data handling: it fixes them and records why.

## What this is

A study and small package. It measures false completion (the agent says done, the verifier says
no) across the public Terminal-Bench 2 trajectories that kept their steps, with validated labels
and task-clustered intervals.

## Conventions

- uv + hatchling, `src/` layout, Python >= 3.11 (local dev 3.13), typed code, `ruff check` +
  `ruff format` (line length 100), pytest. `[tool.uv] exclude-newer` keeps the lock on releases at
  least two weeks old; move it deliberately.
- Runtime dependencies: numpy, scipy, pandas, pyarrow, scikit-learn. statsmodels is a dev-only
  reference for the regression tests.
- `rules.py` is frozen for `results/v1`. Any change voids the validation: `study` refuses, by
  design. A new rule version needs a new `results/v<n>` and a new sample.
- An extraction change that moves closings between strata leaves the rules valid and the sample
  stale. `study` refuses that too, until `trajclaims sample --extend` and `trajclaims label` cover
  the new strata.
- The study makes no model calls. Only `trajclaims label` does, and its output is committed.
- Every stochastic function takes a seed. `summary.json`, `fcr.csv`, `contrasts.csv` and
  `labels.jsonl` regenerate byte for byte.
- Every number in `README.md`, `REPORT.md` and `CITATION.cff` is copied from
  `results/v1/numbers.json`, which `trajclaims summarize` rebuilds (with `tables.md`) from the
  committed results; `tests/test_docs.py` lists each claim and fails on any number it does not
  cover. A new number needs a key in `report.numbers` and a claim, or a fixed entry whose source
  the document states. CI rebuilds and runs `git diff --exit-code results/`.
- Validate before writing; raise `DataError`. Count every excluded trial.
- Long work is unit files written atomically: `data/units/` per row group, `data/detector/` per
  fold under a hash of the inputs, and labeller answers appended per batch. Finished units are skipped; `--force` rebuilds
  extraction. Bump `EXTRACT_SCHEMA` or `DETECTOR_SCHEMA` when output changes.
- `data/` and `tmp/` are gitignored. Never commit raw shards.
- Commits and PRs carry no generated-by trailers. CI runs on pushes to main, pull requests and
  manual dispatch.
- Release hygiene: no personal names, local user paths or machine details in committed files.
- Statistical honesty: report sizes, seeds and intervals; say when a number is a bound; the
  labellers are models, and there was no human adjudication.

## Architecture map

| Path | Role |
|---|---|
| `src/trajclaims/models.py` | raw model string to canonical name |
| `src/trajclaims/closing.py` | how a trajectory ended, and its closing statement |
| `src/trajclaims/verify.py` | verification level from commands |
| `src/trajclaims/data.py` | pinned fetch, extraction units, duplicate handling |
| `src/trajclaims/rules.py` | the frozen labeller |
| `src/trajclaims/validation.py` | development split, sample, rule accuracy |
| `src/trajclaims/labellers.py` | blind labels through the `claude` CLI |
| `src/trajclaims/stats.py` | bootstrap, Holm, Rogan-Gladen, kappa, AUROC, clustered regressions |
| `src/trajclaims/study.py` | rates, bridge, sensitivity, outputs |
| `src/trajclaims/models_effects.py` | estimands 3 and 4 |
| `src/trajclaims/detector.py` | TF-IDF detector |
| `src/trajclaims/report.py` | generated `results/v1/tables.md` and `numbers.json` |
| `scripts/fetch_tasks.py` | rebuild `configs/tasks.json` at a pinned commit |
| `results/v1/` | committed outputs |

## How to test

```sh
uv sync --group dev
uv run pytest                 # offline
uv run pytest --runslow       # plus the clustered-null size check
uv run ruff check .
uv run ruff format --check .
uv lock --check
```

## Reproduce

```sh
uv run trajclaims fetch       # 221 MB, hashes checked
uv run trajclaims extract     # about 3 minutes
uv run trajclaims study       # regenerates results/v1 from the committed validation.jsonl
uv run trajclaims summarize   # tables.md and numbers.json only, from committed results; offline
uv run trajclaims check       # labels byte-identical to the committed file
```

`trajclaims sample` and `trajclaims label` produced `validation.jsonl`; `label` needs the
`claude` CLI, and neither is part of a rerun. `sample --extend` adds strata, and `label` asks only
about unlabelled messages, which needs the raw answers in `data/validation`.

Open work lives in `backlog.md`. Findings are in `README.md`; every figure is in `results/v1`.
