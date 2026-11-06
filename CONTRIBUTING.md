# Contributing

Bug reports and documentation fixes are welcome. Read [docs/DESIGN.md](docs/DESIGN.md) first: it
fixes the labels, estimands and intervals.

## Setup

You need [uv](https://docs.astral.sh/uv/) and git.

```sh
uv sync --group dev
uv run trajclaims --version
```

## Tests and lint

```sh
uv run pytest              # offline
uv run pytest --runslow    # plus the clustered-null size check
uv run ruff check .
uv run ruff format --check .
uv lock --check
```

Tests must run offline, write only under pytest's `tmp_path`, use fixed seeds and pass on Windows.
Statistical assertions state their tolerance in Monte Carlo standard errors. A bug fix comes with
a test that fails without it.

## Rules and results

- `src/trajclaims/rules.py` is frozen for `results/v1`. A change needs a new sample and a new
  results directory.
- Regenerate results with `uv run trajclaims study`; never edit them by hand.
- Never commit raw downloads or caches (`data/`, `tmp/`).
- Add change notes to `docs/release-notes-<version>.md`, not to a CHANGELOG.
- Keep [backlog.md](backlog.md) current: one line per open item.

Security issues: see [SECURITY.md](SECURITY.md).
