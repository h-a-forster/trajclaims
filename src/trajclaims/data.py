"""Fetch the pinned dataset and reduce each trial to one row, unit by unit."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trajclaims.closing import EMBEDDED_REF, REF, closing, tool_cmd
from trajclaims.models import MODELS
from trajclaims.verify import own_check_pattern, verification

DATASET = "yoonholee/terminalbench-trajectories"
REVISION = "04e8940f5b6736a7ce8d22224fe2f2af74163ed2"
SHARDS = {
    "train-00000-of-00002.parquet": (
        "e95bc8dc9f3e1b2f3c6c383d0a060064205d87e93e9ac9541847d1c511effdfa"
    ),
    "train-00001-of-00002.parquet": (
        "00d61614054bedcbadda9974cef30fe1e07d1701e7126edfa210f99840ae99b2"
    ),
}
COLUMNS = [
    "task_name",
    "agent",
    "model",
    "reward",
    "duration_seconds",
    "cost_cents",
    "trial_name",
    "trial_id",
    "steps",
]
KEY = ["task", "scaffold", "model_raw", "trial"]
# Bump when extraction output changes, so stale units are rebuilt.
EXTRACT_SCHEMA = 2
TAIL_CHARS = 6000
# A trial at 98% of its task's agent timeout counts as having hit it.
TIMEOUT_MARGIN = 0.98
ROOT = Path(__file__).resolve().parents[2]
TASKS_FILE = ROOT / "configs" / "tasks.json"


class DataError(Exception):
    """The data do not match what the study was written against."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(raw_dir: Path, log: Callable[[str], None] = print) -> None:
    """Download any missing shard at the pinned revision and check its hash."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in SHARDS.items():
        path = raw_dir / name
        if not path.exists():
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{name}"
            log(f"downloading {url}")
            part = path.with_suffix(".part")
            urllib.request.urlretrieve(url, part)  # noqa: S310 - fixed https URL
            os.replace(part, path)
        actual = sha256_file(path)
        if actual != expected:
            raise DataError(f"{name}: sha256 {actual}, expected {expected}")
        log(f"ok {name}")


def trajectory_tail(steps: list[dict[str, Any]], closing_text: str) -> str:
    """The trajectory before the closing statement, as TF-IDF input.

    Serialised as in arXiv:2606.09863 (arguments to 200 characters, results to 300). Only the
    tail is kept, and scrape references are dropped so a detector cannot read their hex ids.
    """
    parts: list[str] = []
    for step in steps:
        if step.get("src") != "agent":
            continue
        msg = (step.get("msg") or "").strip()
        if msg and msg != closing_text and not REF.fullmatch(msg):
            parts.append("[ASST] " + msg)
        for tool in step.get("tools") or []:
            parts.append(f"[TOOL CALL] {tool.get('fn')}({tool_cmd(tool)[:200]})")
        obs = step.get("obs")
        if isinstance(obs, str) and obs:
            parts.append("[TOOL RESULT] " + obs[:300])
    return EMBEDDED_REF.sub(" ", "\n".join(parts))[-TAIL_CHARS:]


def load_tasks(path: Path = TASKS_FILE) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["tasks"]


def extract_row(row: dict[str, Any], tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """One dataset row as one flat record."""
    if row["model"] not in MODELS:
        raise DataError(f"unknown model {row['model']!r}: add it to models.MODELS")
    raw = row["steps"]
    steps = (json.loads(raw) if raw else None) or []
    if row["task_name"] not in tasks:
        raise DataError(f"unknown task {row['task_name']!r}: rebuild configs/tasks.json")
    task = tasks[row["task_name"]]
    limit = task["agent_timeout_s"]
    timed_out = (row["duration_seconds"] or 0.0) >= TIMEOUT_MARGIN * limit
    end = closing(row["agent"], steps, timed_out)
    check = verification(steps, own_check_pattern(task.get("own_check", [])))
    return {
        "task": row["task_name"],
        "scaffold": row["agent"],
        "model": MODELS[row["model"]],
        "model_raw": row["model"],
        "trial": row["trial_name"],
        "full_scrape": bool(row["trial_id"]),
        "reward": int(row["reward"]),
        "duration_s": row["duration_seconds"],
        "timed_out": timed_out,
        "cost_cents": row["cost_cents"],
        "n_steps": len(steps),
        "steps_sha": hashlib.sha256((raw or "").encode("utf-8")).hexdigest()[:16],
        "ending": end.ending if steps else "none",
        "status": end.status if steps else "no-steps",
        "signal": end.signal,
        "closing_text": end.text,
        "with_command": end.with_command,
        "verification": check.level,
        "verified_after_edit": check.after_last_edit,
        "shell_commands": check.shell_commands,
        "opaque_commands": check.opaque_commands,
        "blank_commands": check.blank_commands,
        "tail": trajectory_tail(steps, end.text),
    }


def extract(
    raw_dir: Path, unit_dir: Path, log: Callable[[str], None] = print, force: bool = False
) -> None:
    """Write one parquet unit per row group; a finished unit is skipped on rerun."""
    unit_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    for name in sorted(p.name for p in raw_dir.glob("*.parquet")):
        parquet = pq.ParquetFile(raw_dir / name)
        for group in range(parquet.metadata.num_row_groups):
            unit = unit_dir / f"s{EXTRACT_SCHEMA}-{name.split('-of-')[0]}-rg{group:02d}.parquet"
            if unit.exists() and not force:
                continue
            rows = parquet.read_row_group(group, columns=COLUMNS).to_pylist()
            table = pa.Table.from_pylist([extract_row(row, tasks) for row in rows])
            part = unit.with_suffix(".part")
            pq.write_table(table, part, compression="zstd")
            os.replace(part, unit)
            log(f"unit {unit.name}: {len(rows)} rows")


def load_trials(unit_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """All units as one table, with one row per trial.

    4,952 trials are listed twice: once from a full scrape, once from the partial one. The full
    row is kept; its partial twin's status stays in `partial_status` as the bridge sample of
    what the partial scrape loses. Twins must agree on reward and step count.
    """
    units = sorted(unit_dir.glob(f"s{EXTRACT_SCHEMA}-*.parquet"))
    if not units:
        raise DataError(f"no extracted units in {unit_dir}: run `trajclaims extract`")
    frame = pd.concat([pd.read_parquet(u) for u in units], ignore_index=True)
    groups = frame.groupby(KEY, sort=False)
    if (groups["reward"].nunique() > 1).any() or (groups["n_steps"].nunique() > 1).any():
        raise DataError("duplicate trial keys disagree on reward or step count")
    if (groups.size() > 2).any():
        raise DataError("a trial is listed more than twice")
    twins = frame[frame.duplicated(KEY, keep=False)]
    if (twins.groupby(KEY, sort=False)["full_scrape"].sum() != 1).any():
        raise DataError("a duplicate pair is not one full and one partial row")
    partial = twins.loc[~twins["full_scrape"], [*KEY, "status", "closing_text"]].rename(
        columns={"status": "partial_status", "closing_text": "partial_text"}
    )
    kept = frame.sort_values("full_scrape", ascending=False, kind="stable")
    trials = kept.drop_duplicates(KEY, keep="first").merge(partial, on=KEY, how="left")
    trials = trials.sort_values(KEY, ignore_index=True)
    counts = {"rows": len(frame), "duplicates": len(frame) - len(trials), "trials": len(trials)}
    return trials, counts
