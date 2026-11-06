"""End to end on a small parquet shaped like the dataset; offline."""

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trajclaims import data, detector, labellers, study, validation

TASKS = {"alpha": {"agent_timeout_s": 900.0}, "beta": {"agent_timeout_s": 900.0}}


def row(**over):
    steps = [
        {"src": "user", "msg": "$31", "tools": None, "obs": None},
        {"src": "agent", "msg": "Done. Wrote the file.", "tools": None, "obs": None},
    ]
    base = {
        "task_name": "alpha",
        "agent": "codex",
        "model": "gpt-5@openai",
        "reward": 0,
        "duration_seconds": 100.0,
        "cost_cents": 1.0,
        "trial_name": "alpha__1",
        "trial_id": "",
        "steps": json.dumps(steps),
    }
    return {**base, **over}


def write_shard(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def tasks(monkeypatch):
    monkeypatch.setattr(data, "load_tasks", lambda: TASKS)


def built(tmp_path, rows):
    raw, units = tmp_path / "raw", tmp_path / "units"
    write_shard(raw / "train-00000-of-00001.parquet", rows)
    data.extract(raw, units, lambda _: None)
    return units


def test_labels_regenerate_byte_identically(tmp_path, tasks):
    rows = [
        row(trial_name=f"alpha__{i}", reward=i % 2, duration_seconds=100.0 + i) for i in range(30)
    ]
    first = built(tmp_path / "one", rows)
    second = built(tmp_path / "two", rows)
    a = study.labels_jsonl(study.assign_labels(data.load_trials(first)[0]))
    b = study.labels_jsonl(study.assign_labels(data.load_trials(second)[0]))
    assert a == b
    assert a.count(b"\n") == 30 and b"\r" not in a


def test_finished_units_are_skipped(tmp_path, tasks):
    units = built(tmp_path, [row()])
    unit = next(units.glob("*.parquet"))
    stamp = unit.stat().st_mtime_ns
    data.extract(tmp_path / "raw", units, lambda _: None)
    assert unit.stat().st_mtime_ns == stamp


def test_the_full_scrape_wins_a_duplicate_listing(tmp_path, tasks):
    lost = [{"src": "agent", "msg": "$3e", "tools": None, "obs": None}]
    full = [{"src": "agent", "msg": "Done. Wrote the file.", "tools": None, "obs": None}]
    units = built(
        tmp_path,
        [row(steps=json.dumps(lost)), row(steps=json.dumps(full), trial_id="abc")],
    )
    trials, counts = data.load_trials(units)
    assert counts == {"rows": 2, "duplicates": 1, "trials": 1}
    one = trials.iloc[0]
    assert (one["status"], one["partial_status"]) == ("readable", "lost")


def test_twins_that_disagree_stop_the_run(tmp_path, tasks):
    units = built(tmp_path, [row(reward=0), row(reward=1, trial_id="abc")])
    with pytest.raises(data.DataError, match="disagree"):
        data.load_trials(units)


def test_an_unknown_model_stops_the_run():
    with pytest.raises(data.DataError, match="unknown model"):
        data.extract_row(row(model="new-model@x"), TASKS)


def test_a_wrong_shard_hash_stops_the_run(tmp_path):
    name = next(iter(data.SHARDS))
    (tmp_path / name).write_bytes(b"not the dataset")
    other = next(n for n in data.SHARDS if n != name)
    (tmp_path / other).write_bytes(b"")
    with pytest.raises(data.DataError, match="sha256"):
        data.fetch(tmp_path, lambda _: None)


def test_scrape_references_do_not_reach_detector_text():
    steps = [
        {"src": "agent", "msg": "Run it", "tools": [{"fn": "Bash", "cmd": "ls"}], "obs": "$4f"},
        {"src": "agent", "msg": "$5a", "tools": None, "obs": None},
    ]
    tail = data.trajectory_tail(steps, "")
    assert "$4f" not in tail and "$5a" not in tail and "Run it" in tail


def sample_rows():
    return [
        {"id": f"m{i}", "text": "reward secret scaffold secret"[:6] + str(i)} for i in range(45)
    ]


def fake_cli(prompts):
    def run(argv, stdin):
        prompts.append(stdin)
        ids = [line.split('"')[1] for line in stdin.splitlines() if line.startswith("<message id=")]
        result = json.dumps({"labels": dict.fromkeys(ids, "claims-done")})
        return json.dumps({"result": result, "modelUsage": {"claude-test": {}}})

    return run


def test_labelling_resumes_and_shows_only_the_message(tmp_path):
    prompts: list[str] = []
    out = tmp_path / "labels.jsonl"
    items = sample_rows()
    labellers.run_labeller(labellers.LABELLER_A, items[:20], out, fake_cli(prompts), 1, print)
    labellers.run_labeller(labellers.LABELLER_A, items, out, fake_cli(prompts), 1, print)
    rows = validation.read_jsonl(out)
    assert sorted(r["id"] for r in rows) == sorted(i["id"] for i in items)
    assert len(prompts) == 3  # 20 first, then the remaining 25 in two batches
    assert not any("terminus" in p or "gpt-" in p for p in prompts)


def test_a_bad_message_is_isolated_by_splitting(tmp_path):
    def run(argv, stdin):
        if '"m3"' in stdin and stdin.count("<message id=") > 1:
            return json.dumps({"result": "I can't label these.", "modelUsage": {}})
        return fake_cli([])(argv, stdin)

    out = tmp_path / "labels.jsonl"
    labellers.run_labeller(labellers.LABELLER_B, sample_rows()[:8], out, run, 1, print)
    assert len(validation.read_jsonl(out)) == 8


def test_a_prompt_cannot_change_silently():
    assert labellers.LABELLER_A.prompt_sha != labellers.LABELLER_B.prompt_sha


def test_a_rule_change_voids_the_validation_sample():
    with pytest.raises(ValueError, match="changed after"):
        study.validation_summary([{"rules_sha": "old"}], "new")


def test_population_weights_change_precision():
    sample = pd.DataFrame(
        {
            "stratum": ["a"] * 4 + ["b"] * 4,
            "stratum_n": [100] * 4 + [4] * 4,
            "adjudicated": ["claims-done"] * 4 + ["hedged"] * 4,
            "rule_label": ["claims-done"] * 4 + ["claims-done"] * 2 + ["hedged"] * 2,
        }
    )
    result = validation.rule_accuracy(sample)
    assert result["sample"]["claims-done"]["precision"] == pytest.approx(4 / 6)
    assert result["population"]["claims-done"]["precision"] == pytest.approx(100 / 102)


def test_the_detector_keeps_tasks_disjoint_and_reports_cells(tmp_path):
    words = ["alpha beta", "gamma delta"]
    rows = [
        {
            "task": f"t{i % 10}",
            "scaffold": "s",
            "model": "m",
            "failed": i % 2,
            "closing": f"{words[i % 2]} token{i % 7}",
            "tail": f"{words[(i + 1) % 2]} step{i % 5}",
        }
        for i in range(200)
    ]
    result = detector.run(pd.DataFrame(rows), tmp_path, lambda _: None)
    assert result["closing"]["auroc"] == pytest.approx(1.0)
    assert result["ids"]["auroc"] == pytest.approx(0.5)
    assert result["closing"]["within_cells"] == 1
