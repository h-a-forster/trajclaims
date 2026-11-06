"""End to end on a small parquet shaped like the dataset; offline."""

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trajclaims import (
    cli,
    data,
    detector,
    labellers,
    models_effects,
    report,
    stats,
    study,
    validation,
)

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
    return [{"id": f"m{i}", "text": f"message {i}"} for i in range(45)]


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


def test_a_lone_message_is_retried(tmp_path):
    calls = []

    def run(argv, stdin):
        calls.append(stdin)
        if len(calls) == 1:
            return json.dumps({"result": '{"labels": {"m1": "claims-done"}}', "modelUsage": {}})
        return fake_cli([])(argv, stdin)

    out = tmp_path / "labels.jsonl"
    labellers.run_labeller(labellers.LABELLER_A, sample_rows()[:1], out, run, 1, print)
    assert len(calls) == 2 and len(validation.read_jsonl(out)) == 1


def test_a_corrected_answer_is_read_from_its_last_object():
    answer = '{"labels": {"m1": "hedged"}}\n\nWait, the id is "m7".\n\n{"labels": {"m7": "hedged"}}'
    envelope = json.dumps({"result": answer, "modelUsage": {}})
    assert labellers._parse(envelope, ["m7"])[0] == {"m7": "hedged"}


def test_planted_tokens_are_redacted_before_the_sample_is_committed():
    text = validation.redact("Replaced hf_" + "a" * 12 + " and AKIA" + "B" * 16 + " in config.")
    assert text == "Replaced [redacted-token] and [redacted-token] in config."


def test_a_prompt_cannot_change_silently():
    assert labellers.LABELLER_A.prompt_sha != labellers.LABELLER_B.prompt_sha


def test_a_rule_change_voids_the_validation_sample():
    with pytest.raises(ValueError, match="changed after"):
        study.validation_summary([{"rules_sha": "old"}], "new", pd.DataFrame())


def test_population_weights_change_precision():
    sample = pd.DataFrame(
        {
            "stratum": ["a"] * 4 + ["b"] * 4,
            "stratum_n": [100] * 4 + [4] * 4,
            "reward": 0,
            "adjudicated": ["claims-done"] * 4 + ["hedged"] * 4,
            "rule_label": ["claims-done"] * 4 + ["claims-done"] * 2 + ["hedged"] * 2,
        }
    )
    result = validation.rule_accuracy(sample)
    assert result["sample"]["claims-done"]["precision"] == pytest.approx(4 / 6)
    assert result["population"]["claims-done"]["precision"] == pytest.approx(100 / 102)


def detector_frame():
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
    return pd.DataFrame(rows)


def test_fresh_and_cached_detector_scores_are_the_same_numbers(tmp_path):
    unit = tmp_path / "scores.json"
    fresh = detector._cached(unit, lambda: np.array([1 / 3, 2 / 3]))
    assert np.array_equal(fresh, detector._cached(unit, lambda: np.array([9.0])))


def test_changed_detector_inputs_do_not_reuse_old_scores(tmp_path):
    frame = detector_frame()
    assert detector.run(frame, tmp_path, lambda _: None)["closing"]["auroc"] == pytest.approx(1.0)
    frame["failed"] = 1 - frame["failed"]  # stale scores would now rank exactly backwards
    assert detector.run(frame, tmp_path, lambda _: None)["closing"]["auroc"] == pytest.approx(1.0)


def test_the_detector_keeps_tasks_disjoint_and_reports_cells(tmp_path):
    result = detector.run(detector_frame(), tmp_path, lambda _: None)
    assert result["closing"]["auroc"] == pytest.approx(1.0)
    assert result["ids"]["auroc"] == pytest.approx(0.5)
    assert result["closing"]["within_cells"] == 1


def study_trials(tmp_path, monkeypatch):
    names = [f"t{i}" for i in range(12)]
    monkeypatch.setattr(data, "load_tasks", lambda: {n: {"agent_timeout_s": 900.0} for n in names})
    texts = ["Done. Wrote the file.", "I cannot complete the task.", "Let me check."]
    rows = []
    for i in range(1200):
        tool = [{"fn": "shell", "cmd": "pytest" if i % 5 == 0 else "ls"}]
        steps = [
            {"src": "agent", "msg": "", "tools": tool, "obs": "ok"},
            {"src": "agent", "msg": texts[i % 7 % 3], "tools": None, "obs": None},
        ]
        rows.append(
            row(
                task_name=names[i % 12],
                agent=("codex", "claude-code")[i % 2],
                model=("gpt-5@openai", "gpt-5-mini@openai")[i % 3 % 2],
                reward=int(i % 5 == 1),
                trial_name=f"trial{i}",
                steps=json.dumps(steps),
            )
        )
    return data.load_trials(built(tmp_path, rows))


def write_validation(path, trials):
    """A sample as `trajclaims label` writes it, with the rules as the adjudicated truth."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = validation.sample_rows(study.assign_labels(trials), study.rules_sha())
    for r in rows:
        r.update(
            label_a=r["rule_label"],
            label_b=r["rule_label"],
            adjudicated=r["rule_label"],
            adjudication="agreed",
            models=["test"],
            prompt_shas=["test"],
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


def test_study_outputs_regenerate_byte_identically(tmp_path, monkeypatch):
    trials, counts = study_trials(tmp_path, monkeypatch)
    outputs = []
    for run in ("one", "two"):  # the second run reads the detector units the first wrote
        out = tmp_path / run
        write_validation(out / "validation.jsonl", trials)
        study.run_study(trials, counts, out, tmp_path / "units-d", log=lambda _: None)
        report.write_tables(out)
        outputs.append({p.name: p.read_bytes() for p in out.iterdir()})
    assert outputs[0] == outputs[1]
    assert set(outputs[0]) >= {"labels.jsonl", "fcr.csv", "summary.json", "contrasts.csv"}
    summary = json.loads(outputs[0]["summary.json"])
    assert summary["validation"]["direct_fcr_readable"]["all"]["fcr"] == pytest.approx(
        summary["validation"]["direct_fcr_readable"]["all"]["rule_rate_sample"]
    )
    tables = outputs[0]["tables.md"].decode("utf-8")
    assert "## All scaffolds" in tables and "| all | " in tables


def test_a_stale_validation_sample_stops_the_run_before_anything_is_written(tmp_path, monkeypatch):
    trials, counts = study_trials(tmp_path, monkeypatch)
    out = tmp_path / "out"
    rows = write_validation(out / "validation.jsonl", trials)
    lines = (out / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    (out / "validation.jsonl").write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    assert len(rows) == len(lines)
    with pytest.raises(data.DataError, match="sample --extend"):
        study.run_study(trials, counts, out, tmp_path / "d", log=lambda _: None)
    assert not (out / "labels.jsonl").exists()


def test_extending_a_sample_adds_strata_and_refuses_a_changed_one():
    def sampled(i, stratum):
        return {"id": i, "stratum": stratum, "stratum_n": 9, "rule_label": "hedged", "text": i}

    held = [sampled("a", "s|0")]
    fresh = [sampled("a", "s|0"), sampled("b", "s+command|0")]
    assert [r["id"] for r in validation.extend(held, fresh)] == ["a", "b"]
    moved = [sampled("c", "s|0"), sampled("b", "s+command|0")]
    with pytest.raises(data.DataError):
        validation.extend(held, moved)


def test_a_new_stratum_draws_apart_and_leaves_the_old_one_alone():
    def frame(flags):
        return pd.DataFrame(
            {
                "task": "t",
                "scaffold": "s",
                "model_raw": "m",
                "trial": [f"x{i}" for i in range(len(flags))],
                "reward": 0,
                "status": "readable",
                "dev": False,
                "with_command": flags,
            }
        )

    plain = validation.draw(frame([False] * 100))
    mixed = validation.draw(frame([False] * 100 + [True] * 100))
    assert set(mixed["stratum"]) == {"s|0", "s+command|0"}
    assert set(plain["id"]) == set(mixed.loc[mixed["stratum"] == "s|0", "id"])
    assert set(mixed["stratum_n"]) == {100}


def test_failed_only_precision_ignores_passed_messages():
    sample = pd.DataFrame(
        {
            "stratum": ["a|0"] * 2 + ["a|1"] * 2,
            "stratum_n": [10] * 4,
            "reward": [0, 0, 1, 1],
            "adjudicated": ["claims-done", "hedged", "claims-done", "claims-done"],
            "rule_label": ["claims-done"] * 4,
        }
    )
    result = validation.rule_accuracy(sample)
    assert result["population"]["claims-done"]["precision"] == pytest.approx(0.75)
    assert result["population_failed"]["claims-done"]["precision"] == pytest.approx(0.5)


def test_paths_do_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(data, "fetch", lambda raw, log: seen.append(raw))
    monkeypatch.chdir(tmp_path)
    cli.main(["fetch"])
    assert seen == [data.TASKS_FILE.parents[1] / "data" / "raw"]


def rated(**over):
    """Labelled rows for hand-checked rates. Each group spans two tasks for the bootstrap."""
    base = {
        "scaffold": "a",
        "model": "m",
        "closing_group": "a",
        "reward": 0,
        "status": "readable",
        "ending": "stopped",
    }
    return {**base, **over}


def labelled_table(groups):
    rows = []
    for i, (n, fields) in enumerate(groups):
        rows += [{**rated(**fields), "task": f"t{(i + j) % 2}"} for j in range(n)]
    return pd.DataFrame(rows)


COUNTS = {"true_pos": 9, "pos": 10, "true_neg": 8, "neg": 10, "source": "scaffold"}


def rates_at(level, frame, rates, bridge=None):
    weights = stats.task_weights(frame["task"].nunique(), 300, 1)
    table, _ = study.rate_table(frame, level, weights, rates, bridge or {"*": 0.0})
    return table.set_index("group")


def test_rates_match_a_hand_count():
    frame = labelled_table(
        [
            (6, {"label": "claims-done", "ending": "declared"}),
            (2, {"label": "hedged"}),
            (2, {"label": "no-statement", "status": "no-text"}),
            (2, {"label": "", "status": "lost"}),
            (3, {"label": "claims-done", "reward": 1}),
            (1, {"label": "hedged", "reward": 1}),
        ]
    )
    row = rates_at("scaffold", frame, {"*": COUNTS, "a": COUNTS}, {"*": 0.0, "a": 0.5}).loc["a"]
    assert row["fcr_readable"] == pytest.approx(6 / 8)
    assert row["fcr"] == pytest.approx(6 / 10)
    assert row["fcr_if_lost_silent"] == pytest.approx(6 / 12)
    assert row["fcr_if_lost_claim"] == pytest.approx(8 / 12)
    assert row["fcr_bridged"] == pytest.approx(7 / 12)
    assert row["claim_rate_passed_readable"] == pytest.approx(3 / 4)
    assert row["false_claim_share"] == pytest.approx(6 / 9)
    # With sensitivity 0.9 and specificity 0.8, and rule counts PPV 9/11, NPV 8/9.
    assert row["fcr_readable_corrected"] == pytest.approx((0.75 + 0.8 - 1) / (0.9 + 0.8 - 1))
    assert row["fcr_readable_calibrated"] == pytest.approx(0.75 * 9 / 11 + 0.25 * 1 / 9)
    assert row["fcr_readable_low"] <= 0.75 <= row["fcr_readable_high"]


def test_the_correction_is_clipped_after_aggregating_not_per_cell():
    frame = labelled_table(
        [
            (4, {"label": "claims-done", "model": "m"}),
            (2, {"label": "claims-done", "model": "n"}),
            (2, {"label": "hedged", "model": "n"}),
        ]
    )
    table = rates_at("scaffold", frame, {"*": COUNTS, "a": COUNTS})
    assert table.loc["a", "fcr_readable_corrected"] == pytest.approx(0.55 / 0.7)


def test_correction_draws_centre_on_the_point_estimate():
    many = {"true_pos": 900, "pos": 1000, "true_neg": 800, "neg": 1000, "source": "scaffold"}
    claims = [(30, {"label": "claims-done"}), (10, {"label": "hedged"})]
    frame = labelled_table(claims * 4)
    weights = stats.task_weights(2, 2000, 1)
    block = study._corrected_cells(frame, {"*": many, "a": many}, weights, 7)
    draws = block["count_draws"][:, 0] / block["seen_draws"][:, 0]
    calibrated = block["calibrated_draws"][:, 0] / block["seen_draws"][:, 0]
    assert np.median(draws) == pytest.approx(0.55 / 0.7, abs=0.02)
    assert np.median(calibrated) == pytest.approx(block["calibrated"][0] / 160, abs=0.02)


def test_each_closing_group_is_corrected_with_its_own_error_rates():
    command = {"true_pos": 5, "pos": 10, "true_neg": 10, "neg": 10, "source": "scaffold"}
    frame = labelled_table(
        [
            (3, {"label": "claims-done"}),
            (1, {"label": "hedged"}),
            (2, {"label": "claims-done", "closing_group": "a+command"}),
            (2, {"label": "hedged", "closing_group": "a+command"}),
        ]
    )
    rates = {"*": COUNTS, "a": COUNTS, "a+command": command}
    row = rates_at("scaffold", frame, rates).loc["a"]
    plain, mixed = (0.75 + 0.8 - 1) / 0.7, 0.5 / 0.5
    assert row["fcr_readable_corrected"] == pytest.approx((4 * plain + 4 * mixed) / 8)
    assert row["failed_readable"] == 8


def test_direct_rates_weight_each_stratum_by_its_size():
    sample = pd.DataFrame(
        {
            "stratum": ["a|0"] * 2 + ["a+command|0"] * 2,
            "stratum_n": [30] * 2 + [10] * 2,
            "scaffold": "a",
            "reward": 0,
            "adjudicated": ["claims-done", "claims-done", "claims-done", "hedged"],
            "rule_label": ["claims-done", "claims-done", "hedged", "hedged"],
        }
    )
    labelled = pd.DataFrame(
        {"scaffold": ["a"] * 2, "reward": 0, "status": "readable", "dev": False}
        | {"label": ["claims-done", "hedged"]}
    )
    block = study._direct(sample, labelled)["all"]
    assert block["fcr"] == pytest.approx((30 * 1 + 10 * 0.5) / 40)
    assert block["rule_rate_sample"] == pytest.approx(0.75)
    assert block["rule_rate_population"] == pytest.approx(0.5)
    assert block["effective_n"] == pytest.approx(40**2 / (2 * 15**2 + 2 * 5**2))


def test_the_bridge_measures_lost_text_and_relabels_kept_pairs():
    def twin(status, partial, label, text="", full=""):
        return {
            "scaffold": "openhands",
            "reward": 0,
            "status": status,
            "partial_status": partial,
            "label": label,
            "partial_text": text,
            "closing_text": full,
        }

    frame = pd.DataFrame(
        [
            twin("readable", "lost", "claims-done"),
            twin("readable", "lost", "hedged"),
            twin("readable", "readable", "claims-done", "Done. Wrote the file."),
            twin("readable", "readable", "claims-done", "Let me check."),
        ]
    )
    block = study.bridge(frame)
    assert block["claim_rate_lost"] == pytest.approx(0.5)
    assert block["paired_label_agreement"] == pytest.approx(0.5)


def test_shares_credit_the_factor_that_moves_the_cells():
    rows = []
    for s, rate in (("s1", 0.2), ("s2", 0.5), ("s3", 0.8)):
        for model in ("m1", "m2", "m3"):
            for i in range(50):
                rows.append(
                    {
                        "scaffold": s,
                        "model": model,
                        "task": f"t{i % 10}",
                        "reward": 0,
                        "status": "readable",
                        "label": "claims-done" if i % 10 < rate * 10 else "hedged",
                    }
                )
    block = models_effects.model_vs_scaffold(pd.DataFrame(rows), 40, 1, readable=True)["raw"]
    assert block["scaffold_share"] == pytest.approx(1.0)
    assert block["model_share"] == pytest.approx(0.0, abs=1e-9)
    assert block["scaffold_minus_model"] == pytest.approx(1.0)


def test_an_odds_ratio_interval_surrounds_its_estimate():
    y = np.array([1, 0] * 20 + [1, 1, 1, 0] * 10, dtype=float)
    frame = pd.DataFrame(
        {"y": y, "x": [0.0] * 40 + [1.0] * 40, "task": [f"t{i % 8}" for i in range(80)]}
    )
    block = models_effects._odds(frame, ["x"], [])["x"]
    assert block["ci"][0] < block["odds_ratio"] < block["ci"][1]
    assert block["odds_ratio"] == pytest.approx(3.0)


def test_bridge_rates_keep_code_heavy_closings_out_of_the_prose_rate():
    def bridged(scaffold, claims, readable):
        return {
            "scaffold": scaffold,
            "reward": 0,
            "partial_status": "lost",
            "readable_in_full": readable,
            "claims_done": claims,
        }

    rates = study._bridge_rates(
        [
            bridged("mini-swe-agent", 1, 100),
            bridged("openhands", 80, 100),
            bridged("terminus-2", 60, 100),
        ]
    )
    assert rates["mini-swe-agent"] == pytest.approx(0.01)
    assert rates["*"] == pytest.approx(0.7)
