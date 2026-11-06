"""Command line: fetch, extract, sample, label, study, check."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from trajclaims import __version__, data, labellers, report, study, validation

RAW = data.ROOT / "data/raw"
UNITS = data.ROOT / "data/units"
SAMPLE = data.ROOT / "data/validation/sample.jsonl"
RAW_A = data.ROOT / "data/validation/label_a.jsonl"
RAW_B = data.ROOT / "data/validation/label_b.jsonl"
RAW_ADJ = data.ROOT / "data/validation/adjudicated.jsonl"
OUT = data.ROOT / "results/v1"
DETECTOR_UNITS = data.ROOT / "data/detector"


def _log(line: str) -> None:
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with (OUT / "progress.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{stamp} {line}\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8", newline="\n")


def cmd_fetch(_: argparse.Namespace) -> int:
    data.fetch(RAW, _log)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    data.extract(RAW, UNITS, _log, force=args.force)
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Draw the validation sample; a drawn one is never replaced, only extended by new strata."""
    if SAMPLE.exists() and not args.extend:
        raise SystemExit(f"{SAMPLE} exists: a sample is drawn once, after the rules are frozen")
    trials, _counts = data.load_trials(UNITS)
    sha = study.rules_sha()
    rows = validation.sample_rows(study.assign_labels(trials), sha)
    if SAMPLE.exists():
        rows = validation.extend(validation.read_jsonl(SAMPLE), rows)
    _write_jsonl(SAMPLE, rows)
    _log(f"sample: {len(rows)} messages, rules {sha[:12]}")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    """Blind labels, then adjudication of disagreements; only id and text reach a labeller."""
    rows = validation.read_jsonl(SAMPLE)
    items = [{"id": r["id"], "text": r["text"]} for r in rows]
    labellers.run_labeller(labellers.LABELLER_A, items, RAW_A, workers=args.workers, log=_log)
    labellers.run_labeller(labellers.LABELLER_B, items, RAW_B, workers=args.workers, log=_log)
    a, b = validation.read_jsonl(RAW_A), validation.read_jsonl(RAW_B)
    hints = validation.disagreements(a, b)
    text = {r["id"]: r["text"] for r in rows}
    queue = [{"id": i, "text": text[i], "hint": hint} for i, hint in sorted(hints.items())]
    labellers.run_labeller(labellers.ADJUDICATOR, queue, RAW_ADJ, workers=args.workers, log=_log)
    final = {r["id"]: r for r in validation.read_jsonl(RAW_ADJ)}
    left = {r["id"]: r for r in a}
    right = {r["id"]: r for r in b}
    merged = []
    for row in rows:
        i = row["id"]
        agree = left[i]["label"] == right[i]["label"]
        sources = [left[i], right[i]] + ([] if agree else [final[i]])
        merged.append(
            {
                **row,
                "text": validation.redact(row["text"]),
                "label_a": left[i]["label"],
                "label_b": right[i]["label"],
                "adjudicated": left[i]["label"] if agree else final[i]["label"],
                "adjudication": "agreed" if agree else "adjudicator",
                "models": sorted({m for s in sources for m in s["models"]}),
                "prompt_shas": sorted({s["prompt_sha"] for s in sources}),
            }
        )
    _write_jsonl(OUT / "validation.jsonl", merged)
    _log(f"validation: {len(merged)} messages, {len(hints)} adjudicated")
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    trials, counts = data.load_trials(UNITS)
    summary = study.run_study(
        trials,
        counts,
        OUT,
        DETECTOR_UNITS,
        with_validation=not args.no_validation,
        with_detector=not args.no_detector,
        log=_log,
    )
    report.write_tables(OUT)
    _log(f"study done: labels {summary['labels_sha256'][:12]}")
    return 0


def cmd_check(_: argparse.Namespace) -> int:
    """Oracle: labels regenerated from the pinned data must equal the committed bytes."""
    trials, _counts = data.load_trials(UNITS)
    fresh = study.labels_jsonl(study.assign_labels(trials))
    digest = hashlib.sha256(fresh).hexdigest()
    if fresh != (OUT / "labels.jsonl").read_bytes():
        raise SystemExit(f"labels differ from the committed file: fresh {digest}")
    print(f"labels identical: {digest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trajclaims", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="download the pinned dataset").set_defaults(func=cmd_fetch)
    extract = commands.add_parser("extract", help="reduce trials to one row each")
    extract.add_argument("--force", action="store_true", help="rebuild finished units")
    extract.set_defaults(func=cmd_extract)
    sample = commands.add_parser("sample", help="draw the validation sample")
    sample.add_argument("--extend", action="store_true", help="add strata the sample lacks")
    sample.set_defaults(func=cmd_sample)
    label = commands.add_parser("label", help="blind labels through the claude CLI")
    label.add_argument("--workers", type=int, default=4)
    label.set_defaults(func=cmd_label)
    run = commands.add_parser("study", help="regenerate results/v1")
    run.add_argument("--no-validation", action="store_true")
    run.add_argument("--no-detector", action="store_true")
    run.set_defaults(func=cmd_study)
    commands.add_parser("check", help="labels byte-identical to results").set_defaults(
        func=cmd_check
    )
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
