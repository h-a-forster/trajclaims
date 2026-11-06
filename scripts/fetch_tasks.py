"""Rebuild configs/tasks.json from the Terminal-Bench 2 task repository at a pinned commit."""

from __future__ import annotations

import json
import re
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "harbor-framework/terminal-bench-2"
COMMIT = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"
OUT = Path(__file__).resolve().parent.parent / "configs" / "tasks.json"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - fixed https URLs
        return response.read()


# Checks a task ships to the agent; tests/ at the task root reaches only the verifier.
OWN_CHECK = re.compile(r"(?:^|/)(?:test_outputs\.py|eval\.py|check\.py)$")


def _task(name: str) -> tuple[str, dict[str, object]]:
    raw = _get(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}/task.toml")
    meta = tomllib.loads(raw.decode("utf-8"))
    return name, {
        "agent_timeout_s": meta["agent"]["timeout_sec"],
        "category": meta["metadata"]["category"],
        "difficulty": meta["metadata"]["difficulty"],
    }


def main() -> None:
    tree = json.loads(_get(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    names = sorted(
        item["path"].split("/")[0]
        for item in tree["tree"]
        if item["path"].count("/") == 1 and item["path"].endswith("/task.toml")
    )
    with ThreadPoolExecutor(8) as pool:
        tasks = dict(pool.map(_task, names))
    for item in tree["tree"]:
        task, _, rest = item["path"].partition("/")
        if rest.startswith("environment/") and OWN_CHECK.search(rest):
            tasks[task].setdefault("own_check", []).append(rest.removeprefix("environment/"))
    body = {"repository": REPO, "commit": COMMIT, "tasks": tasks}
    OUT.write_text(
        json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{len(tasks)} tasks written to {OUT.name}")


if __name__ == "__main__":
    main()
