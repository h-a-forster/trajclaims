"""What an agent ran to check its work, from its visible commands."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from trajclaims.closing import REF, SUBMIT_SENTINEL, tool_cmd

VERIFICATION = ("none", "adhoc-check", "test-runner", "task-check")

# Tools that run shell commands, per scaffold. Editors, readers and planners are not checks.
SHELL_TOOLS = frozenset(
    {
        "bash_command",
        "execute_bash",
        "execute_ipython_cell",
        "Bash",
        "shell",
        "run_shell_command",
        "execute",
        "interact_with_shell",
    }
)
EDIT_TOOLS = frozenset(
    {"str_replace_editor", "Write", "Edit", "MultiEdit", "write_file", "edit_file", "replace"}
)
READONLY_EDITOR = frozenset({"view"})

# The verifier uploads tests/test.sh and tests/test_outputs.py only after the agent has finished,
# so a hit here means the agent went looking for them, or fetched them from outside.
TASK_CHECK = re.compile(r"test_outputs\.py|/tests/test\.sh|/tests/test_outputs")
TEST_RUNNER = re.compile(
    r"(?:^|[\s;&|(/])(?:"
    r"pytest|py\.test|tox|nox|ctest|rspec|phpunit|"
    r"python[\d.]*\s+-m\s+(?:unittest|pytest)|"
    r"npm\s+(?:run\s+)?test|yarn\s+test|pnpm\s+test|cargo\s+test|go\s+test|"
    r"make\s+(?:[\w=-]+\s+)*(?:test|tests|check)\b|mvn\s+test|gradle\s+test|dotnet\s+test|"
    r"(?:python[\d.]*|bash|sh|node|Rscript|\./|perl|ruby)\s*\S*(?:/|\b)"
    r"(?:test_\w+|\w+_test|tests?)(?:\.\w+)?(?=\s|$|[;&|)])"
    r")"
)
ADHOC_CHECK = re.compile(
    r"\bassert\b|\bdiff\s+\S|\bcmp\s+\S|sha256sum\s+-c|md5sum\s+-c|"
    r"(?:python[\d.]*|bash|sh|node|Rscript|\./)\s*\S*"
    r"(?:verif|valid|check|compare|sanity|smoke)\w*(?:\.\w+)?"
)
# Writes through the shell. `>` alone would catch `2>/dev/null`.
SHELL_EDIT = re.compile(
    r"(?<![\d&2])>{1,2}\s*(?!/dev/|&)[\w./~\"'-]|\bsed\s+-i|\btee\s+\S|\bpatch\s|\bgit\s+apply\b"
)


@dataclass(frozen=True, slots=True)
class Verification:
    level: str
    after_last_edit: bool
    shell_commands: int
    opaque_commands: int
    blank_commands: int


def verification(steps: list[dict[str, Any]]) -> Verification:
    """The strongest check among visible shell commands, and whether one followed the last edit.

    A command lost to the scrape cannot be read, so `level` is a lower bound; `opaque_commands`
    counts those. A lost command is usually a long heredoc, so it counts as an edit.
    """
    rank = {name: i for i, name in enumerate(VERIFICATION)}
    best = 0
    last_check = last_edit = -1
    shell = opaque = blank = 0
    for index, step in enumerate(s for s in steps if s.get("src") == "agent"):
        for tool in step.get("tools") or []:
            fn, cmd = tool.get("fn") or "", tool_cmd(tool)
            if fn in EDIT_TOOLS:
                if cmd.strip() not in READONLY_EDITOR:
                    last_edit = index
                continue
            if fn not in SHELL_TOOLS or SUBMIT_SENTINEL in cmd:
                continue
            shell += 1
            if not cmd.strip():
                blank += 1
                continue
            if REF.fullmatch(cmd):
                opaque += 1
                last_edit = index
                continue
            level = 0
            if TASK_CHECK.search(cmd):
                level = rank["task-check"]
            elif TEST_RUNNER.search(cmd):
                level = rank["test-runner"]
            elif ADHOC_CHECK.search(cmd):
                level = rank["adhoc-check"]
            if level:
                best = max(best, level)
                last_check = index
            if SHELL_EDIT.search(cmd):
                last_edit = max(last_edit, index)
    return Verification(
        VERIFICATION[best], last_check >= 0 and last_check >= last_edit, shell, opaque, blank
    )
