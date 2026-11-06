"""Find what a trajectory ends with: how it ended, and its closing statement if one survives.

`steps` come from a scrape of a React Server Components payload. Strings of 1024 characters or
more were left as `$<hex>` references to chunks the scrape dropped, so the longest closing
messages are gone. Each scaffold also ends differently; the rules are in docs/DESIGN.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# A string the scrape replaced with a reference to a dropped chunk.
REF = re.compile(r"\s*\$[0-9a-f]+\s*")
EMBEDDED_REF = re.compile(r"(?<![\w$])\$[0-9a-f]{1,5}(?![\w])")

COMPLETION_TOOLS = frozenset({"mark_task_complete", "finish"})
SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
# These scaffolds leave their loop only on a completion signal; any other end is a harness stop.
SIGNAL_SCAFFOLDS = frozenset({"terminus-2", "terminus-3-3", "mini-swe-agent"})
# OpenHands' own text on a `finish` event; the agent's words are in the tool argument. The
# partial scrape blanks that argument, so a `finish` with no text is a lost message, not silence.
BOILERPLATE = frozenset({"All done! What's next on the agenda?"})


@dataclass(frozen=True, slots=True)
class Closing:
    """`ending` is structural; `status` says whether a closing statement can be read.

    `with_command` marks text from a step that also ran commands (terminus-2 may send
    `mark_task_complete` beside them), so the validation can sample those closings apart.
    """

    ending: str
    status: str
    text: str = ""
    signal: str = ""
    with_command: bool = False


def tool_cmd(tool: dict[str, Any]) -> str:
    """A tool call's argument as text (Codex stores an argv list, others an object or number)."""
    cmd = tool.get("cmd")
    if cmd is None:
        return ""
    if isinstance(cmd, str):
        return cmd
    return " ".join(str(part) for part in cmd) if isinstance(cmd, list) else json.dumps(cmd)


def _signal(step: dict[str, Any]) -> str:
    for tool in step.get("tools") or []:
        fn = tool.get("fn") or ""
        if fn in COMPLETION_TOOLS:
            return fn
        if SUBMIT_SENTINEL in tool_cmd(tool):
            return "submit"
    return ""


def _only_closing(step: dict[str, Any]) -> bool:
    tools = step.get("tools") or []
    return not tools or all(
        t.get("fn") in COMPLETION_TOOLS or SUBMIT_SENTINEL in tool_cmd(t) for t in tools
    )


def _finish_message(step: dict[str, Any]) -> str:
    for tool in step.get("tools") or []:
        if tool.get("fn") != "finish":
            continue
        cmd = tool_cmd(tool).strip()
        if not cmd:
            return ""
        try:
            args = json.loads(cmd)
        except ValueError:
            return cmd
        return str(args.get("message") or "") if isinstance(args, dict) else cmd
    return ""


def closing(scaffold: str, steps: list[dict[str, Any]], timed_out: bool = False) -> Closing:
    """Classify the end of one trajectory; pure, so labels regenerate byte for byte.

    `timed_out` means the trial ran to its task's agent timeout. Without a completion signal the
    last message is then mid-work, whatever its step looks like.
    """
    agent = [s for s in steps if s.get("src") == "agent"]
    if not agent:
        return Closing("none", "no-text")
    last = agent[-1]
    signal = _signal(last)
    tools = last.get("tools") or []
    if signal:
        ending = "declared"
    elif (
        scaffold == "mini-swe-agent"
        and not timed_out
        and any(REF.fullmatch(tool_cmd(t)) for t in tools)
    ):
        # Its submit sentinel is a shell command, which could be the lost one.
        return Closing("unknown", "lost")
    elif tools or timed_out or scaffold in SIGNAL_SCAFFOLDS:
        return Closing("cutoff", "cutoff", (last.get("msg") or "").strip())
    else:
        ending = "stopped"
    for step in reversed(agent):
        only_closing = _only_closing(step)
        if not only_closing and not _signal(step):
            break
        for candidate in (_finish_message(step), step.get("msg") or ""):
            text = candidate.strip()
            if not text or text in BOILERPLATE:
                continue
            if REF.fullmatch(text):
                return Closing(ending, "lost", signal=signal)
            return Closing(ending, "readable", text, signal, not only_closing)
    return Closing(ending, "lost" if signal == "finish" else "no-text", signal=signal)
