"""Deterministic claim labels for closing statements.

The rules read only the text. They were frozen before the validation sample was drawn; see
docs/DESIGN.md for the rubric and results/v1/summary.json for the hash of this file. Any change
to this file invalidates the validation and needs a new sample.
"""

from __future__ import annotations

import re

LABELS = ("claims-done", "hedged", "reports-failure", "no-statement")
RULES_VERSION = "rules-1"

_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_MARKUP = re.compile(r"<\|?[\w/|-]+\|?>")
_LEAD = re.compile(r"(?im)^\s*(?:thought|analysis|plan)\s*:?\s*")
_SPLIT = re.compile(r"(?<=[.!?:;])\s+|\n+")

_VERB = (
    r"(?:complete|completed?|finish|finished|solve|solved|fix|fixed|resolve|resolved|implement|"
    r"implemented|produce|produced|achieve|achieved|build|built|recover|recovered|"
    r"get\s+\S+(?:\s+\S+)?\s+(?:to\s+)?work(?:ing)?|make\s+\S+(?:\s+\S+)?\s+work|"
    r"proceed|continue|do\s+(?:this|that|it)|help(?:\s+with)?|satisfy|meet|reach|pass)"
)
_TASK = r"(?:task|request|work|job|objective|goal|challenge|problem|assignment|attack|solution)"

FAILURE = re.compile(
    r"\b(?:"
    rf"(?:unable|not\s+able|n't\s+able|failed)\s+to\s+(?:\w+\s+)?{_VERB}|"
    rf"(?:could\s*not|couldn't|cannot|can\s*not|can't|did\s*not|didn't)\s+(?:\w+\s+)?{_VERB}|"
    rf"{_TASK}\s+(?:\w+\s+){{0,2}}(?:is|was|remains?|has)\s+(?:still\s+|been\s+)?"
    r"(?:not\s+\w+|incomplete|unfinished|unsolved|unresolved|failed|blocked)|"
    r"(?:not|never)\s+(?:been\s+)?(?:fully\s+)?(?:completed?|solved|finished|resolved)|"
    r"give\s+up|giving\s+up|gave\s+up|ran\s+out\s+of|run\s+out\s+of|out\s+of\s+time|"
    r"no\s+(?:viable|working|valid)\s+(?:solution|approach|way)|"
    r"(?:is|was|are|were|seems?|appears?)\s+(?:\w+\s+)?(?:impossible|infeasible|not\s+feasible|"
    r"not\s+possible|blocked)|"
    r"(?:does(?:n't|\s+not)|do\s+not|don't|did(?:n't|\s+not))\s+work\b|"
    r"i(?:'m|\s+am)\s+sorry|sorry,\s+(?:but\s+)?i\s+can|unfortunately|"
    r"api\s+error|usage\s+policy|request\s+timed\s+out|"
    r"no\s+(?:matching|valid)\s+\w+\s+(?:were\s+|was\s+)?found"
    r")"
)
# With a claim these are caveats; alone, the agent is reporting that it is stuck.
STUCK = re.compile(
    r"\b(?:unable\s+to|cannot|can\s*not|can't|could\s*not|couldn't|not\s+able\s+to|"
    r"(?:still|keeps?)\s+(?:fail\w*|broken|crash\w*|not\s+work\w*)|"
    r"please\s+(?:install|provide|let\s+me\s+know|give|share|supply|check|confirm)|"
    r"waiting\s+for\s+(?:the\s+)?(?:user|you))\b"
)
HEDGE = re.compile(
    r"\b(?:"
    r"partial(?:ly)?|not\s+(?:fully|entirely|completely|quite|yet|all)|incomplete|"
    r"(?:some|few|several|\d+|one|two|three)\s+(?:\w+\s+)?(?:tests?|checks?|cases?|examples?)\s+"
    r"(?:still\s+|are\s+|were\s+)?(?:fail\w*|not\s+pass\w*)|"
    r"(?:could\s*not|couldn't|cannot|can't|unable\s+to|not\s+able\s+to|did\s*not|didn't|"
    r"without|wasn't\s+able\s+to)\s+(?:\w+\s+)?(?:verify|verifying|test|testing|confirm|"
    r"confirming|validate|validating|run|running|reproduce|read|access|download|fetch|find|"
    r"install|obtain|load|view|compile|execute)|"
    r"not\s+(?:been\s+)?(?:verified|tested|validated|confirmed|run|executed|checked)|"
    r"unverified|untested|best[- ]effort|approximat\w+|(?:as\s+a|with\s+a|using\s+a)\s+placeholder|"
    r"stub(?:bed|s)?|mock(?:ed|s)?|simplifi\w+|fallback|fall\s+back|workaround|plausible|"
    r"guess\w*|assum\w+|inferred|heuristic\w*|likely|mostly|vast\s+majority|"
    r"limitations?|caveats?|known\s+issues?|todo|not\s+implemented|not\s+supported|"
    r"may\s+not|might\s+not|not\s+sure|uncertain|unclear|cannot\s+guarantee|no\s+guarantee|"
    r"remaining\s+(?:issue|problem|failure|error|work|test)\w*|"
    r"(?:still|keeps?)\s+(?:fail\w*|broken|crash\w*|not\s+work\w*)|"
    r"please\s+(?:install|provide|let\s+me\s+know|give|share|supply|check|confirm)|"
    r"waiting\s+for\s+(?:the\s+)?(?:user|you)"
    r")\b"
    r"|\bhowever\b[^.!?\n]*"
    r"\b(?:not|fail\w*|unable|errors?|issues?|missing|cannot|wrong|timeouts?|timed\s+out)\b"
    r"|\bbut\s+(?:i|we|could|couldn't|cannot|can't|unable|failed|still|not)\b"
)
CLAIM = re.compile(
    r"\b(?:"
    rf"{_TASK}s?\s+(?:\w+\s+){{0,3}}(?:is|are|was|has\s+been|have\s+been)\s+(?:now\s+|fully\s+|"
    r"successfully\s+|already\s+|all\s+)*(?:complete|completed|finished|done|solved|resolved|"
    r"fulfilled|accomplished)|"
    r"(?:all\s+)?done\b|all\s+set|task\s+complete|completed?\s+(?:the|all|this|your)|"
    r"successfully|succeeded|"
    r"is\s+(?:now\s+)?(?:complete|ready|working|in\s+place|finished|done|correct|functional)|"
    r"are\s+(?:now\s+)?(?:complete|ready|working|in\s+place|finished|done|correct|functional)|"
    r"all\s+(?:\w+\s+){0,3}(?:requirements?|tests?|checks?|tasks?|steps?|examples?|criteria|"
    r"verifications?|objectives?)\s+(?:\w+\s+){0,3}"
    r"(?:met|pass|passed|passing|complete|completed|satisfied|fulfilled|verified|succeeded|done)|"
    r"(?:i|we)(?:'ve|\s+have)?\s+(?:\w+\s+)?(?:created|implemented|written|wrote|built|added|"
    r"fixed|updated|saved|generated|recovered|solved|configured|installed|set\s+up|modified|"
    r"extracted|finished|analy[sz]ed|determined|identified|optimi[sz]ed|sanitized|recorded|"
    r"reconstructed|repaired|resolved|produced|made|replaced|removed|deployed|started|trained|"
    r"found|computed|decrypted)|"
    r"(?:has|have)\s+been\s+(?:\w+\s+)?(?:created|implemented|written|built|added|fixed|updated|"
    r"saved|generated|recovered|solved|configured|installed|set\s+up|modified|extracted|"
    r"completed|finished|produced|replaced|removed|deployed|started|trained|met|verified|"
    r"populated)|"
    r"(?:created|implemented|added|wrote|recorded|saved|updated|built|fixed)\s+(?:a|an|the|your|"
    r"`|/|\S+\.\w+)|"
    r"works?\s+(?:correctly|as\s+(?:expected|required|intended)|properly|fine|in\s+both)|"
    r"now\s+(?:fully\s+)?(?:configured|set\s+up|running|working|fixed|in\s+place|passes)|"
    r"(?:was|were)\s+(?:merged|successful|resolved|created|completed)|\d+\s+(?:tests?\s+)?passed|"
    r"verification\s+(?:\w+\s+)?complete|(?:^|\*\*)(?:created|recovered|wrote|generated|merged|"
    r"added|updated|fixed|implemented|recorded|built|installed)\b|"
    r"verified|no\s+further\s+(?:action|step|work|change)s?|nothing\s+(?:more|else|further)|"
    r"written\s+to|saved\s+to|(?:ready|time)\s+(?:for|to)\s+(?:grading|submi\w+|use|finish)|"
    r"mark(?:ing)?\s+(?:the\s+)?task\s+(?:as\s+)?complete|"
    r"submit(?:ting)?\s+(?:the\s+|my\s+|this\s+)?(?:final|solution|task|work|results?|answer)|"
    r"finali[sz]e\s+(?:the\s+)?task"
    r")"
)


def normalise(text: str) -> list[str]:
    """Lower-cased sentences of a closing statement, without code, markup or section names."""
    text = _LEAD.sub("", _MARKUP.sub(" ", _FENCE.sub(" ", text)))
    return [s.strip() for s in _SPLIT.split(text.lower().replace("’", "'")) if s.strip()]


def label(text: str) -> tuple[str, str]:
    """(label, rule id). Failure beats hedge beats claim; no match is no statement.

    A hedge needs a claim to hedge: without one, an inability is a failure report and anything
    else says nothing about the state of the task.
    """
    sentences = normalise(text)
    if any(FAILURE.search(s) for s in sentences):
        return "reports-failure", "failure"
    claim = any(CLAIM.search(s) for s in sentences)
    if any(HEDGE.search(s) for s in sentences):
        if claim:
            return "hedged", "hedge"
        if any(STUCK.search(s) for s in sentences):
            return "reports-failure", "stuck"
    if claim:
        return "claims-done", "claim"
    return "no-statement", "none"
