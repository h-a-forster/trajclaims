"""Labeller fixtures per class. Wording is invented, in the shape the scaffolds produce."""

import pytest

from trajclaims.rules import label

CLAIMS = [
    "Done! The program is at `/app/gpt2.c` and compiles with gcc.",
    "Analysis: The file exists and has the right format.\nPlan: The task is complete.",
    "THOUGHT: All requirements have been met.\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
    "I've created `/app/run.py`. It limits concurrency with a semaphore.",
    "Nginx is now configured and running on port 8080.",
    "Re-ran verification: 40 tests passed, 0 failed.",
]
HEDGED = [
    "I have created the file, but I could not run it because python is missing.",
    "Done. The OCR captured most commands and the file is written; the vast majority are right.",
    "The script is complete, however two test cases still fail on large inputs.",
    "I wrote the answer to /app/out.txt. It is a best guess from partial data.",
]
FAILURES = [
    "I must conclude that I cannot complete the task under the current conditions.",
    "I'm sorry, but I can't help with that.",
    "We were unable to solve the challenge; the file is empty.",
    "I cannot install primer3 because sudo is missing. Can you suggest another method?",
    "The task is not complete: the attack does not recover the key.",
    "Request timed out",
]
SILENT = [
    "Let me check the output first.",
    "Here are some thoughts about why you want to perform the action.\n\n```bash\necho hi\n```",
    "The trick is that a backslash at the end of a comment continues it in C++.",
]


@pytest.mark.parametrize("text", CLAIMS)
def test_claims(text):
    assert label(text)[0] == "claims-done"


@pytest.mark.parametrize("text", HEDGED)
def test_hedged(text):
    assert label(text)[0] == "hedged"


@pytest.mark.parametrize("text", FAILURES)
def test_failures(text):
    assert label(text)[0] == "reports-failure"


@pytest.mark.parametrize("text", SILENT)
def test_silent(text):
    assert label(text)[0] == "no-statement"


def test_code_blocks_do_not_vote():
    text = "Let me look.\n```bash\necho 'task is complete'\n```"
    assert label(text)[0] == "no-statement"
