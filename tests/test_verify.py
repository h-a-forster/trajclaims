"""Verification levels from commands."""

import pytest

from trajclaims.verify import own_check_pattern, verification


def run(*commands, fn="bash_command"):
    return {"src": "agent", "tools": [{"fn": fn, "cmd": c} for c in commands]}


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "cd /app && python -m pytest tests",
        "python3 -m unittest discover",
        "npm test",
        "cargo test --release",
        "go test ./...",
        "make test",
        "python test_solution.py",
        "bash tests.sh",
    ],
)
def test_test_runners(command):
    assert verification([run(command)]).level == "test-runner"


OWN = own_check_pattern(["tests/test_outputs.py", "src/eval.py"])


@pytest.mark.parametrize(
    "command",
    [
        "python /app/test_outputs.py",
        "cd /app && python3.11 test_outputs.py 2>&1",
        "pytest -q tests/test_outputs.py",
        "python3 src/eval.py",
    ],
)
def test_running_a_shipped_check(command):
    assert verification([run(command)], OWN).level == "task-check"


def test_reading_a_shipped_check_is_not_running_it():
    assert verification([run("cat /app/test_outputs.py")], OWN).level == "none"


def test_a_file_of_that_name_on_another_task_is_the_agents_own():
    assert verification([run("python test_outputs.py")]).level == "test-runner"


@pytest.mark.parametrize(
    "command",
    ["python verify_output.py", "diff out.txt expected.txt", "python -c 'assert 1 == 1'"],
)
def test_adhoc_checks(command):
    assert verification([run(command)]).level == "adhoc-check"


@pytest.mark.parametrize(
    "command",
    ["ls -la", "pip install pytest-xdist==1", "cat test_results.txt", "git status"],
)
def test_other_commands_are_not_checks(command):
    assert verification([run(command)]).level == "none"


def test_strongest_level_wins():
    assert verification([run("pytest"), run("diff a b")]).level == "test-runner"


def test_a_check_must_follow_the_last_edit():
    edit = {"src": "agent", "tools": [{"fn": "Write", "cmd": "/app/x.py"}]}
    assert verification([edit, run("pytest", fn="Bash")]).after_last_edit
    assert not verification([run("pytest", fn="Bash"), edit]).after_last_edit


def test_a_viewing_editor_is_not_an_edit():
    view = {"src": "agent", "tools": [{"fn": "str_replace_editor", "cmd": "view"}]}
    assert verification([run("pytest", fn="execute_bash"), view]).after_last_edit


def test_lost_commands_are_counted_and_treated_as_edits():
    found = verification([run("pytest"), run("$4f")])
    assert (found.level, found.opaque_commands, found.after_last_edit) == (
        "test-runner",
        1,
        False,
    )


def test_blank_commands_are_counted():
    found = verification([run("")])
    assert (found.level, found.blank_commands) == ("none", 1)


def test_the_submit_sentinel_is_not_a_command():
    found = verification([run("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")])
    assert found.shell_commands == 0
