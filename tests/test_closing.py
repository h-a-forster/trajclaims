"""How a trajectory ends, per scaffold. Shapes follow the dataset; contents are invented."""

from trajclaims.closing import closing


def agent(msg="", tools=None, obs=None):
    return {"src": "agent", "msg": msg, "tools": tools, "obs": obs}


BASH = [{"fn": "bash_command", "cmd": "ls"}]
MARK = [{"fn": "mark_task_complete", "cmd": ""}]
SUBMIT = [{"fn": "bash_command", "cmd": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}]


def test_a_placeholder_is_lost_not_a_message():
    end = closing("claude-code", [agent("Working.", [{"fn": "Bash", "cmd": "ls"}]), agent("$6d")])
    assert (end.ending, end.status, end.text) == ("stopped", "lost", "")


def test_a_voluntary_stop_keeps_its_text():
    end = closing("codex", [agent("Done. Wrote the file.")])
    assert (end.ending, end.status, end.text) == ("stopped", "readable", "Done. Wrote the file.")


def test_a_mid_work_step_is_a_cutoff_whatever_it_says():
    end = closing(
        "claude-code", [agent("All done, one last check.", [{"fn": "Bash", "cmd": "ls"}])]
    )
    assert (end.ending, end.status) == ("cutoff", "cutoff")


def test_a_signal_scaffold_cannot_stop_without_its_signal():
    assert closing("terminus-2", [agent("Thinking.")]).status == "cutoff"


def test_running_into_the_timeout_is_a_cutoff_even_with_no_tool_call():
    steps = [agent("Partial thought")]
    assert closing("openhands", steps).ending == "stopped"
    assert closing("openhands", steps, timed_out=True).ending == "cutoff"


def test_a_declared_end_survives_the_timeout():
    end = closing("terminus-2", [agent("Task complete.", MARK)], timed_out=True)
    assert (end.ending, end.signal) == ("declared", "mark_task_complete")


def test_terminus_uses_the_last_confirming_step():
    end = closing(
        "terminus-2", [agent("Run.", BASH), agent("First.", MARK), agent("Second.", MARK)]
    )
    assert (end.ending, end.text) == ("declared", "Second.")


def test_a_silent_signal_has_no_text():
    end = closing("terminus-3-3", [agent("", BASH), agent("", MARK), agent("", MARK)])
    assert (end.ending, end.status) == ("declared", "no-text")


def test_mini_swe_submit_is_a_declared_end():
    end = closing("mini-swe-agent", [agent("THOUGHT: finished.", SUBMIT)])
    assert (end.ending, end.status, end.signal) == ("declared", "readable", "submit")


def test_mini_swe_with_a_lost_command_could_have_submitted():
    steps = [agent("THOUGHT: x", [{"fn": "bash_command", "cmd": "$4f"}])]
    assert closing("mini-swe-agent", steps).status == "lost"
    assert closing("mini-swe-agent", steps, timed_out=True).status == "cutoff"


def test_openhands_reads_the_finish_argument_over_its_boilerplate():
    finish = [{"fn": "finish", "cmd": '{"message": "Built the binary."}'}]
    steps = [agent("All done! What's next on the agenda?", finish)]
    assert closing("openhands", steps).text == "Built the binary."


def test_openhands_finish_with_a_blank_argument_is_lost():
    # Every such case in the bridge sample had text in the full scrape.
    finish = [{"fn": "finish", "cmd": ""}]
    end = closing("openhands", [agent("All done! What's next on the agenda?", finish)])
    assert (end.ending, end.status) == ("declared", "lost")


def test_argv_commands_are_read():
    argv = [{"fn": "finish", "cmd": ["bash", "-lc", "x"]}]
    assert closing("openhands", [agent("", argv)]).ending == "declared"


def test_no_agent_steps():
    assert closing("codex", [{"src": "user", "msg": "hi"}]).ending == "none"


def test_non_string_arguments_do_not_stop_extraction():
    for value in (7, {"path": "/app"}):
        steps = [agent("Reading.", [{"fn": "read_file", "cmd": value}]), agent("Done.")]
        assert closing("gemini-cli", steps).text == "Done."
