"""Controllers execute model-authored steps, so the guards matter more than the happy path."""

import builtins

import pytest

from src.controller import (
    KEY_CODES, BrowserController, Controller, DesktopController, ShellController,
    allow_all, ask_terminal, available, status,
)
from src.plan import Plan, Step


class TestShellTools:
    def test_run_returns_output(self, shell):
        assert shell.execute(Step("run", {"command": "echo hi"})).output == "hi"

    def test_run_reports_nonzero_exit_as_failure(self, shell):
        result = shell.execute(Step("run", {"command": "exit 3"}))
        assert not result.ok and "exit 3" in result.output

    def test_run_with_no_output_still_succeeds(self, shell):
        assert shell.execute(Step("run", {"command": "true"})).ok

    def test_read_and_write_round_trip(self, shell, workdir):
        shell.execute(Step("write_file", {"filename": "new.txt", "content": "abc"}))
        assert (workdir / "new.txt").read_text() == "abc"
        assert shell.execute(Step("read_file", {"filename": "new.txt"})).output == "abc"

    def test_write_creates_missing_directories(self, shell, workdir):
        shell.execute(Step("write_file", {"filename": "a/b/c.txt", "content": "x"}))
        assert (workdir / "a/b/c.txt").read_text() == "x"

    def test_list_dir_marks_directories(self, shell):
        assert "sub/" in shell.execute(Step("list_dir", {"path": "."})).output


class TestGuards:
    @pytest.mark.parametrize("filename", [
        "../outside.txt", "../../etc/passwd", "sub/../../escape.txt", "/etc/passwd",
    ])
    def test_file_tools_cannot_escape_the_root(self, shell, filename):
        # The model picks these paths; traversal will be attempted eventually.
        result = shell.execute(Step("read_file", {"filename": filename}))
        assert not result.ok and "escapes the project root" in result.output

    def test_write_cannot_escape_the_root(self, shell, tmp_path):
        result = shell.execute(
            Step("write_file", {"filename": "../pwned.txt", "content": "x"})
        )
        assert not result.ok
        assert not (tmp_path / "pwned.txt").exists()

    def test_unknown_tool_is_reported_not_raised(self, shell):
        result = shell.execute(Step("fly", {}))
        assert not result.ok
        assert "no such tool" in result.output
        assert "run" in result.output          # names what is available

    def test_wrong_arguments_are_reported_not_raised(self, shell):
        result = shell.execute(Step("read_file", {"nonsense": 1}))
        assert not result.ok and "bad arguments" in result.output

    def test_missing_required_argument_is_reported(self, shell):
        assert not shell.execute(Step("read_file", {})).ok

    def test_exceptions_become_failed_results(self, shell):
        result = shell.execute(Step("read_file", {"filename": "nope.txt"}))
        assert not result.ok and "FileNotFoundError" in result.output


class TestConfirmation:
    def test_mutating_step_is_declined_when_refused(self, workdir):
        controller = ShellController(root=workdir, confirm=lambda step: False)
        result = controller.execute(Step("write_file", {"filename": "x", "content": "y"}))
        assert not result.ok and result.output == "declined by operator"
        assert not (workdir / "x").exists()

    def test_read_only_step_is_not_gated(self, workdir):
        # list_dir is not in MUTATING, so it must run without asking.
        asked = []
        controller = ShellController(
            root=workdir, confirm=lambda step: asked.append(step) or True
        )
        assert controller.execute(Step("list_dir", {})).ok
        assert asked == []

    def test_confirmation_sees_the_step_it_is_approving(self, workdir):
        seen = []
        controller = ShellController(root=workdir, confirm=lambda s: seen.append(str(s)) or False)
        controller.execute(Step("run", {"command": "rm -rf /"}))
        assert "rm -rf /" in seen[0]

    @pytest.mark.parametrize("answer, expected", [
        ("y", True), ("yes", True), ("Y", True),
        ("n", False), ("", False), ("nope", False),
    ])
    def test_ask_terminal_parses_answers(self, monkeypatch, answer, expected):
        monkeypatch.setattr(builtins, "input", lambda _: answer)
        assert ask_terminal(Step("run", {"command": "ls"})) is expected

    @pytest.mark.parametrize("interrupt", [EOFError, KeyboardInterrupt])
    def test_ask_terminal_treats_interrupt_as_no(self, monkeypatch, interrupt):
        # Regression: Ctrl-D at a confirmation prompt used to crash Rote.
        def raise_it(_):
            raise interrupt

        monkeypatch.setattr(builtins, "input", raise_it)
        assert ask_terminal(Step("run", {"command": "ls"})) is False


class TestRunPlan:
    def test_runs_every_step_in_order(self, shell, workdir):
        plan = Plan(controller="shell", steps=[
            Step("write_file", {"filename": "1.txt", "content": "one"}),
            Step("write_file", {"filename": "2.txt", "content": "two"}),
        ])
        assert all(r.ok for r in shell.run_plan(plan))
        assert (workdir / "2.txt").read_text() == "two"

    def test_stops_at_first_failure_by_default(self, shell, workdir):
        plan = Plan(controller="shell", steps=[
            Step("read_file", {"filename": "missing.txt"}),
            Step("write_file", {"filename": "after.txt", "content": "x"}),
        ])
        results = shell.run_plan(plan)
        assert len(results) == 1                      # second step never ran
        assert not (workdir / "after.txt").exists()

    def test_can_continue_past_failure(self, shell):
        plan = Plan(controller="shell", steps=[
            Step("read_file", {"filename": "missing.txt"}),
            Step("run", {"command": "echo done"}),
        ])
        results = shell.run_plan(plan, stop_on_failure=False)
        assert [r.ok for r in results] == [False, True]

    def test_empty_plan_runs_nothing(self, shell):
        assert shell.run_plan(Plan(controller="shell", steps=[])) == []


class TestAvailability:
    def test_shell_is_always_usable(self):
        assert ShellController().is_available()[0]

    def test_unusable_controller_explains_how_to_fix_it(self):
        ok, reason = BrowserController().is_available()
        if not ok:
            assert "pip install playwright" in reason

    def test_only_usable_filters_the_menu(self):
        # The bridge must never offer a controller whose every step would fail.
        usable = available(only_usable=True)
        for controller in usable.values():
            assert controller.is_available()[0]
        assert "shell" in usable

    def test_status_covers_every_controller(self):
        assert {name for name, _, _ in status()} == {"shell", "desktop", "browser"}

    def test_base_controller_requires_tools(self):
        with pytest.raises(NotImplementedError):
            Controller().tools()


class TestKeyCodes:
    @pytest.mark.parametrize("key, code", [
        ("ctrl", 29), ("s", 31), ("alt", 56), ("f4", 62), ("enter", 28),
        ("a", 30), ("z", 44), ("1", 2), ("0", 11), ("f12", 88),
    ])
    def test_known_evdev_codes(self, key, code):
        # Wrong codes mean silently pressing the wrong key, so pin the common ones.
        assert KEY_CODES[key] == code

    def test_letters_are_not_alphabetical(self):
        # Guards the bug where a placeholder mapped a..z to 0..25 in order.
        assert KEY_CODES["a"] != KEY_CODES["b"] - 1

    def test_every_letter_and_digit_is_mapped(self):
        for character in "abcdefghijklmnopqrstuvwxyz0123456789":
            assert character in KEY_CODES

    def test_unknown_key_is_rejected(self, monkeypatch):
        # Validation is ydotool's (it takes keycodes); pin the backend so the
        # test does not depend on which input tool this machine has installed.
        desktop = DesktopController()
        monkeypatch.setattr(desktop, "_backend", lambda: "ydotool")
        with pytest.raises(ValueError, match="unknown key"):
            desktop.key("ctrl+nosuchkey")

    def test_empty_key_string_is_rejected(self):
        with pytest.raises(ValueError, match="no keys given"):
            DesktopController().key("")
