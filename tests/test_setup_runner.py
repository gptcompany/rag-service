"""Tests for the setup runner and interactive menu."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from scripts.setup._runner import run_steps, run_interactive_menu


class MockStep:
    def __init__(self, name="Test Step", check_val=False, install_val=True, verify_val=True, skip=False):
        self.name = name
        self.description = f"Description for {name}"
        self.check_val = check_val
        self.install_val = install_val
        self.verify_val = verify_val
        self.skip = skip
        self.install_called = 0

    def check(self):
        return self.check_val

    def install(self, console: Console):
        self.install_called += 1
        return self.install_val

    def verify(self):
        return self.verify_val

    def skip_when(self):
        return self.skip


@pytest.fixture
def console():
    return MagicMock(spec=Console)


def test_run_steps_all_ok(console):
    steps = [MockStep(check_val=True), MockStep(check_val=False)]
    with patch("questionary.confirm") as mock_confirm:
        mock_confirm.return_value.ask.return_value = True
        assert run_steps(steps, console) is True
        assert steps[1].install_called == 1


def test_run_steps_skip_via_skip_when(console):
    steps = [MockStep(skip=True)]
    assert run_steps(steps, console) is True
    assert steps[0].install_called == 0


def test_run_steps_skip_via_confirm(console):
    steps = [MockStep(check_val=False)]
    with patch("questionary.confirm") as mock_confirm:
        mock_confirm.return_value.ask.return_value = False
        assert run_steps(steps, console) is True
        assert steps[0].install_called == 0


def test_run_steps_fail_and_abort(console):
    steps = [MockStep(install_val=False)]
    with patch("questionary.confirm") as mock_confirm, \
         patch("questionary.select") as mock_select:
        mock_confirm.return_value.ask.return_value = True
        mock_select.return_value.ask.return_value = "Abort"
        assert run_steps(steps, console) is False


def test_run_steps_fail_retry_success(console):
    step = MockStep(install_val=False)
    steps = [step]
    with patch("questionary.confirm") as mock_confirm, \
         patch("questionary.select") as mock_select:
        mock_confirm.return_value.ask.return_value = True
        mock_select.return_value.ask.return_value = "Retry"
        
        # First call fails, second call (retry) we make it succeed
        def side_effect(console):
            step.install_val = True
            return False
        
        with patch.object(step, "install", side_effect=side_effect):
            # This is a bit tricky due to how I structured MockStep, 
            # let's just use a real side_effect on a mock
            pass

    # Simplified retry test
    mock_step = MagicMock()
    mock_step.name = "RetryStep"
    mock_step.check.return_value = False
    mock_step.install.side_effect = [False, True]
    mock_step.verify.return_value = True
    mock_step.skip_when.return_value = False

    with patch("questionary.confirm") as mock_confirm, \
         patch("questionary.select") as mock_select:
        mock_confirm.return_value.ask.return_value = True
        mock_select.return_value.ask.return_value = "Retry"
        assert run_steps([mock_step], console) is True
        assert mock_step.install.call_count == 2


def test_run_interactive_menu_exit(console):
    steps = [MockStep(check_val=True)]
    with patch("questionary.select") as mock_select:
        mock_select.return_value.ask.return_value = ("exit", None)
        assert run_interactive_menu(steps, console) is True


def test_run_interactive_menu_run_step_then_exit(console):
    steps = [MockStep(check_val=False)]
    with patch("questionary.select") as mock_select:
        mock_select.return_value.ask.side_effect = [
            ("step", 0),
            ("exit", None)
        ]
        assert run_interactive_menu(steps, console) is True
        assert steps[0].install_called == 1


def test_run_interactive_menu_run_all(console):
    steps = [MockStep(name="S1", check_val=False), MockStep(name="S2", check_val=False)]
    with patch("questionary.select") as mock_select:
        mock_select.return_value.ask.side_effect = [
            ("run_all", None),
            ("exit", None)
        ]
        assert run_interactive_menu(steps, console) is True
        assert steps[0].install_called == 1
        assert steps[1].install_called == 1


def test_run_interactive_menu_interrupt(console):
    steps = [MockStep()]
    with patch("questionary.select") as mock_select:
        mock_select.return_value.ask.side_effect = KeyboardInterrupt
        assert run_interactive_menu(steps, console) is False
