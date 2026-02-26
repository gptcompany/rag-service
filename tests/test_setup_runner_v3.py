"""Deep coverage tests for the wizard runner."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from scripts.setup._runner import run_steps, run_interactive_menu

class MockStep:
    def __init__(self, name="S1", check_val=False, install_val=True, verify_val=True, skip=False):
        self.name = name
        self.description = "Desc"
        self.check_val = check_val
        self.install_val = install_val
        self.verify_val = verify_val
        self.skip = skip
        self.install_count = 0

    def check(self): return self.check_val
    def install(self, console: Console): 
        self.install_count += 1
        return self.install_val
    def verify(self): return self.verify_val
    def skip_when(self): return self.skip

@pytest.fixture
def console():
    return MagicMock(spec=Console)

def test_run_steps_failure_retry_failure(console):
    # Test path where install fails, user retries, and it fails again
    step = MockStep(install_val=False)
    with patch("questionary.confirm") as mock_conf, \
         patch("questionary.select") as mock_sel:
        mock_conf.return_value.ask.return_value = True
        # Initial fail -> user selects Retry -> second fail (hardcoded in run_steps to return failed)
        mock_sel.return_value.ask.side_effect = ["Retry"]
        
        # run_steps will return False because the retry failed
        assert run_steps([step], console) is False
        assert step.install_count == 2

def test_run_steps_abort(console):
    step = MockStep(install_val=False)
    with patch("questionary.confirm") as mock_conf, \
         patch("questionary.select") as mock_sel:
        mock_conf.return_value.ask.return_value = True
        mock_sel.return_value.ask.return_value = "Abort"
        assert run_steps([step], console) is False

def test_interactive_menu_keyboard_interrupt_mid_run(console):
    # Test interrupt while running a step from the menu
    step = MockStep(check_val=False)
    with patch("questionary.select") as mock_sel, \
         patch.object(step, "install", side_effect=KeyboardInterrupt):
        mock_sel.return_value.ask.side_effect = [("step", 0), ("exit", None)]
        assert run_interactive_menu([step], console) is False # Interrupt during install -> False

def test_interactive_menu_failure_skip_branch(console):
    # Test failure during step run from menu, then user chooses Skip
    step = MockStep(install_val=False)
    with patch("questionary.select") as mock_sel:
        # 1. Main menu select step 0
        # 2. Step failed menu select Skip
        # 3. Main menu select exit
        mock_sel.return_value.ask.side_effect = [("step", 0), "Skip and continue", ("exit", None)]
        assert run_interactive_menu([step], console) is True
