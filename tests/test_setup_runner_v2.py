"""More tests for setup runner internal methods."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from rich.console import Console
from scripts.setup._runner import (
    _menu_status_label,
    _print_menu,
    _print_summary,
    _collect_menu_statuses,
    _run_menu_step
)

@pytest.fixture
def console():
    return MagicMock(spec=Console)

def test_menu_status_label():
    assert "OK" in _menu_status_label("ok")
    assert "Pending" in _menu_status_label("pending")
    assert "unknown" == _menu_status_label("unknown")

def test_print_menu(console):
    step = MagicMock()
    step.name = "S1"
    step.description = "D1"
    _print_menu([step], ["ok"], console)
    console.print.assert_called_once()

def test_print_summary(console):
    _print_summary([("S1", "ok")], console)
    console.print.assert_called_once()

def test_collect_menu_statuses(console):
    step = MagicMock()
    step.name = "S1"
    step.check.return_value = True
    step.skip_when.return_value = False
    
    statuses = _collect_menu_statuses([step], console, {})
    assert statuses == ["ok"]

def test_run_menu_step_skip(console):
    step = MagicMock()
    step.name = "S1"
    step.skip_when.return_value = True
    
    assert _run_menu_step(step, console, {}) is True

def test_run_menu_step_already_ok(console):
    step = MagicMock()
    step.name = "S1"
    step.skip_when.return_value = False
    step.check.return_value = True
    
    assert _run_menu_step(step, console, {}) is True
