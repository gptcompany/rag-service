"""Tests for the setup wizard entry point."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from scripts.setup.main import main, _handle_get, _handle_set


@pytest.fixture
def console():
    return MagicMock(spec=Console)


def test_main_help(console):
    with patch("scripts.setup.main.Console", return_value=console):
        assert main(["--help"]) == 0
        assert main(["-h"]) == 0
        assert main(["help"]) == 0


def test_main_no_args_calls_interactive(console):
    with patch("scripts.setup.main.Console", return_value=console), \
         patch("scripts.setup.main.run_interactive_menu", return_value=True) as mock_menu, \
         patch("scripts.setup.main._all_steps", return_value=[]):
        assert main([]) == 0
        mock_menu.assert_called_once()


def test_main_subcommand_calls_run_steps(console):
    with patch("scripts.setup.main.Console", return_value=console), \
         patch("scripts.setup.main.run_steps", return_value=True) as mock_run, \
         patch("scripts.setup.main.SUBCOMMANDS", {"test": (MagicMock(return_value=[]), "desc")}):
        assert main(["test"]) == 0
        mock_run.assert_called_once()


def test_main_unknown_subcommand(console):
    with patch("scripts.setup.main.Console", return_value=console):
        assert main(["unknown"]) == 1


def test_handle_get_all(console):
    with patch("scripts.setup._config_presets.get_env", return_value="some_val"):
        assert _handle_get([], console) == 0
        console.print.assert_called_once() # Should print table


def test_handle_get_single_ok(console, capsys):
    with patch("scripts.setup._config_presets.get_env", return_value="8080"):
        assert _handle_get(["RAG_PORT"], console) == 0
        out, _ = capsys.readouterr()
        assert "8080" in out


def test_handle_get_single_fail(console):
    with patch("scripts.setup._config_presets.get_env", return_value=None):
        assert _handle_get(["UNKNOWN"], console) == 1


def test_handle_set_ok(console):
    with patch("scripts.setup._config_presets.set_env", return_value=True):
        assert _handle_set(["RAG_PORT", "9000"], console) == 0


def test_handle_set_invalid_port(console):
    assert _handle_set(["RAG_PORT", "invalid"], console) == 1


def test_handle_set_invalid_dim(console):
    assert _handle_set(["RAG_EMBEDDING_DIM", "-1"], console) == 1


def test_handle_set_invalid_deploy(console):
    assert _handle_set(["RAG_DEPLOY_MODE", "cloud"], console) == 1


def test_handle_set_invalid_ollama(console):
    assert _handle_set(["RAG_OLLAMA_MODE", "internal"], console) == 1


def test_handle_set_fail(console):
    with patch("scripts.setup._config_presets.set_env", return_value=False):
        assert _handle_set(["RAG_PORT", "8080"], console) == 1


def test_handle_set_missing_args(console):
    assert _handle_set(["RAG_PORT"], console) == 1
