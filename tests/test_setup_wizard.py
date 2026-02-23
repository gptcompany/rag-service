"""Unit tests for RAG Service setup wizard steps."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ── PythonStep ───────────────────────────────────────────────

class TestPythonStep:
    def _make_step(self):
        from scripts.setup._python import PythonStep
        return PythonStep()

    @patch("scripts.setup._python.VENV_DIR", new_callable=lambda: PropertyMock)
    def test_check_fails_when_no_venv(self, tmp_path):
        step = self._make_step()
        with patch.object(type(step), "check", return_value=False):
            assert step.check() is False

    def test_install_returns_bool(self):
        step = self._make_step()
        console = MagicMock()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="fail")
            result = step.install(console)
            assert isinstance(result, bool)

    def test_verify_delegates_to_check(self):
        step = self._make_step()
        with patch.object(step, "check", return_value=True):
            assert step.verify() is True


# ── MineruStep ───────────────────────────────────────────────

class TestMineruStep:
    def _make_step(self):
        from scripts.setup._mineru import MineruStep
        return MineruStep()

    def test_check_false_when_no_cache(self, tmp_path):
        step = self._make_step()
        with patch("scripts.setup._mineru.HF_CACHE", tmp_path / "nonexistent"):
            assert step.check() is False

    def test_check_true_when_models_cached(self, tmp_path):
        step = self._make_step()
        cache = tmp_path / "hub"
        cache.mkdir()
        (cache / "models--opendatalab--PDF-Extract-Kit-v1").mkdir()
        with patch("scripts.setup._mineru.HF_CACHE", cache):
            assert step.check() is True

    def test_install_fails_when_no_venv(self):
        step = self._make_step()
        console = MagicMock()
        with patch("scripts.setup._mineru.VENV_PYTHON", Path("/nonexistent/python")):
            result = step.install(console)
            assert result is False

    def test_install_fails_on_subprocess_error(self):
        step = self._make_step()
        console = MagicMock()
        with patch("scripts.setup._mineru.VENV_PYTHON", Path("/tmp/fake_python")), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="model error")
            result = step.install(console)
            assert result is False


# ── OllamaStep ───────────────────────────────────────────────

class TestOllamaStep:
    def _make_step(self):
        from scripts.setup._ollama import OllamaStep
        return OllamaStep()

    @patch("shutil.which", return_value=None)
    def test_ollama_not_installed(self, _):
        step = self._make_step()
        assert step._ollama_installed() is False
        assert step.check() is False

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_ollama_installed(self, _):
        step = self._make_step()
        assert step._ollama_installed() is True

    @patch("urllib.request.urlopen", side_effect=OSError)
    def test_ollama_not_serving(self, _):
        step = self._make_step()
        assert step._ollama_serving() is False

    @patch("subprocess.run")
    def test_model_exists(self, mock_run):
        mock_run.return_value = MagicMock(stdout="qwen3:8b\t4.7GB", returncode=0)
        step = self._make_step()
        assert step._model_exists() is True

    @patch("subprocess.run")
    def test_model_missing(self, mock_run):
        mock_run.return_value = MagicMock(stdout="llama3:latest", returncode=0)
        step = self._make_step()
        assert step._model_exists() is False

    @patch("shutil.which", return_value=None)
    def test_install_not_installed(self, _):
        step = self._make_step()
        console = MagicMock()
        assert step.install(console) is False
        # Should suggest install command
        assert any("install" in str(c) for c in console.print.call_args_list)

    @patch("shutil.which", return_value="/usr/bin/ollama")
    @patch("urllib.request.urlopen", side_effect=OSError)
    def test_install_not_serving(self, *_):
        step = self._make_step()
        console = MagicMock()
        assert step.install(console) is False

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_install_pulls_model(self, _):
        step = self._make_step()
        console = MagicMock()
        with patch.object(step, "_ollama_serving", return_value=True), \
             patch.object(step, "_model_exists", return_value=False), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = step.install(console)
            assert result is True
            mock_run.assert_called_once()


# ── LibreOfficeStep ──────────────────────────────────────────

class TestLibreOfficeStep:
    def _make_step(self):
        from scripts.setup._libreoffice import LibreOfficeStep
        return LibreOfficeStep()

    @patch("shutil.which", return_value="/usr/bin/libreoffice")
    def test_check_passes(self, _):
        step = self._make_step()
        assert step.check() is True

    @patch("shutil.which", return_value=None)
    def test_check_fails(self, _):
        step = self._make_step()
        assert step.check() is False

    @patch("shutil.which", return_value=None)
    def test_install_shows_instructions(self, _):
        step = self._make_step()
        console = MagicMock()
        assert step.install(console) is False
        console.print.assert_called()


# ── SecretsStep ──────────────────────────────────────────────

class TestSecretsStep:
    def _make_step(self):
        from scripts.setup._secrets import SecretsStep
        return SecretsStep()

    @patch("subprocess.run")
    def test_check_passes_when_key_exists(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        step = self._make_step()
        assert step._key_exists() is True

    @patch("subprocess.run")
    def test_check_fails_when_key_missing(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        step = self._make_step()
        assert step._key_exists() is False

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_check_fails_on_error(self, _):
        step = self._make_step()
        assert step._key_exists() is False

    def test_install_returns_false(self):
        """Cannot auto-install secrets."""
        step = self._make_step()
        console = MagicMock()
        assert step.install(console) is False


# ── ServiceStep ──────────────────────────────────────────────

class TestServiceStep:
    def _make_step(self):
        from scripts.setup._service import ServiceStep
        return ServiceStep()

    @patch("urllib.request.urlopen", side_effect=OSError)
    def test_check_fails_when_service_down(self, _):
        step = self._make_step()
        assert step.check() is False

    def test_install_shows_instructions(self):
        step = self._make_step()
        console = MagicMock()
        assert step.install(console) is False
        console.print.assert_called()


# ── VerifyStep ───────────────────────────────────────────────

class TestVerifyStep:
    def _make_step(self):
        from scripts.setup._verify import VerifyStep
        return VerifyStep()

    def test_check_always_false(self):
        step = self._make_step()
        assert step.check() is False

    def test_verify_always_true(self):
        step = self._make_step()
        assert step.verify() is True

    def test_install_informational(self):
        step = self._make_step()
        console = MagicMock()
        assert step.install(console) is True


# ── Runner ───────────────────────────────────────────────────

class TestRunner:
    @patch("questionary.confirm")
    def test_all_checks_pass(self, mock_confirm):
        from scripts.setup._runner import run_steps

        step = MagicMock()
        step.name = "Test"
        step.check.return_value = True

        console = MagicMock()
        console.status.return_value.__enter__ = MagicMock()
        console.status.return_value.__exit__ = MagicMock()

        result = run_steps([step], console)
        assert result is True
        step.install.assert_not_called()

    @patch("questionary.confirm")
    @patch("questionary.select")
    def test_abort_on_failure(self, mock_select, mock_confirm):
        from scripts.setup._runner import run_steps

        step = MagicMock()
        step.name = "Failing"
        step.check.return_value = False
        step.install.return_value = False

        mock_confirm.return_value.ask.return_value = True
        mock_select.return_value.ask.return_value = "Abort"

        console = MagicMock()
        console.status.return_value.__enter__ = MagicMock()
        console.status.return_value.__exit__ = MagicMock()

        result = run_steps([step], console)
        assert result is False

    @patch("questionary.confirm")
    def test_user_skips_step(self, mock_confirm):
        from scripts.setup._runner import run_steps

        step = MagicMock()
        step.name = "Optional"
        step.check.return_value = False
        mock_confirm.return_value.ask.return_value = False

        console = MagicMock()
        console.status.return_value.__enter__ = MagicMock()
        console.status.return_value.__exit__ = MagicMock()

        result = run_steps([step], console)
        assert result is True
        step.install.assert_not_called()


# ── Main CLI ─────────────────────────────────────────────────

class TestMain:
    def test_help(self):
        from scripts.setup.main import main
        result = main(["--help"])
        assert result == 0

    def test_unknown_subcommand(self):
        from scripts.setup.main import main
        result = main(["nonexistent"])
        assert result == 1

    @patch("scripts.setup.main.run_steps", return_value=True)
    def test_all_steps(self, mock_run):
        from scripts.setup.main import main
        result = main([])
        assert result == 0
        mock_run.assert_called_once()

    @patch("scripts.setup.main.run_steps", return_value=True)
    def test_verify_subcommand(self, mock_run):
        from scripts.setup.main import main
        result = main(["verify"])
        assert result == 0

    @patch("scripts.setup.main.run_steps", return_value=False)
    def test_failure_returns_1(self, mock_run):
        from scripts.setup.main import main
        result = main([])
        assert result == 1
