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

    @patch("scripts.setup._python.VENV_DIR")
    @patch("sys.version_info", (3, 11, 0))
    def test_check_passes(self, mock_venv_dir):
        step = self._make_step()
        mock_venv_dir.__truediv__.return_value.__truediv__.return_value.exists.return_value = True
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert step.check() is True

    @patch("sys.version_info", (3, 9, 0))
    def test_check_fails_old_python(self):
        step = self._make_step()
        assert step.check() is False

    @patch("scripts.setup._python.VENV_DIR")
    @patch("sys.version_info", (3, 11, 0))
    def test_check_fails_no_venv(self, mock_venv_dir):
        step = self._make_step()
        mock_venv_dir.__truediv__.return_value.__truediv__.return_value.exists.return_value = False
        assert step.check() is False

    @patch("scripts.setup._python.VENV_DIR")
    @patch("sys.version_info", (3, 11, 0))
    def test_check_fails_import_error(self, mock_venv_dir):
        step = self._make_step()
        mock_venv_dir.__truediv__.return_value.__truediv__.return_value.exists.return_value = True
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert step.check() is False

    @patch("scripts.setup._python.VENV_DIR")
    def test_install_success(self, mock_venv_dir):
        step = self._make_step()
        console = MagicMock()
        mock_venv_dir.__truediv__.return_value.__truediv__.return_value.exists.return_value = False
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert step.install(console) is True
            # Should call venv creation, pip install -e, and pip install deps
            assert mock_run.call_count == 3

    @patch("scripts.setup._python.VENV_DIR")
    def test_install_venv_fails(self, mock_venv_dir):
        step = self._make_step()
        console = MagicMock()
        mock_venv_dir.__truediv__.return_value.__truediv__.return_value.exists.return_value = False
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="venv fail")
            assert step.install(console) is False
            assert mock_run.call_count == 1

    @patch("scripts.setup._python.VENV_DIR")
    def test_install_pip_editable_fails(self, mock_venv_dir):
        step = self._make_step()
        console = MagicMock()
        mock_venv_dir.__truediv__.return_value.__truediv__.return_value.exists.return_value = True
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [MagicMock(returncode=1, stderr="pip fail")]
            assert step.install(console) is False

    @patch("scripts.setup._python.VENV_DIR")
    def test_install_deps_fails(self, mock_venv_dir):
        step = self._make_step()
        console = MagicMock()
        mock_venv_dir.__truediv__.return_value.__truediv__.return_value.exists.return_value = True
        with patch("subprocess.run") as mock_run:
            # 1. pip install -e success
            # 2. pip install deps fails
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=1, stderr="deps fail")
            ]
            assert step.install(console) is False

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
        mock_run.return_value = MagicMock(returncode=0, stdout="sk-test\n")
        step = self._make_step()
        assert step._key_exists() is True

    @patch("subprocess.run")
    def test_check_fails_when_key_missing(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        step = self._make_step()
        assert step._key_exists() is False

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_check_fails_on_error(self, _):
        step = self._make_step()
        assert step._key_exists() is False

    @patch("subprocess.run")
    def test_check_does_not_use_bash_shell_wrapper(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="sk-test\n")
        step = self._make_step()
        assert step._key_exists() is True
        args = mock_run.call_args.args[0]
        assert args[0] == "dotenvx"
        assert "bash" not in args

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

    @patch("scripts.setup._service._get_deploy_mode", return_value="host")
    @patch("scripts.setup._service.ServiceStep._systemd_enabled", return_value=True)
    @patch("urllib.request.urlopen")
    def test_check_passes_when_host_and_systemd_enabled(self, mock_urlopen, *_):
        step = self._make_step()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp
        assert step.check() is True

    @patch("scripts.setup._service._get_deploy_mode", return_value="host")
    @patch("scripts.setup._service.ServiceStep._systemd_enabled", return_value=False)
    @patch("urllib.request.urlopen")
    def test_check_fails_when_host_and_systemd_disabled(self, mock_urlopen, *_):
        step = self._make_step()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp
        assert step.check() is False

    @patch("scripts.setup._service.shutil.which", return_value="/usr/bin/systemctl")
    @patch("scripts.setup._service.subprocess.run")
    def test_systemd_enabled_detects_enabled_unit(self, mock_run, _mock_which):
        from scripts.setup._service import ServiceStep

        mock_run.return_value = MagicMock(returncode=0, stdout="enabled\n")
        assert ServiceStep._systemd_enabled() is True

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
        step.skip_when = None  # Not a skippable step

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

    @patch("questionary.select")
    def test_interactive_menu_exit_immediately(self, mock_select):
        from scripts.setup._runner import run_interactive_menu

        step = MagicMock()
        step.name = "Already OK"
        step.description = "Test step"
        step.check.return_value = True

        mock_select.return_value.ask.return_value = ("exit", None)

        console = MagicMock()
        result = run_interactive_menu([step], console)

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

    @patch("scripts.setup.main.run_interactive_menu", return_value=True)
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

    @patch("scripts.setup.main.run_interactive_menu", return_value=False)
    def test_failure_returns_1(self, mock_run):
        from scripts.setup.main import main
        result = main([])
        assert result == 1

    @patch("scripts.setup.main.run_steps", return_value=False)
    def test_subcommand_failure_returns_1(self, mock_run):
        from scripts.setup.main import main
        result = main(["verify"])
        assert result == 1

    @patch("scripts.setup.main.run_steps", return_value=True)
    def test_config_subcommand(self, mock_run):
        from scripts.setup.main import main
        result = main(["config"])
        assert result == 0

    @patch("scripts.setup.main.run_steps", return_value=True)
    def test_deploy_subcommand(self, mock_run):
        from scripts.setup.main import main
        result = main(["deploy"])
        assert result == 0


# ── DeployStep ──────────────────────────────────────────────

class TestDeployStep:
    def _make_step(self):
        from scripts.setup._deploy import DeployStep
        return DeployStep()

    @patch("scripts.setup._config_presets.subprocess.run")
    def test_check_passes_when_mode_set(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="host\n")
        step = self._make_step()
        assert step.check() is True

    @patch("scripts.setup._config_presets.subprocess.run")
    def test_check_fails_when_mode_missing(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        step = self._make_step()
        assert step.check() is False

    @patch("scripts.setup._deploy.set_env", return_value=True)
    @patch("questionary.select")
    def test_install_host_mode(self, mock_select, mock_set):
        step = self._make_step()
        console = MagicMock()
        mock_select.return_value.ask.return_value = "host"
        result = step.install(console)
        assert result is True
        # Should persist deploy_mode=host and ollama_mode=external
        assert mock_set.call_count == 2

    @patch("scripts.setup._deploy.set_env", return_value=True)
    @patch("questionary.select")
    def test_install_docker_sidecar(self, mock_select, mock_set):
        step = self._make_step()
        console = MagicMock()
        mock_select.return_value.ask.side_effect = ["docker", "sidecar"]
        result = step.install(console)
        assert result is True

    @patch("questionary.select")
    def test_install_cancelled(self, mock_select):
        step = self._make_step()
        console = MagicMock()
        mock_select.return_value.ask.return_value = None
        result = step.install(console)
        assert result is False

    @patch("scripts.setup._deploy.set_env", side_effect=[False])
    @patch("questionary.select")
    def test_install_fails_when_deploy_mode_not_persisted(self, mock_select, mock_set):
        step = self._make_step()
        console = MagicMock()
        mock_select.return_value.ask.return_value = "host"
        result = step.install(console)
        assert result is False
        mock_set.assert_called_once()
        assert any("Failed to persist deploy mode" in str(c) for c in console.print.call_args_list)

    @patch("scripts.setup._deploy.set_env", side_effect=[True, False])
    @patch("questionary.select")
    def test_install_fails_when_ollama_mode_not_persisted(self, mock_select, mock_set):
        step = self._make_step()
        console = MagicMock()
        mock_select.return_value.ask.return_value = "host"
        result = step.install(console)
        assert result is False
        assert mock_set.call_count == 2
        assert any("Failed to persist Ollama mode" in str(c) for c in console.print.call_args_list)

    def test_verify_delegates_to_check(self):
        step = self._make_step()
        with patch.object(step, "check", return_value=True):
            assert step.verify() is True


class TestStepMetadata:
    def test_all_steps_have_descriptions(self):
        from scripts.setup.main import _all_steps

        for step in _all_steps():
            assert isinstance(step.description, str)
            assert step.description.strip()


# ── ConfigStep ──────────────────────────────────────────────

class TestConfigStep:
    def _make_step(self):
        from scripts.setup._config import ConfigStep
        return ConfigStep()

    @patch("scripts.setup._config.get_env")
    def test_check_passes_when_configured(self, mock_get):
        from scripts.setup._config_presets import ENV_VARS
        values = {
            ENV_VARS["openai_model"]: "gpt-4o-mini",
            ENV_VARS["ollama_model"]: "qwen3:8b",
            ENV_VARS["embedding_model"]: "BAAI/bge-large-en-v1.5",
            ENV_VARS["embedding_dim"]: "1024",
            ENV_VARS["default_parser"]: "mineru",
            ENV_VARS["port"]: "8767",
            ENV_VARS["host"]: "0.0.0.0",
        }
        mock_get.side_effect = lambda key: values.get(key, "default")
        step = self._make_step()
        assert step.check() is True

    @patch("scripts.setup._config.get_env", return_value=None)
    def test_check_fails_when_not_configured(self, _):
        step = self._make_step()
        assert step.check() is False

    @patch("scripts.setup._config.get_env")
    def test_check_fails_when_config_is_partial(self, mock_get):
        from scripts.setup._config_presets import ENV_VARS
        values = {
            ENV_VARS["openai_model"]: "gpt-4o-mini",
            ENV_VARS["ollama_model"]: "qwen3:8b",
            ENV_VARS["embedding_model"]: "BAAI/bge-large-en-v1.5",
            # Missing embedding_dim
            ENV_VARS["default_parser"]: "mineru",
            ENV_VARS["port"]: "8767",
            ENV_VARS["host"]: "0.0.0.0",
        }
        mock_get.side_effect = lambda key: values.get(key)
        step = self._make_step()
        assert step.check() is False

    @patch("scripts.setup._config.get_env")
    def test_check_fails_when_port_is_invalid(self, mock_get):
        from scripts.setup._config_presets import ENV_VARS
        values = {
            ENV_VARS["openai_model"]: "gpt-4o-mini",
            ENV_VARS["ollama_model"]: "qwen3:8b",
            ENV_VARS["embedding_model"]: "BAAI/bge-large-en-v1.5",
            ENV_VARS["embedding_dim"]: "1024",
            ENV_VARS["default_parser"]: "mineru",
            ENV_VARS["port"]: "banana",
            ENV_VARS["host"]: "0.0.0.0",
        }
        mock_get.side_effect = lambda key: values.get(key)
        step = self._make_step()
        assert step.check() is False

    @patch("scripts.setup._config.get_env")
    def test_check_fails_when_embedding_dim_is_zero(self, mock_get):
        from scripts.setup._config_presets import ENV_VARS
        values = {
            ENV_VARS["openai_model"]: "gpt-4o-mini",
            ENV_VARS["ollama_model"]: "qwen3:8b",
            ENV_VARS["embedding_model"]: "BAAI/bge-large-en-v1.5",
            ENV_VARS["embedding_dim"]: "0",
            ENV_VARS["default_parser"]: "mineru",
            ENV_VARS["port"]: "8767",
            ENV_VARS["host"]: "0.0.0.0",
        }
        mock_get.side_effect = lambda key: values.get(key)
        step = self._make_step()
        assert step.check() is False

    def test_verify_delegates_to_check(self):
        step = self._make_step()
        with patch.object(step, "check", return_value=True):
            assert step.verify() is True

    @patch("scripts.setup._config.get_env", return_value=None)
    @patch("scripts.setup._config.set_env", return_value=True)
    @patch("scripts.setup._config.discover_ollama_models", return_value=[])
    @patch("questionary.select")
    @patch("questionary.text")
    @patch("questionary.confirm")
    def test_install_full_flow(
        self, mock_confirm, mock_text, mock_select, mock_discover, mock_set, mock_get
    ):
        from scripts.setup._config_presets import EMBEDDING_PRESETS

        step = self._make_step()
        console = MagicMock()

        # Simulate user choices: openai, ollama custom, embed preset, rerank, parser, port, vision, confirm
        mock_select.return_value.ask.side_effect = [
            "gpt-4o-mini",          # 1/6 OpenAI
            EMBEDDING_PRESETS[0],   # 3/6 Embedding (preset)
            "mineru",               # 5/6 Parser
        ]
        mock_text.return_value.ask.side_effect = [
            "qwen3:8b",  # 2/6 Ollama (custom fallback)
            "8767",       # 6/6 Port
        ]
        mock_confirm.return_value.ask.side_effect = [
            True,   # 4/6 Reranker enable
            False,  # 6/6 Vision disable
            True,   # Apply config confirm
        ]

        result = step.install(console)
        assert result is True
        assert mock_set.call_count >= 8  # At least 8 env vars set

    def test_external_compose_template_supports_linux_host_gateway(self):
        from scripts.setup._config import COMPOSE_EXTERNAL_TEMPLATE
        assert "host.docker.internal:host-gateway" in COMPOSE_EXTERNAL_TEMPLATE

    @patch("scripts.setup._config.get_env", return_value=None)
    @patch("scripts.setup._config.set_env", return_value=True)
    @patch("scripts.setup._config.discover_ollama_models", return_value=[])
    @patch("questionary.select")
    @patch("questionary.text")
    @patch("questionary.confirm")
    def test_install_rejects_invalid_custom_embedding_dim(
        self, mock_confirm, mock_text, mock_select, mock_discover, mock_set, mock_get
    ):
        step = self._make_step()
        console = MagicMock()

        mock_select.return_value.ask.side_effect = [
            "gpt-4o-mini",  # OpenAI
            "custom",       # Embedding choice
        ]
        mock_text.return_value.ask.side_effect = [
            "qwen3:8b",                 # Ollama model
            "BAAI/bge-large-en-v1.5",   # Embedding model
            "abc",                      # Invalid embedding dim
        ]

        result = step.install(console)
        assert result is False
        mock_set.assert_not_called()
        assert any("Invalid embedding dimension" in str(c) for c in console.print.call_args_list)

    @patch("scripts.setup._config.get_env", return_value=None)
    @patch("scripts.setup._config.set_env", return_value=True)
    @patch("scripts.setup._config.discover_ollama_models", return_value=[])
    @patch("questionary.select")
    @patch("questionary.text")
    @patch("questionary.confirm")
    def test_install_rejects_invalid_port(
        self, mock_confirm, mock_text, mock_select, mock_discover, mock_set, mock_get
    ):
        from scripts.setup._config_presets import EMBEDDING_PRESETS

        step = self._make_step()
        console = MagicMock()

        mock_select.return_value.ask.side_effect = [
            "gpt-4o-mini",        # OpenAI
            EMBEDDING_PRESETS[0], # Embedding preset
            "mineru",             # Parser
        ]
        mock_text.return_value.ask.side_effect = [
            "qwen3:8b",  # Ollama model
            "70000",     # Invalid port
        ]
        mock_confirm.return_value.ask.side_effect = [
            True,  # Reranker enable
        ]

        result = step.install(console)
        assert result is False
        mock_set.assert_not_called()
        assert any("Invalid port" in str(c) for c in console.print.call_args_list)

    @patch("questionary.confirm")
    @patch("scripts.setup._config.get_env", return_value="external")
    def test_generate_docker_files_does_not_overwrite_compose_when_declined(self, mock_get, mock_confirm, tmp_path):
        from scripts.setup import _config as config_mod

        step = self._make_step()
        console = MagicMock()
        mock_confirm.return_value.ask.return_value = False

        compose = tmp_path / "docker-compose.yml"
        original = "services:\n  custom: {}\n"
        compose.write_text(original)
        (tmp_path / "Dockerfile").write_text("FROM scratch\n")

        with patch.object(config_mod, "_SERVICE_ROOT", tmp_path):
            step._generate_docker_files(console)

        assert compose.read_text() == original
        assert any("Skipped docker-compose.yml generation" in str(c) for c in console.print.call_args_list)


# ── ConfigPresets ───────────────────────────────────────────

class TestConfigPresets:
    def test_env_vars_all_prefixed(self):
        from scripts.setup._config_presets import ENV_VARS
        for key, var_name in ENV_VARS.items():
            assert var_name.startswith("RAG_"), f"{key} -> {var_name} missing RAG_ prefix"

    def test_embedding_presets_valid(self):
        from scripts.setup._config_presets import EMBEDDING_PRESETS
        assert len(EMBEDDING_PRESETS) >= 4
        for preset in EMBEDDING_PRESETS:
            assert preset.dim > 0
            assert preset.model.startswith("BAAI/")
            assert len(preset.label) > 0

    def test_openai_presets_has_custom(self):
        from scripts.setup._config_presets import OPENAI_PRESETS
        assert "Custom" in OPENAI_PRESETS
        assert "gpt-4o-mini" in OPENAI_PRESETS

    def test_parsers_valid(self):
        from scripts.setup._config_presets import PARSERS
        names = [name for name, _ in PARSERS]
        assert "mineru" in names
        assert "docling" in names
        assert "paddleocr" in names

    @patch("urllib.request.urlopen")
    def test_discover_ollama_success(self, mock_urlopen):
        import io
        from scripts.setup._config_presets import discover_ollama_models
        response_data = b'{"models":[{"name":"qwen3:8b"},{"name":"llama3:latest"}]}'
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        models = discover_ollama_models("http://localhost:11434")
        assert models == ["qwen3:8b", "llama3:latest"]

    @patch("urllib.request.urlopen", side_effect=OSError("connection refused"))
    def test_discover_ollama_failure(self, _):
        from scripts.setup._config_presets import discover_ollama_models
        models = discover_ollama_models("http://localhost:11434")
        assert models == []


# ── Runner skip_when ────────────────────────────────────────

class TestRunnerSkipWhen:
    @patch("questionary.confirm")
    def test_step_with_skip_when_true_is_skipped(self, mock_confirm):
        from scripts.setup._runner import run_steps

        step = MagicMock()
        step.name = "Skippable"
        step.skip_when = MagicMock(return_value=True)

        console = MagicMock()
        console.status.return_value.__enter__ = MagicMock()
        console.status.return_value.__exit__ = MagicMock()

        result = run_steps([step], console)
        assert result is True
        step.check.assert_not_called()
        step.install.assert_not_called()

    @patch("questionary.confirm")
    def test_step_without_skip_when_runs_normally(self, mock_confirm):
        from scripts.setup._runner import run_steps

        step = MagicMock(spec=["name", "check", "install", "verify"])
        step.name = "Normal"
        step.check.return_value = True

        console = MagicMock()
        console.status.return_value.__enter__ = MagicMock()
        console.status.return_value.__exit__ = MagicMock()

        result = run_steps([step], console)
        assert result is True
        step.check.assert_called_once()

    @patch("questionary.confirm")
    def test_step_with_skip_when_false_runs(self, mock_confirm):
        from scripts.setup._runner import run_steps

        step = MagicMock()
        step.name = "NotSkipped"
        step.skip_when = MagicMock(return_value=False)
        step.check.return_value = True

        console = MagicMock()
        console.status.return_value.__enter__ = MagicMock()
        console.status.return_value.__exit__ = MagicMock()

        result = run_steps([step], console)
        assert result is True
        step.check.assert_called_once()
