"""Tests for setup config and presets."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest
from scripts.setup._config_presets import discover_ollama_models, get_env, set_env
from scripts.setup._config import ConfigStep


def test_discover_ollama_models_ok():
    with patch("urllib.request.urlopen") as mock_url:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"models": [{"name": "m1"}, {"name": "m2"}]}'
        mock_resp.__enter__.return_value = mock_resp
        mock_url.return_value = mock_resp
        
        models = discover_ollama_models("http://localhost:11434")
        assert models == ["m1", "m2"]


def test_discover_ollama_models_fail():
    with patch("urllib.request.urlopen", side_effect=OSError("connection failed")):
        assert discover_ollama_models() == []


def test_get_env_success():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="val\n")
        assert get_env("KEY") == "val"


def test_set_env_success():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assert set_env("KEY", "VAL") is True


def test_config_step_check():
    step = ConfigStep()
    with patch("scripts.setup._config.get_env") as mock_get:
        # check() calls get_env for 7 required keys + port + dim = 9 calls
        mock_get.side_effect = ["val"] * 7 + ["8767", "1024"]
        assert step.check() is True


def test_config_step_install_custom(tmp_path):
    step = ConfigStep()
    console = MagicMock()
    
    with patch("questionary.select") as mock_select, \
         patch("questionary.text") as mock_text, \
         patch("questionary.confirm") as mock_confirm, \
         patch("scripts.setup._config_presets.discover_ollama_models", return_value=[]), \
         patch("scripts.setup._config_presets.set_env", return_value=True), \
         patch("scripts.setup._config.get_env", return_value="host"):
        
        mock_select.return_value.ask.side_effect = [
            "Custom",       # OpenAI select
            "Custom",       # Ollama select
            "custom",       # Embed pick
            "mineru"        # Parser
        ]
        mock_text.return_value.ask.side_effect = [
            "custom-llm",   # OpenAI text
            "custom-ollama",# Ollama text
            "hf/embed",     # Embed model
            "512",          # Embed dim
            "8888"          # Port
        ]
        mock_confirm.return_value.ask.side_effect = [
            True, # Reranker
            True, # Vision
            True # Apply
        ]
        
        assert step.install(console) is True
