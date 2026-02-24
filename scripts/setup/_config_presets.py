"""Configuration presets and auto-discovery for the setup wizard."""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

# Environment variable names for RAG service configuration
ENV_VARS = {
    "host": "RAG_HOST",
    "port": "RAG_PORT",
    "embedding_model": "RAG_EMBEDDING_MODEL",
    "embedding_dim": "RAG_EMBEDDING_DIM",
    "openai_model": "RAG_OPENAI_MODEL",
    "ollama_model": "RAG_OLLAMA_MODEL",
    "rerank_model": "RAG_RERANK_MODEL",
    "enable_rerank": "RAG_ENABLE_RERANK",
    "default_parser": "RAG_DEFAULT_PARSER",
    "enable_vision": "RAG_ENABLE_VISION",
    "deploy_mode": "RAG_DEPLOY_MODE",
    "ollama_mode": "RAG_OLLAMA_MODE",
}

_SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = os.getenv("RAG_ENV_FILE", str(_SERVICE_ROOT / ".env"))


def get_env(key: str) -> str | None:
    """Read a value from the dotenvx-encrypted .env file."""
    try:
        result = subprocess.run(
            ["dotenvx", "get", key, "-f", ENV_FILE],
            capture_output=True,
            text=True,
            timeout=10,
        )
        value = result.stdout.strip()
        if result.returncode == 0 and value:
            return value
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def set_env(key: str, value: str) -> bool:
    """Write a value to the dotenvx-encrypted .env file."""
    try:
        result = subprocess.run(
            ["dotenvx", "set", key, value, "-f", ENV_FILE],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            if stderr:
                print(f"[Config] dotenvx set {key} failed: {stderr}")
            return False
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"[Config] dotenvx set {key} failed: {exc}")
        return False


@dataclass(frozen=True)
class EmbeddingPreset:
    label: str
    model: str
    dim: int


EMBEDDING_PRESETS: list[EmbeddingPreset] = [
    EmbeddingPreset("Large (1024d) — best quality", "BAAI/bge-large-en-v1.5", 1024),
    EmbeddingPreset("Base (768d) — balanced", "BAAI/bge-base-en-v1.5", 768),
    EmbeddingPreset("Small (384d) — fastest", "BAAI/bge-small-en-v1.5", 384),
    EmbeddingPreset("M3 (1024d) — multilingual", "BAAI/bge-m3", 1024),
]

OPENAI_PRESETS: list[str] = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "Custom",
]

PARSERS: list[tuple[str, str]] = [
    ("mineru", "MinerU — best for complex PDFs (default)"),
    ("docling", "Docling — IBM, good alternative"),
    ("paddleocr", "PaddleOCR — OCR-focused"),
]


def discover_ollama_models(host: str = "http://localhost:11434") -> list[str]:
    """Auto-discover installed Ollama models via API.

    Returns list of model names, or empty list if Ollama is unreachable.
    """
    try:
        url = f"{host}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        return []
