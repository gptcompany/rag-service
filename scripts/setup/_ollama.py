"""Setup step: Ollama LLM runtime and model."""
from __future__ import annotations

import shutil
import subprocess
import urllib.request
import urllib.error

from rich.console import Console

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:8b"


class OllamaStep:
    name = "Ollama + qwen3:8b"
    description = "Verify Ollama is running, then pull the model if missing"

    def _ollama_installed(self) -> bool:
        return shutil.which("ollama") is not None

    def _ollama_serving(self) -> bool:
        try:
            req = urllib.request.Request(OLLAMA_URL, method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def _model_exists(self) -> bool:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return MODEL_NAME in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def check(self) -> bool:
        return self._ollama_installed() and self._ollama_serving() and self._model_exists()

    def install(self, console: Console) -> bool:
        if not self._ollama_installed():
            console.print("  [yellow]Ollama is not installed.[/]")
            console.print("  Install with:")
            console.print("    [bold]curl -fsSL https://ollama.com/install.sh | sh[/]")
            console.print("  Then re-run this setup step.")
            return False

        if not self._ollama_serving():
            console.print("  [yellow]Ollama is installed but not serving.[/]")
            console.print("  Start it with:")
            console.print("    [bold]ollama serve[/]")
            console.print("  (or enable the systemd service: systemctl --user start ollama)")
            return False

        if not self._model_exists():
            console.print(f"  Pulling model {MODEL_NAME}...")
            console.print("  This may take several minutes on first download.")
            result = subprocess.run(
                ["ollama", "pull", MODEL_NAME],
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min for large model download
            )
            if result.returncode != 0:
                console.print(f"  [red]ollama pull failed:[/] {result.stderr[-500:]}")
                return False
            console.print(f"  [green]Model {MODEL_NAME} pulled successfully.[/]")
            return True

        return True

    def verify(self) -> bool:
        return self.check()
