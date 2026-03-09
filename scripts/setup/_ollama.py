"""Setup step: Ollama LLM runtime and model."""
from __future__ import annotations

import platform
import shutil
import subprocess
import urllib.request
import urllib.error
from urllib.parse import urlparse

import questionary
from rich.console import Console

MODEL_NAME = "qwen3:8b"


class OllamaStep:
    name = "Ollama + qwen3:8b"
    description = "Verify Ollama is running, then pull the model if missing"

    def _ollama_url(self) -> str:
        try:
            from ._config_presets import ENV_VARS, get_env
            return (
                get_env(ENV_VARS["ollama_host"])
                or get_env("OLLAMA_HOST")
                or "http://localhost:11434"
            )
        except Exception:
            return "http://localhost:11434"

    def _is_local_endpoint(self) -> bool:
        host = (urlparse(self._ollama_url()).hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1", ""}

    def _ollama_installed(self) -> bool:
        return shutil.which("ollama") is not None

    def _ollama_serving(self) -> bool:
        try:
            req = urllib.request.Request(self._ollama_url(), method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def _model_exists(self) -> bool:
        if not self._is_local_endpoint() or not self._ollama_installed():
            return self._model_exists_via_api()
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

    def _model_exists_via_api(self) -> bool:
        try:
            req = urllib.request.Request(
                f"{self._ollama_url().rstrip('/')}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                payload = resp.read().decode()
            return MODEL_NAME in payload
        except (urllib.error.URLError, OSError):
            return False

    def _install_ollama_local(self, console: Console) -> bool:
        if platform.system() == "Darwin":
            if shutil.which("port"):
                try:
                    do_port = questionary.confirm(
                        "Install Ollama with MacPorts now?",
                        default=True,
                    ).ask()
                except EOFError:
                    do_port = False
                if do_port:
                    res = subprocess.run(["sudo", "port", "install", "ollama"], check=False)
                    return res.returncode == 0 and self._ollama_installed()
            if shutil.which("brew"):
                try:
                    do_brew = questionary.confirm(
                        "Install Ollama with Homebrew now?",
                        default=True,
                    ).ask()
                except EOFError:
                    do_brew = False
                if do_brew:
                    res = subprocess.run(["brew", "install", "--cask", "ollama"], check=False)
                    return res.returncode == 0 and self._ollama_installed()
            return False
        if platform.system() == "Linux":
            try:
                do_install = questionary.confirm(
                    "Install Ollama now? (official install script)",
                    default=True,
                ).ask()
            except EOFError:
                do_install = False
            if do_install:
                res = subprocess.run(["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"], check=False)
                return res.returncode == 0 and self._ollama_installed()
            return False
        return False

    def check(self) -> bool:
        if self._is_local_endpoint():
            return self._ollama_installed() and self._ollama_serving() and self._model_exists()
        return self._ollama_serving() and self._model_exists()

    def install(self, console: Console) -> bool:
        ollama_url = self._ollama_url()
        local_endpoint = self._is_local_endpoint()

        if local_endpoint and not self._ollama_installed():
            console.print("  [yellow]Ollama is not installed.[/]")
            if not self._install_ollama_local(console):
                console.print("  Install with:")
                console.print("    [bold]curl -fsSL https://ollama.com/install.sh | sh[/]")
                console.print("  Then re-run this setup step.")
                return False

        if not self._ollama_serving():
            if local_endpoint:
                console.print("  [yellow]Ollama is installed but not serving.[/]")
                console.print("  Start it with:")
                console.print("    [bold]ollama serve[/]")
                console.print("  (or enable the systemd service: systemctl --user start ollama)")
            else:
                console.print(f"  [yellow]Remote Ollama endpoint not reachable:[/] {ollama_url}")
                console.print("  Verify OLLAMA_HOST and network reachability.")
            return False

        if not self._model_exists():
            if local_endpoint:
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
            console.print(
                f"  [yellow]Model {MODEL_NAME} not found on remote Ollama endpoint {ollama_url}.[/]"
            )
            console.print(f"  Run on remote host: [bold]ollama pull {MODEL_NAME}[/]")
            return False

        return True

    def verify(self) -> bool:
        return self.check()
