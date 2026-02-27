"""Setup step: RAG service health check and startup guidance."""
from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
import urllib.error
import getpass
from pathlib import Path

from rich.console import Console

_SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
START_SCRIPT = str(_SERVICE_ROOT / "scripts" / "raganything_start.sh")
UPDATE_SYSTEMD_SCRIPT = str(_SERVICE_ROOT / "update-systemd.sh")
SYSTEMD_UNIT = "raganything.service"


def _get_port() -> str:
    from ._config_presets import get_env, ENV_VARS
    return get_env(ENV_VARS["port"]) or os.getenv("RAG_PORT", "8767")


def _get_deploy_mode() -> str:
    from ._config_presets import get_env, ENV_VARS
    return get_env(ENV_VARS["deploy_mode"]) or "host"


class ServiceStep:
    name = "RAG Service"
    description = "Check /health and show startup instructions for host or Docker"

    def _health_ok(self) -> bool:
        port = _get_port()
        try:
            url = f"http://localhost:{port}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    @staticmethod
    def _systemd_enabled() -> bool:
        """Return True when raganything service is enabled for boot."""
        if not shutil.which("systemctl"):
            return False

        for cmd in (
            ["systemctl", "is-enabled", SYSTEMD_UNIT],
            ["systemctl", "--user", "is-enabled", SYSTEMD_UNIT],
        ):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except OSError:
                continue
            if result.returncode == 0 and result.stdout.strip().startswith("enabled"):
                return True
        return False

    def check(self) -> bool:
        if not self._health_ok():
            return False
        # In host mode, insist on boot persistence as part of setup "done".
        if _get_deploy_mode() == "host":
            return self._systemd_enabled()
        return True

    def install(self, console: Console) -> bool:
        port = _get_port()
        deploy_mode = _get_deploy_mode()
        health_ok = self._health_ok()
        persistent = self._systemd_enabled() if deploy_mode == "host" else True

        if health_ok and not persistent:
            console.print(
                "  [yellow]RAG service is running but not boot-persistent (systemd not enabled).[/]"
            )
        else:
            console.print(f"  [yellow]RAG service is not running on port {port}.[/]")
        console.print()

        if deploy_mode == "docker":
            console.print("  [bold]Start with Docker Compose:[/]")
            console.print("    [bold]docker compose up -d[/]")
            console.print()
            console.print("  [dim]Check status: docker compose ps[/]")
            console.print("  [dim]View logs: docker compose logs -f rag[/]")
        else:
            console.print("  [bold]Option 1: Foreground (development)[/]")
            console.print(f"    [bold]{START_SCRIPT}[/]")
            console.print()
            console.print("  [bold]Option 2: systemd (production)[/]")
            console.print(f"    [dim]Unit file: /etc/systemd/system/{SYSTEMD_UNIT}[/]")
            console.print("    [dim]Install/update unit with:[/]")
            console.print(f"    [bold]sudo bash {UPDATE_SYSTEMD_SCRIPT}[/]")
            console.print("    [dim]Or enable existing unit with:[/]")
            console.print(f"    [bold]sudo systemctl enable --now {SYSTEMD_UNIT}[/]")
            console.print()
            console.print("    Example unit file contents:")
            console.print("    [dim][Unit][/]")
            console.print("    [dim]Description=RAGAnything HTTP Service[/]")
            console.print("    [dim]After=network.target[/]")
            console.print("    [dim][Service][/]")
            console.print(f"    [dim]ExecStart={START_SCRIPT}[/]")
            console.print("    [dim]Restart=on-failure[/]")
            console.print(f"    [dim]User={getpass.getuser()}[/]")
            console.print("    [dim][Install][/]")
            console.print("    [dim]WantedBy=multi-user.target[/]")

        console.print()
        console.print(
            "  [dim]Make the service boot-persistent, then re-run this step to verify.[/]"
        )
        return False

    def verify(self) -> bool:
        return self.check()
