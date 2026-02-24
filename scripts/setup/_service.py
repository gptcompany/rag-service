"""Setup step: RAG service health check and startup guidance."""
from __future__ import annotations

import os
import urllib.request
import urllib.error
from pathlib import Path

from rich.console import Console

_SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
START_SCRIPT = str(_SERVICE_ROOT / "scripts" / "raganything_start.sh")
SYSTEMD_UNIT = "raganything.service"


def _get_port() -> str:
    from ._config_presets import get_env, ENV_VARS
    return get_env(ENV_VARS["port"]) or os.getenv("RAG_PORT", "8767")


def _get_deploy_mode() -> str:
    from ._config_presets import get_env, ENV_VARS
    return get_env(ENV_VARS["deploy_mode"]) or "host"


class ServiceStep:
    name = "RAG Service"

    def _health_ok(self) -> bool:
        port = _get_port()
        try:
            url = f"http://localhost:{port}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def check(self) -> bool:
        return self._health_ok()

    def install(self, console: Console) -> bool:
        port = _get_port()
        deploy_mode = _get_deploy_mode()

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
            console.print("    [dim]Create with:[/]")
            console.print(f"    [bold]sudo systemctl enable --now {SYSTEMD_UNIT}[/]")
            console.print()
            console.print("    Example unit file contents:")
            console.print("    [dim][Unit][/]")
            console.print("    [dim]Description=RAGAnything HTTP Service[/]")
            console.print("    [dim]After=network.target[/]")
            console.print("    [dim][Service][/]")
            console.print(f"    [dim]ExecStart={START_SCRIPT}[/]")
            console.print("    [dim]Restart=on-failure[/]")
            console.print("    [dim]User=sam[/]")
            console.print("    [dim][Install][/]")
            console.print("    [dim]WantedBy=multi-user.target[/]")

        console.print()
        console.print("  [dim]Start the service, then re-run this step to verify.[/]")
        return False

    def verify(self) -> bool:
        return self.check()
