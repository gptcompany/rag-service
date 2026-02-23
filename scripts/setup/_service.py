"""Setup step: RAG service health check and startup guidance."""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path

from rich.console import Console

SERVICE_URL = "http://localhost:8767"
HEALTH_ENDPOINT = f"{SERVICE_URL}/health"
_SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
START_SCRIPT = str(_SERVICE_ROOT / "scripts" / "raganything_start.sh")
SYSTEMD_UNIT = "raganything.service"


class ServiceStep:
    name = "RAG Service"

    def _health_ok(self) -> bool:
        try:
            req = urllib.request.Request(HEALTH_ENDPOINT, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def check(self) -> bool:
        return self._health_ok()

    def install(self, console: Console) -> bool:
        console.print("  [yellow]RAG service is not running on port 8767.[/]")
        console.print()
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
