"""Setup step: Full service verification and status display."""
from __future__ import annotations

import json
import urllib.request
import urllib.error

from rich.console import Console
from rich.table import Table

SERVICE_URL = "http://localhost:8767"


class VerifyStep:
    name = "Service verification"

    def check(self) -> bool:
        # Always run full verification -- never skip
        return False

    def _fetch_json(self, path: str) -> dict | None:
        try:
            req = urllib.request.Request(f"{SERVICE_URL}{path}", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None

    def install(self, console: Console) -> bool:
        console.print("  Running full service verification...\n")

        # Health check
        health = self._fetch_json("/health")
        status_data = self._fetch_json("/status")

        table = Table(title="RAG Service Status", show_lines=True)
        table.add_column("Component", style="bold")
        table.add_column("Status")
        table.add_column("Details")

        if health is None:
            table.add_row(
                "Service",
                "[red]DOWN[/]",
                "Cannot reach localhost:8767",
            )
            console.print(table)
            console.print("\n  [yellow]Start the service first, then re-run verify.[/]")
            return True  # Informational -- don't block

        # Parse health response
        h_status = health.get("status", "unknown")
        color = "green" if h_status == "ok" else "yellow"
        table.add_row(
            "Health",
            f"[{color}]{h_status.upper()}[/]",
            f"port 8767",
        )

        if status_data:
            # Circuit breaker
            cb = status_data.get("circuit_breaker", {})
            cb_state = cb.get("state", "unknown")
            cb_color = "green" if cb_state == "closed" else "red"
            cb_failures = cb.get("failure_count", "?")
            table.add_row(
                "Circuit Breaker",
                f"[{cb_color}]{cb_state.upper()}[/]",
                f"failures: {cb_failures}",
            )

            # Job queue
            jobs = status_data.get("jobs", {})
            active = jobs.get("active", 0)
            queued = jobs.get("queued", 0)
            completed = jobs.get("completed", 0)
            table.add_row(
                "Job Queue",
                f"[green]OK[/]",
                f"active: {active}, queued: {queued}, completed: {completed}",
            )

            # LLM backend
            llm = status_data.get("llm_backend", status_data.get("llm", "unknown"))
            table.add_row("LLM Backend", "[cyan]INFO[/]", str(llm))

            # Parser
            parser = status_data.get("parser", "unknown")
            table.add_row("Parser", "[cyan]INFO[/]", str(parser))
        else:
            table.add_row(
                "Status endpoint",
                "[yellow]N/A[/]",
                "/status not available",
            )

        console.print(table)
        return True  # Informational step

    def verify(self) -> bool:
        # Always passes -- informational only
        return True
