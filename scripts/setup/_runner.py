"""Cascade runner: check -> install -> verify for each setup step."""
from __future__ import annotations

from typing import Protocol

import questionary
from rich.console import Console


class SetupStep(Protocol):
    name: str

    def check(self) -> bool: ...
    def install(self, console: Console) -> bool: ...
    def verify(self) -> bool: ...


def run_steps(steps: list[SetupStep], console: Console) -> bool:
    results: list[tuple[str, str]] = []
    for step in steps:
        with console.status(f"[bold cyan]Checking {step.name}...[/]"):
            if step.check():
                console.print(
                    f"  [green]\u2705 {step.name}[/] \u2014 already configured"
                )
                results.append((step.name, "ok"))
                continue
        if not questionary.confirm(f"Configure {step.name}?", default=True).ask():
            console.print(
                f"  [yellow]\u23ed\ufe0f  {step.name}[/] \u2014 skipped"
            )
            results.append((step.name, "skipped"))
            continue
        success = step.install(console)
        if not success:
            action = questionary.select(
                f"{step.name} failed. What to do?",
                choices=["Skip and continue", "Retry", "Abort"],
            ).ask()
            if action == "Abort":
                console.print("[bold red]Setup aborted.[/]")
                return False
            if action == "Retry":
                success = step.install(console)
                if not success:
                    console.print(
                        f"  [red]\u274c {step.name}[/] \u2014 retry failed, skipping"
                    )
                    results.append((step.name, "failed"))
                    continue
            else:
                results.append((step.name, "skipped"))
                continue
        if step.verify():
            console.print(f"  [green]\u2705 {step.name}[/] \u2014 verified!")
            results.append((step.name, "ok"))
        else:
            console.print(
                f"  [yellow]\u26a0\ufe0f  {step.name}[/] \u2014 installed but verify failed"
            )
            results.append((step.name, "warn"))
    console.print()
    _print_summary(results, console)
    return all(s != "failed" for _, s in results)


def _print_summary(results: list[tuple[str, str]], console: Console) -> None:
    from rich.table import Table

    table = Table(title="Setup Summary", show_lines=False)
    table.add_column("Step", style="bold")
    table.add_column("Status")
    status_map = {
        "ok": "[green]\u2705 OK[/]",
        "skipped": "[yellow]\u23ed\ufe0f  Skipped[/]",
        "failed": "[red]\u274c Failed[/]",
        "warn": "[yellow]\u26a0\ufe0f  Warning[/]",
    }
    for name, status in results:
        table.add_row(name, status_map.get(status, status))
    console.print(table)
