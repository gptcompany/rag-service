"""Cascade runner: check -> install -> verify for each setup step."""
from __future__ import annotations

from typing import Protocol

import questionary
from rich.console import Console
from rich.table import Table


class SetupStep(Protocol):
    name: str
    description: str

    def check(self) -> bool: ...
    def install(self, console: Console) -> bool: ...
    def verify(self) -> bool: ...


def run_steps(steps: list[SetupStep], console: Console) -> bool:
    """Legacy linear runner used by subcommands."""
    results: list[tuple[str, str]] = []
    for step in steps:
        skip_fn = getattr(step, "skip_when", None)
        if callable(skip_fn) and skip_fn():
            console.print(
                f"  [dim]⏭️  {step.name}[/] — not needed for this deployment"
            )
            results.append((step.name, "skipped"))
            continue
        with console.status(f"[bold cyan]Checking {step.name}...[/]"):
            if step.check():
                console.print(
                    f"  [green]✅ {step.name}[/] — already configured"
                )
                results.append((step.name, "ok"))
                continue
        if not questionary.confirm(f"Configure {step.name}?", default=True).ask():
            console.print(
                f"  [yellow]⏭️  {step.name}[/] — skipped"
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
                        f"  [red]❌ {step.name}[/] — retry failed, skipping"
                    )
                    results.append((step.name, "failed"))
                    continue
            else:
                results.append((step.name, "skipped"))
                continue
        if step.verify():
            console.print(f"  [green]✅ {step.name}[/] — verified!")
            results.append((step.name, "ok"))
        else:
            console.print(
                f"  [yellow]⚠️  {step.name}[/] — installed but verify failed"
            )
            results.append((step.name, "warn"))
    console.print()
    _print_summary(results, console)
    return all(status != "failed" for _, status in results)


def run_interactive_menu(steps: list[SetupStep], console: Console) -> bool:
    """Interactive menu with free navigation and a 'run all pending' action."""
    _print_welcome_guide(steps, console)
    last_outcomes: dict[str, str] = {}

    while True:
        statuses = _collect_menu_statuses(steps, console, last_outcomes)
        console.print()
        _print_menu(steps, statuses, console)
        choices = _build_menu_choices(steps, statuses)

        try:
            selection = questionary.select("Select step:", choices=choices).ask()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted (Ctrl+C).[/]")
            final_statuses = _collect_menu_statuses(steps, console, last_outcomes)
            console.print()
            _print_summary(
                [(step.name, status) for step, status in zip(steps, final_statuses)],
                console,
            )
            return False

        if selection is None:
            selection = ("exit", None)

        action, index = selection
        if action == "exit":
            break

        if action == "run_all":
            pending_indices = [
                idx for idx, status in enumerate(_collect_menu_statuses(steps, console, last_outcomes))
                if status in {"pending", "failed", "warn"}
            ]
            for pending_idx in pending_indices:
                if not _run_menu_step(steps[pending_idx], console, last_outcomes):
                    final_statuses = _collect_menu_statuses(steps, console, last_outcomes)
                    console.print()
                    _print_summary(
                        [(step.name, status) for step, status in zip(steps, final_statuses)],
                        console,
                    )
                    return False
            continue

        if index is None:
            continue

        if not _run_menu_step(steps[index], console, last_outcomes):
            final_statuses = _collect_menu_statuses(steps, console, last_outcomes)
            console.print()
            _print_summary(
                [(step.name, status) for step, status in zip(steps, final_statuses)],
                console,
            )
            return False

    final_statuses = _collect_menu_statuses(steps, console, last_outcomes)
    console.print()
    _print_summary(
        [(step.name, status) for step, status in zip(steps, final_statuses)],
        console,
    )
    return "failed" not in final_statuses


def _run_menu_step(step: SetupStep, console: Console, last_outcomes: dict[str, str]) -> bool:
    """Run a single step selected from the menu."""
    skip_fn = getattr(step, "skip_when", None)
    if callable(skip_fn) and skip_fn():
        console.print(
            f"  [dim]⏭️  {step.name}[/] — not needed for this deployment"
        )
        last_outcomes.pop(step.name, None)
        return True

    with console.status(f"[bold cyan]Checking {step.name}...[/]"):
        if step.check():
            console.print(f"  [green]✅ {step.name}[/] — already configured")
            last_outcomes.pop(step.name, None)
            return True

    try:
        success = step.install(console)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted while running step.[/]")
        return False

    while not success:
        try:
            action = questionary.select(
                f"{step.name} failed. What to do?",
                choices=["Skip and continue", "Retry", "Abort"],
            ).ask()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted while handling step failure.[/]")
            return False

        if action in (None, "Abort"):
            console.print("[bold red]Setup aborted.[/]")
            last_outcomes[step.name] = "failed"
            return False
        if action == "Skip and continue":
            console.print(
                f"  [yellow]⏭️  {step.name}[/] — skipped"
            )
            # Keep the step pending in the menu; skipping is not a terminal state.
            last_outcomes.pop(step.name, None)
            return True
        if action == "Retry":
            try:
                success = step.install(console)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted while retrying step.[/]")
                return False

    if step.verify():
        console.print(f"  [green]✅ {step.name}[/] — verified!")
        last_outcomes.pop(step.name, None)
    else:
        console.print(
            f"  [yellow]⚠️  {step.name}[/] — installed but verify failed"
        )
        last_outcomes[step.name] = "warn"
    return True


def _collect_menu_statuses(
    steps: list[SetupStep],
    console: Console,
    last_outcomes: dict[str, str],
) -> list[str]:
    statuses: list[str] = []
    for step in steps:
        skip_fn = getattr(step, "skip_when", None)
        if callable(skip_fn) and skip_fn():
            last_outcomes.pop(step.name, None)
            statuses.append("not-needed")
            continue

        try:
            ok = step.check()
        except Exception as exc:  # pragma: no cover - defensive fallback
            console.print(f"  [red]Check failed for {step.name}:[/] {exc}")
            last_outcomes[step.name] = "failed"
            statuses.append("failed")
            continue

        if ok:
            last_outcomes.pop(step.name, None)
            statuses.append("ok")
            continue

        statuses.append(last_outcomes.get(step.name, "pending"))
    return statuses


def _build_menu_choices(steps: list[SetupStep], statuses: list[str]) -> list[questionary.Choice]:
    choices: list[questionary.Choice] = []
    for idx, (step, status) in enumerate(zip(steps, statuses), start=1):
        choices.append(
            questionary.Choice(
                title=f"{idx:>2}. {step.name} ({_menu_status_label(status)})",
                value=("step", idx - 1),
            )
        )

    if any(status in {"pending", "failed", "warn"} for status in statuses):
        choices.append(
            questionary.Choice(
                title="Run all pending steps",
                value=("run_all", None),
            )
        )

    choices.append(questionary.Choice(title="Exit", value=("exit", None)))
    return choices


def _print_welcome_guide(steps: list[SetupStep], console: Console) -> None:
    console.print("[bold]Welcome Guide[/]")
    console.print("Use the menu to run any step in any order. Status refreshes after each action.")
    for idx, step in enumerate(steps, start=1):
        console.print(f"  [bold cyan]{idx}.[/] {step.name} — {step.description}")


def _print_menu(steps: list[SetupStep], statuses: list[str], console: Console) -> None:
    table = Table(title="RAG Service Setup Wizard", show_lines=False)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Step", style="bold")
    table.add_column("Description")
    table.add_column("Status")
    for idx, (step, status) in enumerate(zip(steps, statuses), start=1):
        table.add_row(
            str(idx),
            step.name,
            step.description,
            _menu_status_label(status),
        )
    console.print(table)


def _menu_status_label(status: str) -> str:
    return {
        "ok": "✅ OK",
        "pending": "⬜ Pending",
        "failed": "❌ Failed",
        "warn": "⚠️ Warning",
        "not-needed": "⏭️ Skipped (mode)",
    }.get(status, status)


def _print_summary(results: list[tuple[str, str]], console: Console) -> None:
    table = Table(title="Setup Summary", show_lines=False)
    table.add_column("Step", style="bold")
    table.add_column("Status")
    status_map = {
        "ok": "[green]✅ OK[/]",
        "skipped": "[yellow]⏭️  Skipped[/]",
        "failed": "[red]❌ Failed[/]",
        "warn": "[yellow]⚠️  Warning[/]",
        "pending": "[yellow]⬜ Pending[/]",
        "not-needed": "[dim]⏭️  Not needed[/]",
    }
    for name, status in results:
        table.add_row(name, status_map.get(status, status))
    console.print(table)
