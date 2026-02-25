"""RAG Service Setup Wizard -- CLI entry point."""
from __future__ import annotations

import sys

from rich.console import Console

from ._runner import run_interactive_menu, run_steps
from ._deploy import DeployStep
from ._python import PythonStep
from ._mineru import MineruStep
from ._ollama import OllamaStep
from ._libreoffice import LibreOfficeStep
from ._secrets import SecretsStep
from ._config import ConfigStep
from ._service import ServiceStep
from ._verify import VerifyStep

BANNER = r"""
  ___    _    ___   ___              _
 | _ \  /_\  / __| / __| ___ _ ___ _(_) __ ___
 |   / / _ \| (_ | \__ \/ -_) '_\ V / |/ _/ -_)
 |_|_\/_/ \_\\___| |___/\___|_|  \_/|_|\__\___|
                          Setup Wizard
"""


def _docker_skip() -> bool:
    """Skip step when deployment mode is Docker."""
    from ._config_presets import get_env, ENV_VARS
    return get_env(ENV_VARS["deploy_mode"]) == "docker"


def _docker_sidecar_skip() -> bool:
    """Skip step when Docker + Ollama sidecar."""
    from ._config_presets import get_env, ENV_VARS
    return (
        get_env(ENV_VARS["deploy_mode"]) == "docker"
        and get_env(ENV_VARS["ollama_mode"]) == "sidecar"
    )


def _all_steps() -> list:
    python_step = PythonStep()
    mineru_step = MineruStep()
    ollama_step = OllamaStep()
    libreoffice_step = LibreOfficeStep()

    # In Docker mode, these are handled by the container image
    python_step.skip_when = _docker_skip
    mineru_step.skip_when = _docker_skip
    libreoffice_step.skip_when = _docker_skip
    # Ollama skips only in Docker + sidecar (external still needs host Ollama)
    ollama_step.skip_when = _docker_sidecar_skip

    return [
        DeployStep(),
        python_step,
        mineru_step,
        ollama_step,
        libreoffice_step,
        SecretsStep(),
        ConfigStep(),
        ServiceStep(),
        VerifyStep(),
    ]


def _deps_steps() -> list:
    return [
        PythonStep(),
        MineruStep(),
        OllamaStep(),
        LibreOfficeStep(),
    ]


def _models_steps() -> list:
    return [
        MineruStep(),
        OllamaStep(),
    ]


def _config_steps() -> list:
    return [ConfigStep()]


def _deploy_steps() -> list:
    return [DeployStep()]


def _service_steps() -> list:
    return [ServiceStep()]


def _verify_steps() -> list:
    return [VerifyStep()]


SUBCOMMANDS = {
    "deps": (_deps_steps, "Install dependencies: Python venv, MinerU, Ollama, LibreOffice"),
    "models": (_models_steps, "Download/verify AI models: MinerU + Ollama"),
    "config": (_config_steps, "Configure service: models, parser, network"),
    "deploy": (_deploy_steps, "Choose deployment mode: host or Docker"),
    "service": (_service_steps, "Check RAG service health and startup"),
    "verify": (_verify_steps, "Full service verification and status display"),
    "get": (None, "Show configuration variables (e.g., 'rag-setup get' or 'rag-setup get RAG_PORT')"),
    "set": (None, "Set a configuration variable (e.g., 'rag-setup set RAG_PORT 8080')"),
}


def _handle_get(args: list[str], console: Console) -> int:
    from ._config_presets import get_env, ENV_VARS
    from rich.table import Table

    if not args:
        table = Table(title="RAG Service Configuration")
        table.add_column("Key")
        table.add_column("Value")
        # Show all known RAG_ vars
        for key in sorted(ENV_VARS.values()):
            val = get_env(key)
            table.add_row(key, val or "[dim]not set[/]")
        # Also include OPENAI_API_KEY which is essential but not in ENV_VARS mapping
        val = get_env("OPENAI_API_KEY")
        table.add_row("OPENAI_API_KEY", val or "[dim]not set[/]")
        console.print(table)
        return 0

    key = args[0]
    val = get_env(key)
    if val:
        # Just print the value for easy shell consumption
        print(val)
        return 0
    else:
        console.print(f"[red]Error:[/] Key '{key}' not found or not set.")
        return 1


def _handle_set(args: list[str], console: Console) -> int:
    from ._config_presets import set_env, ENV_VARS
    from ._config import _parse_positive_int

    if len(args) < 2:
        console.print("[red]Usage:[/] rag-setup set <KEY> <VALUE>")
        return 1

    key, value = args[0], args[1]

    # Validation for known keys
    if key == "RAG_PORT":
        if _parse_positive_int(value, min_value=1, max_value=65535) is None:
            console.print(f"[red]Error:[/] Invalid port (must be 1-65535): {value}")
            return 1
    elif key == "RAG_EMBEDDING_DIM":
        if _parse_positive_int(value, min_value=1) is None:
            console.print(f"[red]Error:[/] Invalid embedding dimension: {value}")
            return 1
    elif key == "RAG_DEPLOY_MODE":
        if value not in ("host", "docker"):
            console.print("[red]Error:[/] RAG_DEPLOY_MODE must be 'host' or 'docker'")
            return 1
    elif key == "RAG_OLLAMA_MODE":
        if value not in ("external", "sidecar"):
            console.print("[red]Error:[/] RAG_OLLAMA_MODE must be 'external' or 'sidecar'")
            return 1

    if set_env(key, value):
        console.print(f"[green]Successfully set {key}={value}[/]")
        return 0
    else:
        console.print(f"[red]Failed to set {key}[/]")
        return 1


def main(args: list[str] | None = None) -> int:
    if args is None:
        args = sys.argv[1:]

    console = Console()
    console.print(BANNER, style="bold cyan")

    # Help
    if args and args[0] in ("-h", "--help", "help"):
        console.print("[bold]Usage:[/] rag-setup [subcommand] [args...]\n")
        console.print("[bold]Subcommands:[/]")
        console.print("  [dim](no args)[/]   Interactive setup menu (free navigation)")
        for name, (_, desc) in SUBCOMMANDS.items():
            console.print(f"  [bold]{name:10s}[/] {desc}")
        return 0

    # Select subcommands or interactive menu
    if not args:
        console.print("[bold]Interactive setup menu[/]\n")
        steps = _all_steps()
        ok = run_interactive_menu(steps, console)
        return 0 if ok else 1

    cmd = args[0]
    if cmd == "get":
        return _handle_get(args[1:], console)
    if cmd == "set":
        return _handle_set(args[1:], console)

    if cmd in SUBCOMMANDS:
        factory, desc = SUBCOMMANDS[cmd]
        if factory is None:  # Should not happen given the logic above
            return 1
        console.print(f"[bold]{desc}[/]\n")
        steps = factory()
        ok = run_steps(steps, console)
        return 0 if ok else 1
    else:
        console.print(f"[red]Unknown subcommand:[/] {cmd}")
        console.print(f"Valid: {', '.join(SUBCOMMANDS.keys())}")
        return 1
