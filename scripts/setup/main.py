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
}


def main(args: list[str] | None = None) -> int:
    if args is None:
        args = sys.argv[1:]

    console = Console()
    console.print(BANNER, style="bold cyan")

    # Help
    if args and args[0] in ("-h", "--help", "help"):
        console.print("[bold]Usage:[/] python -m scripts.setup [subcommand]\n")
        console.print("[bold]Subcommands:[/]")
        console.print("  [dim](no args)[/]   Interactive setup menu (free navigation)")
        for name, (_, desc) in SUBCOMMANDS.items():
            console.print(f"  [bold]{name:10s}[/] {desc}")
        return 0

    # Select steps
    if args and args[0] in SUBCOMMANDS:
        factory, desc = SUBCOMMANDS[args[0]]
        console.print(f"[bold]{desc}[/]\n")
        steps = factory()
    elif args:
        console.print(f"[red]Unknown subcommand:[/] {args[0]}")
        console.print(f"Valid: {', '.join(SUBCOMMANDS.keys())}")
        return 1
    else:
        console.print("[bold]Interactive setup menu[/]\n")
        steps = _all_steps()
        ok = run_interactive_menu(steps, console)
        return 0 if ok else 1

    ok = run_steps(steps, console)
    return 0 if ok else 1
