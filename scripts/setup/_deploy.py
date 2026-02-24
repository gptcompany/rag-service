"""Setup step: Deployment mode (host or Docker)."""
from __future__ import annotations

import questionary
from rich.console import Console

from ._config_presets import ENV_VARS, get_env, set_env


class DeployStep:
    name = "Deployment mode"

    def check(self) -> bool:
        return get_env(ENV_VARS["deploy_mode"]) is not None

    def install(self, console: Console) -> bool:
        mode = questionary.select(
            "Deployment mode:",
            choices=[
                questionary.Choice(
                    "Host — install directly on this machine (systemd)",
                    value="host",
                ),
                questionary.Choice(
                    "Docker — generate Dockerfile + docker-compose.yml",
                    value="docker",
                ),
            ],
        ).ask()
        if mode is None:
            return False

        ollama_mode = "external"
        if mode == "docker":
            ollama_mode = questionary.select(
                "Ollama setup:",
                choices=[
                    questionary.Choice(
                        "External — use Ollama already running on host",
                        value="external",
                    ),
                    questionary.Choice(
                        "Sidecar — dedicated Ollama container in docker-compose",
                        value="sidecar",
                    ),
                ],
            ).ask()
            if ollama_mode is None:
                return False

        set_env(ENV_VARS["deploy_mode"], mode)
        set_env(ENV_VARS["ollama_mode"], ollama_mode)

        console.print(f"  Deploy mode: [bold]{mode}[/]")
        if mode == "docker":
            console.print(f"  Ollama: [bold]{ollama_mode}[/]")
        return True

    def verify(self) -> bool:
        return self.check()
