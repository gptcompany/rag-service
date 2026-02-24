"""Setup step: Service configuration (models, parser, network)."""
from __future__ import annotations

import os
from pathlib import Path

import questionary
from rich.console import Console
from rich.table import Table

from ._config_presets import (
    ENV_VARS,
    EMBEDDING_PRESETS,
    OPENAI_PRESETS,
    PARSERS,
    discover_ollama_models,
    get_env,
    set_env,
)

_SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
RAG_KB_DIR = _SERVICE_ROOT / "data" / "rag_knowledge_base"

# -- Docker file templates ------------------------------------------------

DOCKERFILE_TEMPLATE = """\
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    libreoffice-core curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsS https://dotenvx.sh | sh

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ./raganything

# Models NOT included — mount ~/.cache/huggingface as volume
EXPOSE 8767
CMD ["dotenvx", "run", "-f", ".env", "--", "python", "scripts/raganything_service.py"]
"""

COMPOSE_EXTERNAL_TEMPLATE = """\
services:
  rag:
    build: .
    ports:
      - "${RAG_PORT:-8767}:${RAG_PORT:-8767}"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - ./.env:/app/.env:ro
      - ./.env.keys:/app/.env.keys:ro
      - huggingface_cache:/root/.cache/huggingface
      - ./data:/app/data
    environment:
      - OLLAMA_HOST=${OLLAMA_HOST:-http://host.docker.internal:11434}

volumes:
  huggingface_cache:
"""

COMPOSE_SIDECAR_TEMPLATE = """\
services:
  rag:
    build: .
    ports:
      - "${RAG_PORT:-8767}:${RAG_PORT:-8767}"
    volumes:
      - ./.env:/app/.env:ro
      - ./.env.keys:/app/.env.keys:ro
      - huggingface_cache:/root/.cache/huggingface
      - ./data:/app/data
    environment:
      - OLLAMA_HOST=http://ollama:11434
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

volumes:
  huggingface_cache:
  ollama_data:
"""


def _parse_positive_int(value: str, *, min_value: int = 1, max_value: int | None = None) -> int | None:
    """Parse a positive integer with optional bounds."""
    try:
        parsed = int(value.strip())
    except (AttributeError, ValueError):
        return None
    if parsed < min_value:
        return None
    if max_value is not None and parsed > max_value:
        return None
    return parsed


class ConfigStep:
    name = "Service configuration"

    def check(self) -> bool:
        required_keys = (
            "openai_model",
            "ollama_model",
            "embedding_model",
            "embedding_dim",
            "default_parser",
            "port",
            "host",
        )
        return all(get_env(ENV_VARS[key]) is not None for key in required_keys)

    def install(self, console: Console) -> bool:
        config: dict[str, str] = {}

        # 1/6 OpenAI model
        console.print("\n  [bold cyan]1/6[/] OpenAI model")
        openai_choice = questionary.select(
            "Select OpenAI model:",
            choices=OPENAI_PRESETS,
            default="gpt-4o-mini",
        ).ask()
        if openai_choice is None:
            return False
        if openai_choice == "Custom":
            openai_choice = questionary.text("Enter model name:").ask()
            if not openai_choice:
                return False
        config[ENV_VARS["openai_model"]] = openai_choice

        # 2/6 Ollama model (auto-discovery)
        console.print("\n  [bold cyan]2/6[/] Ollama model (local LLM)")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        models = discover_ollama_models(ollama_host)
        if models:
            ollama_choice = questionary.select(
                "Select Ollama model:",
                choices=models + ["Custom"],
            ).ask()
        else:
            console.print("  [dim]Ollama not reachable, using manual input.[/]")
            ollama_choice = "Custom"

        if ollama_choice is None:
            return False
        if ollama_choice == "Custom":
            ollama_choice = questionary.text(
                "Enter Ollama model name:", default="qwen3:8b"
            ).ask()
            if not ollama_choice:
                return False
        config[ENV_VARS["ollama_model"]] = ollama_choice

        # 3/6 Embedding model
        console.print("\n  [bold cyan]3/6[/] Embedding model")
        embed_choices = [
            questionary.Choice(p.label, value=p) for p in EMBEDDING_PRESETS
        ] + [questionary.Choice("Custom", value="custom")]
        embed_pick = questionary.select(
            "Select embedding model:",
            choices=embed_choices,
        ).ask()

        if embed_pick is None:
            return False
        if embed_pick == "custom":
            embed_model = questionary.text(
                "HuggingFace model name:", default="BAAI/bge-large-en-v1.5"
            ).ask()
            embed_dim = questionary.text(
                "Embedding dimension:", default="1024"
            ).ask()
            if not embed_model or not embed_dim:
                return False
            parsed_embed_dim = _parse_positive_int(embed_dim, min_value=1)
            if parsed_embed_dim is None:
                console.print("  [red]Invalid embedding dimension (must be a positive integer).[/]")
                return False
            config[ENV_VARS["embedding_model"]] = embed_model
            config[ENV_VARS["embedding_dim"]] = str(parsed_embed_dim)
        else:
            config[ENV_VARS["embedding_model"]] = embed_pick.model
            config[ENV_VARS["embedding_dim"]] = str(embed_pick.dim)

        # Warn if changing embedding model with existing knowledge base
        current_embed = get_env(ENV_VARS["embedding_model"])
        new_embed = config[ENV_VARS["embedding_model"]]
        if (
            current_embed
            and current_embed != new_embed
            and RAG_KB_DIR.exists()
            and any(RAG_KB_DIR.iterdir())
        ):
            console.print(
                "  [yellow]Warning: Embedding model changed. "
                "Existing knowledge base will need rebuilding.[/]"
            )

        # 4/6 Reranker toggle
        console.print("\n  [bold cyan]4/6[/] Reranker (bge-reranker-v2-m3)")
        enable_rerank = questionary.confirm(
            "Enable reranker? (improves quality, free local model)",
            default=True,
        ).ask()
        if enable_rerank is None:
            return False
        config[ENV_VARS["enable_rerank"]] = str(enable_rerank).lower()
        if enable_rerank:
            config[ENV_VARS["rerank_model"]] = "BAAI/bge-reranker-v2-m3"

        # 5/6 Parser
        console.print("\n  [bold cyan]5/6[/] Document parser")
        parser_choices = [
            questionary.Choice(desc, value=name) for name, desc in PARSERS
        ]
        parser_choice = questionary.select(
            "Select default parser:",
            choices=parser_choices,
        ).ask()
        if parser_choice is None:
            return False
        config[ENV_VARS["default_parser"]] = parser_choice

        # 6/6 Network
        console.print("\n  [bold cyan]6/6[/] Network")
        port = questionary.text("Service port:", default="8767").ask()
        if port is None:
            return False
        parsed_port = _parse_positive_int(port, min_value=1, max_value=65535)
        if parsed_port is None:
            console.print("  [red]Invalid port (must be an integer between 1 and 65535).[/]")
            return False
        config[ENV_VARS["port"]] = str(parsed_port)
        config[ENV_VARS["host"]] = "0.0.0.0"

        enable_vision = questionary.confirm(
            "Enable vision processing? (uses GPT-4o, expensive)",
            default=False,
        ).ask()
        if enable_vision is None:
            return False
        config[ENV_VARS["enable_vision"]] = str(enable_vision).lower()

        # Review table
        console.print()
        table = Table(title="Configuration Review", show_lines=False)
        table.add_column("Setting", style="bold")
        table.add_column("Value")
        for key, value in config.items():
            table.add_row(key, value)
        console.print(table)

        if not questionary.confirm("Apply this configuration?", default=True).ask():
            console.print("  [yellow]Configuration cancelled.[/]")
            return False

        # Persist all values via dotenvx
        for key, value in config.items():
            if not set_env(key, value):
                console.print(f"  [red]Failed to set {key}[/]")
                return False

        console.print("  [green]Configuration saved to .env[/]")

        # Generate Docker files if in Docker mode
        deploy_mode = get_env(ENV_VARS["deploy_mode"])
        if deploy_mode == "docker":
            self._generate_docker_files(console)

        return True

    def _generate_docker_files(self, console: Console) -> None:
        """Generate Dockerfile and docker-compose.yml."""
        ollama_mode = get_env(ENV_VARS["ollama_mode"]) or "external"

        dockerfile = _SERVICE_ROOT / "Dockerfile"
        if not dockerfile.exists():
            dockerfile.write_text(DOCKERFILE_TEMPLATE)
            console.print("  [green]Generated Dockerfile[/]")

        compose = _SERVICE_ROOT / "docker-compose.yml"
        template = (
            COMPOSE_SIDECAR_TEMPLATE
            if ollama_mode == "sidecar"
            else COMPOSE_EXTERNAL_TEMPLATE
        )
        if compose.exists():
            existing = compose.read_text()
            if existing != template:
                overwrite = questionary.confirm(
                    "docker-compose.yml already exists. Overwrite with wizard template?",
                    default=False,
                ).ask()
                if not overwrite:
                    console.print(
                        "  [yellow]Skipped docker-compose.yml generation (kept existing file)[/]"
                    )
                    return
        compose.write_text(template)
        console.print(
            f"  [green]Generated docker-compose.yml (ollama: {ollama_mode})[/]"
        )

    def verify(self) -> bool:
        return self.check()
