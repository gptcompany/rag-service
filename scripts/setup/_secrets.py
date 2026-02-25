"""Setup step: Verify required secrets (OPENAI_API_KEY via dotenvx)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from rich.console import Console

_SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = os.getenv("RAG_ENV_FILE", str(_SERVICE_ROOT / ".env"))


class SecretsStep:
    name = "Secrets (OPENAI_API_KEY)"
    description = "Ensure OPENAI_API_KEY exists in dotenvx-managed .env secrets"

    def _key_exists(self) -> bool:
        """Check if OPENAI_API_KEY is set in dotenvx. NEVER reveals the value."""
        try:
            result = subprocess.run(
                ["dotenvx", "get", "OPENAI_API_KEY", "-f", ENV_FILE],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0 and bool((result.stdout or "").strip())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def check(self) -> bool:
        return self._key_exists()

    def install(self, console: Console) -> bool:
        console.print("  [yellow]OPENAI_API_KEY not found in dotenvx secrets.[/]")
        console.print()
        console.print("  Manually add to the dotenvx-encrypted env file:")
        console.print(f"    [bold]dotenvx set OPENAI_API_KEY <your-key> -f {ENV_FILE}[/]")
        console.print()
        console.print("  [dim]The key is needed for GPT-4o-mini embeddings and queries.[/]")
        console.print("  [dim]Re-run this step after adding the key.[/]")
        # Cannot auto-install secrets -- user must provide
        return False

    def verify(self) -> bool:
        return self.check()
