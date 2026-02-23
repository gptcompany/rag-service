"""Setup step: Verify required secrets (OPENAI_API_KEY via dotenvx)."""
from __future__ import annotations

import subprocess

from rich.console import Console

ENV_FILE = "/media/sam/1TB/.env"


class SecretsStep:
    name = "Secrets (OPENAI_API_KEY)"

    def _key_exists(self) -> bool:
        """Check if OPENAI_API_KEY is set in dotenvx. NEVER reveals the value."""
        try:
            result = subprocess.run(
                ["bash", "-c",
                 f'dotenvx get OPENAI_API_KEY -f {ENV_FILE} 2>/dev/null | grep -q .'],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def check(self) -> bool:
        return self._key_exists()

    def install(self, console: Console) -> bool:
        console.print("  [yellow]OPENAI_API_KEY not found in dotenvx secrets.[/]")
        console.print()
        console.print("  Add it with the secret-add helper:")
        console.print(f"    [bold]secret-add OPENAI_API_KEY[/]")
        console.print()
        console.print("  Or manually add to the dotenvx-encrypted env file:")
        console.print(f"    [bold]dotenvx set OPENAI_API_KEY <your-key> -f {ENV_FILE}[/]")
        console.print()
        console.print("  [dim]The key is needed for GPT-4o-mini embeddings and queries.[/]")
        console.print("  [dim]Re-run this step after adding the key.[/]")
        # Cannot auto-install secrets -- user must provide
        return False

    def verify(self) -> bool:
        return self.check()
