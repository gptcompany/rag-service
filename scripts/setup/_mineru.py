"""Setup step: MinerU model download."""
from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console

_SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PYTHON = _SERVICE_ROOT / ".venv" / "bin" / "python3"

# MinerU models expected in HuggingFace cache
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
MINERU_MODEL_PREFIXES = [
    "models--opendatalab--PDF-Extract-Kit",
]


class MineruStep:
    name = "MinerU models"
    description = "Download and verify MinerU PDF extraction models (~2GB)"

    def check(self) -> bool:
        """Check if MinerU model files exist in HuggingFace cache."""
        if not HF_CACHE.exists():
            return False
        cached_dirs = [d.name for d in HF_CACHE.iterdir() if d.is_dir()]
        for prefix in MINERU_MODEL_PREFIXES:
            if not any(d.startswith(prefix) for d in cached_dirs):
                return False
        return True

    def install(self, console: Console) -> bool:
        if not VENV_PYTHON.exists():
            console.print("  [red]Python venv not found. Run Python step first.[/]")
            return False

        console.print("  Downloading MinerU models (~2GB)...")
        console.print("  This may take several minutes on first run.")

        # Trigger MinerU model download via its model loading code
        result = subprocess.run(
            [
                str(VENV_PYTHON),
                "-c",
                (
                    "from mineru.model.doc_analyze_by_custom_model "
                    "import ModelSingleton; "
                    "ModelSingleton().get_model()"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min for large model download
        )
        if result.returncode != 0:
            console.print(f"  [red]MinerU model download failed:[/]")
            # Show last 500 chars of stderr for diagnostics
            stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
            console.print(f"  {stderr_tail}")
            return False

        console.print("  [green]MinerU models downloaded.[/]")
        return True

    def verify(self) -> bool:
        return self.check()
