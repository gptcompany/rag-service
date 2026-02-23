"""Setup step: Python venv and raganything installation."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rich.console import Console

SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_DIR = SERVICE_ROOT / ".venv"
RAGANYTHING_DIR = SERVICE_ROOT / "raganything"


class PythonStep:
    name = "Python venv + raganything"

    def check(self) -> bool:
        """Python >= 3.10, venv exists, raganything importable."""
        if sys.version_info < (3, 10):
            return False
        venv_python = VENV_DIR / "bin" / "python3"
        if not venv_python.exists():
            return False
        # Check raganything is importable in the venv
        result = subprocess.run(
            [str(venv_python), "-c", "import raganything; print(raganything.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0

    def install(self, console: Console) -> bool:
        venv_python = VENV_DIR / "bin" / "python3"
        venv_pip = VENV_DIR / "bin" / "pip"

        # Create venv if missing
        if not venv_python.exists():
            console.print("  Creating virtual environment...")
            result = subprocess.run(
                [sys.executable, "-m", "venv", str(VENV_DIR)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                console.print(f"  [red]venv creation failed:[/] {result.stderr}")
                return False

        # Install raganything in editable mode
        console.print("  Installing raganything (editable)...")
        result = subprocess.run(
            [str(venv_pip), "install", "-e", str(RAGANYTHING_DIR)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            console.print(f"  [red]pip install failed:[/] {result.stderr[-500:]}")
            return False

        # Install wizard dependencies too
        console.print("  Installing wizard dependencies (rich, questionary)...")
        result = subprocess.run(
            [str(venv_pip), "install", "rich", "questionary"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            console.print(f"  [red]Wizard deps install failed:[/] {result.stderr[-300:]}")
            return False

        console.print("  [green]Installation complete.[/]")
        return True

    def verify(self) -> bool:
        return self.check()
