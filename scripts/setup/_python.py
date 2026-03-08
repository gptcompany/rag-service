"""Setup step: Python venv and raganything installation."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_DIR = SERVICE_ROOT / ".venv"
RAGANYTHING_DIR = SERVICE_ROOT / "raganything"


class PythonStep:
    name = "Python venv + raganything"
    description = "Create .venv and install raganything + wizard dependencies"

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
        uv_bin = self._find_uv()
        use_uv = uv_bin is not None

        # Create venv if missing
        if not venv_python.exists():
            if use_uv:
                console.print("  Creating virtual environment with uv...")
                result = subprocess.run(
                    [uv_bin, "venv", str(VENV_DIR)],
                    capture_output=True,
                    text=True,
                )
            else:
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
        if use_uv:
            console.print("  Installing raganything (editable) with uv...")
            result = subprocess.run(
                [uv_bin, "pip", "--python", str(venv_python), "install", "-e", str(RAGANYTHING_DIR)],
                capture_output=True,
                text=True,
                timeout=600,
            )
        else:
            console.print("  Installing raganything (editable)...")
            if not self._ensure_venv_pip(console, venv_python, venv_pip):
                return False
            result = subprocess.run(
                [str(venv_pip), "install", "-e", str(RAGANYTHING_DIR)],
                capture_output=True,
                text=True,
                timeout=600,
            )
        if result.returncode != 0:
            console.print(f"  [red]editable install failed:[/] {result.stderr[-500:]}")
            return False

        # Install wizard dependencies too
        if use_uv:
            console.print("  Installing wizard dependencies (rich, questionary) with uv...")
            result = subprocess.run(
                [uv_bin, "pip", "--python", str(venv_python), "install", "rich", "questionary"],
                capture_output=True,
                text=True,
                timeout=120,
            )
        else:
            console.print("  Installing wizard dependencies (rich, questionary)...")
            if not self._ensure_venv_pip(console, venv_python, venv_pip):
                return False
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

    @staticmethod
    def _find_uv() -> str | None:
        uv_bin = shutil.which("uv")
        if uv_bin:
            return uv_bin
        # Common uv install location on macOS/Linux when PATH isn't refreshed.
        home_uv = Path.home() / ".local" / "bin" / "uv"
        if home_uv.exists():
            return str(home_uv)
        return None

    @staticmethod
    def _ensure_venv_pip(console: Console, venv_python: Path, venv_pip: Path) -> bool:
        if venv_pip.exists():
            return True
        console.print("  [yellow]pip missing in venv. Bootstrapping ensurepip...[/]")
        result = subprocess.run(
            [str(venv_python), "-m", "ensurepip", "--upgrade"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            console.print(f"  [red]ensurepip failed:[/] {result.stderr[-400:]}")
            return False
        return venv_pip.exists()
