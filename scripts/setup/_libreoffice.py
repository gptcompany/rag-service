"""Setup step: LibreOffice (optional, for .doc/.ppt/.xls processing)."""
from __future__ import annotations

import platform
import shutil

from rich.console import Console


class LibreOfficeStep:
    name = "LibreOffice (optional)"

    def _find_binary(self) -> bool:
        return shutil.which("libreoffice") is not None or shutil.which("soffice") is not None

    def check(self) -> bool:
        return self._find_binary()

    def install(self, console: Console) -> bool:
        console.print("  [yellow]LibreOffice is optional.[/]")
        console.print("  Only needed for .doc, .ppt, .xls file processing.")
        console.print()

        system = platform.system()
        if system == "Linux":
            console.print("  Install on Ubuntu/Debian:")
            console.print("    [bold]sudo apt install libreoffice-core[/]")
        elif system == "Darwin":
            console.print("  Install on macOS:")
            console.print("    [bold]brew install --cask libreoffice[/]")
        else:
            console.print(f"  Download from: https://www.libreoffice.org/download/")

        console.print()
        console.print("  [dim]Re-run this step after installing.[/]")
        return False

    def verify(self) -> bool:
        return self.check()
