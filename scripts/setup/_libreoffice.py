"""Setup step: LibreOffice (optional, for .doc/.ppt/.xls processing)."""
from __future__ import annotations

import platform
import shutil
from pathlib import Path

from rich.console import Console


class LibreOfficeStep:
    name = "LibreOffice (optional)"
    description = "Optional Office file conversion support (.doc/.ppt/.xls)"

    def _find_binary(self) -> bool:
        if shutil.which("libreoffice") is not None or shutil.which("soffice") is not None:
            return True
        if platform.system() == "Darwin":
            app_soffice = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
            user_soffice = Path.home() / "Applications" / "LibreOffice.app" / "Contents" / "MacOS" / "soffice"
            return app_soffice.exists() or user_soffice.exists()
        return False

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
            if shutil.which("port"):
                console.print("    [bold]sudo port install libreoffice[/]  [dim](recommended on macOS 12)[/]")
            if shutil.which("brew"):
                console.print("    [bold]brew install --cask libreoffice[/]")
            if not shutil.which("port") and not shutil.which("brew"):
                console.print("    [bold]Download installer:[/] https://www.libreoffice.org/download/")
        else:
            console.print(f"  Download from: https://www.libreoffice.org/download/")

        console.print()
        console.print("  [dim]Re-run this step after installing.[/]")
        return False

    def verify(self) -> bool:
        return self.check()
