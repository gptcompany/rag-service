"""Sensori termici per la prova dei modelli (F-775, decisione A dell'operatore, 20/11).

Soglie dai sensori (coretemp temp1_max, `nvidia-smi -q` slowdown), mai a mano; sensore illeggibile, assente o fuori
dominio = SensorError (fail-closed). Stesso schema di dell_guard (temp1_max del coretemp, ignoto = blocco): il modulo
di dell_guard sta in un altro repository e non e' importabile da qui, per cui la lettura del coretemp ne ripete il criterio.
"""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

HWMON = Path("/sys/class/hwmon")
# LIMITE di risorsa: nvidia-smi che non risponde entro 10 s e' telemetria persa (fail-closed), non un'attesa da ripetere.
SENSOR_READ_TIMEOUT_S = 10


class SensorError(Exception):
    """Sensore assente, illeggibile o fuori dominio."""


def _positive(value: float, what: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise SensorError(f"{what} fuori dominio: {value}")
    return value


def gpu_slowdown_limit() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-q", "-d", "TEMPERATURE"], capture_output=True, text=True, check=False, timeout=SENSOR_READ_TIMEOUT_S
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SensorError(f"nvidia-smi -q: {exc}") from exc
    m = re.search(r"GPU Slowdown Temp\s*:\s*(\d+)", out)
    if not m:
        raise SensorError("soglia GPU Slowdown Temp non dichiarata da nvidia-smi")
    return _positive(float(m.group(1)), "GPU Slowdown Temp")


def cpu_limit() -> tuple[Path, float]:
    """(file temp1_input, soglia in gradi): temp1_max del coretemp, come dell_guard."""
    try:
        for hw in sorted(HWMON.glob("hwmon*")):
            if (hw / "name").read_text().strip() == "coretemp":
                return hw / "temp1_input", _positive(int((hw / "temp1_max").read_text()) / 1000, "coretemp temp1_max")
    except (OSError, ValueError) as exc:
        raise SensorError(f"coretemp temp1_max: {exc}") from exc
    raise SensorError("coretemp assente")


class Sensors:
    """Soglie lette una volta dai sensori; `read()` da' le temperature correnti o alza SensorError."""

    def __init__(self) -> None:
        self.cpu_input, cpu = cpu_limit()
        self.limits = {"gpu": gpu_slowdown_limit(), "cpu": cpu}
        self.peaks = {"vram_mib": 0, "gpu_c": 0.0, "cpu_c": 0.0}

    def read(self) -> dict:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
                timeout=SENSOR_READ_TIMEOUT_S,
            ).stdout.strip()
            mem, gpu = (float(x) for x in out.split(","))
            cpu = int(self.cpu_input.read_text()) / 1000
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise SensorError(str(exc)) from exc
        if not all(math.isfinite(v) and v >= 0 for v in (mem, gpu, cpu)):
            raise SensorError(f"lettura fuori dominio: vram={mem} gpu={gpu} cpu={cpu}")
        self.peaks = {
            "vram_mib": max(self.peaks["vram_mib"], int(mem)),
            "gpu_c": max(self.peaks["gpu_c"], gpu),
            "cpu_c": max(self.peaks["cpu_c"], cpu),
        }
        return {"gpu": gpu, "cpu": cpu, "vram_mib": int(mem)}
