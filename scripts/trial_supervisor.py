"""Supervisore termico di processo per la prova dei modelli (F-775, decisione A dell'operatore, 20/11).

Un solo supervisore, padre della prova: legge i sensori PRIMA di avviare il figlio, poi lo tiene d'occhio mentre
gira in un gruppo di processi proprio. Soglie dai sensori (coretemp temp1_max, `nvidia-smi -q` slowdown), mai a mano;
sensore illeggibile, assente o fuori dominio = non si parte (fail-closed) e, in corsa, arresto. Stesso schema di
dell_guard (temp1_max del coretemp, ignoto = blocco); il modulo di dell_guard sta in un altro repository e non e'
importabile da qui, per cui la lettura del coretemp ne ripete il criterio.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

HWMON = Path("/sys/class/hwmon")
EXIT_THERMAL = 75  # arresto termico o sensore perso: figlio terminato, modelli scaricati (EX_TEMPFAIL)
EXIT_PREFLIGHT = 78  # soglie illeggibili o gia' superate: il figlio non e' mai partito (EX_CONFIG)
# LIMITE di sicurezza: dopo SIGTERM il figlio ha 5 s per uscire; oltre, SIGKILL al gruppo. Non e' un parametro di
# misura: limita quanto a lungo la GPU puo' restare caricata dopo che i sensori hanno dato l'allarme.
TERM_GRACE_S = 5
# LIMITE di risorsa: nvidia-smi che non risponde entro 10 s e' telemetria persa (fail-closed), non un'attesa da ripetere.
SENSOR_READ_TIMEOUT_S = 10
# LIMITE di risorsa: /api/ps e lo scarico del modello rispondono subito o la risposta non serve piu' (si esce comunque).
UNLOAD_TIMEOUT_S = 30
# LIMITE di reattivita': verifica che il gruppo sia sparito 20 volte al secondo, costo nullo.
GROUP_POLL_S = 0.05


class SensorError(Exception):
    """Sensore assente, illeggibile o fuori dominio."""


@dataclass
class Outcome:
    code: int
    tripped: dict | None  # {"reason": "thermal_limit"|"sensor_lost", ...} se il supervisore ha fermato il figlio
    peaks: dict
    how: str = ""  # "SIGTERM" o "SIGKILL": come si e' chiuso il gruppo


def log(event: str, **fields) -> None:
    print(json.dumps({"t": round(time.time(), 3), "event": event, **fields}, sort_keys=True), file=sys.stderr, flush=True)


def _positive(value: float, what: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise SensorError(f"{what} fuori dominio: {value}")
    return value


def _nvidia_smi(*args: str) -> str:
    """Stdout di nvidia-smi, o SensorError se non parte, scade o esce con errore (un exit != 0 invalida anche l'output parsabile)."""
    try:
        done = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=False, timeout=SENSOR_READ_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:  # ValueError: output non decodificabile
        raise SensorError(f"nvidia-smi {args[0]}: {exc}") from exc
    if done.returncode != 0:
        raise SensorError(f"nvidia-smi {args[0]}: exit {done.returncode}")
    return done.stdout


def gpu_slowdown_limit() -> float:
    m = re.search(r"^\s*GPU Slowdown Temp\s*:\s*(\d+) C\s*$", _nvidia_smi("-q", "-d", "TEMPERATURE"), re.MULTILINE)
    if not m:
        raise SensorError("soglia GPU Slowdown Temp non dichiarata da nvidia-smi")
    return _positive(float(m.group(1)), "GPU Slowdown Temp")


def cpu_limit() -> tuple[Path, float]:
    """(file temp1_input, soglia in gradi): temp1_max del coretemp, come dell_guard. Un solo pacchetto: con piu' coretemp
    se ne leggerebbe uno e gli altri potrebbero superare la propria soglia senza che nessuno se ne accorga (fail-closed)."""
    try:
        found = [hw for hw in sorted(HWMON.glob("hwmon*")) if (hw / "name").read_text().strip() == "coretemp"]
        if len(found) > 1:
            raise SensorError(f"{len(found)} pacchetti coretemp: se ne legge uno solo, prova non avviata")
        if found:
            return found[0] / "temp1_input", _positive(int((found[0] / "temp1_max").read_text()) / 1000, "coretemp temp1_max")
    except (OSError, ValueError, OverflowError) as exc:
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
            out = _nvidia_smi("--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits").strip()
            fields = re.fullmatch(r"(\d+), (\d+)", out)  # nounits: due interi decimali; underscore, segni o esponenti sono output corrotto
            if not fields:
                raise ValueError(f"output di nvidia-smi inatteso: {out[:80]!r}")
            mem, gpu = int(fields[1]), int(fields[2])
            cpu = int(self.cpu_input.read_text()) / 1000
            if not all(math.isfinite(v) and v >= 0 for v in (mem, gpu, cpu)):
                raise ValueError(f"lettura fuori dominio: vram={mem} gpu={gpu} cpu={cpu}")
        except (OSError, ValueError, OverflowError) as exc:
            raise SensorError(str(exc)) from exc
        self.peaks = {
            "vram_mib": max(self.peaks["vram_mib"], int(mem)),
            "gpu_c": max(self.peaks["gpu_c"], gpu),
            "cpu_c": max(self.peaks["cpu_c"], cpu),
        }
        return {"gpu": gpu, "cpu": cpu, "vram_mib": int(mem)}


def _hot(sensors, readings: dict) -> bool:
    return readings["gpu"] >= sensors.limits["gpu"] or readings["cpu"] >= sensors.limits["cpu"]


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def terminate_group(proc: subprocess.Popen) -> str:
    """SIGTERM al gruppo del figlio, SIGKILL se dopo TERM_GRACE_S resta qualcuno. Vale anche per i discendenti."""
    pgid = proc.pid  # start_new_session: il figlio e' capogruppo
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return "SIGTERM"
    deadline = time.monotonic() + TERM_GRACE_S
    while time.monotonic() < deadline:
        proc.poll()  # raccoglie il figlio: uno zombie terrebbe vivo il gruppo
        if not _group_alive(pgid):
            return "SIGTERM"
        time.sleep(GROUP_POLL_S)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    return "SIGKILL"


def _ollama(host: str, method: str, path: str, body: dict | None = None) -> tuple[int, bytes]:
    url = urlparse(host)
    conn = http.client.HTTPConnection(url.hostname or "127.0.0.1", url.port or 80, timeout=UNLOAD_TIMEOUT_S)
    try:
        conn.request(method, path, json.dumps(body) if body else None, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def unload_models(host: str) -> None:
    """Scarica ogni modello residente (`/api/ps` + keep_alive=0): l'equivalente di `ollama stop` senza conoscere il nome."""
    try:
        models = [m.get("name") for m in json.loads(_ollama(host, "GET", "/api/ps")[1]).get("models", [])]
    except (OSError, ValueError, AttributeError, http.client.HTTPException) as exc:
        log("thermal_model_stop_failed", error=f"/api/ps: {exc}")
        return
    for name in models:
        try:
            status = _ollama(host, "POST", "/api/generate", {"model": name, "keep_alive": 0})[0]
            log("thermal_model_stop" if status == 200 else "thermal_model_stop_failed", model=name, status=status)
        except (OSError, http.client.HTTPException) as exc:
            log("thermal_model_stop_failed", model=name, error=str(exc))


def supervise(argv: list[str], sensors, unload, env: dict | None = None) -> Outcome:
    """Esegue `argv` sotto controllo termico. Il figlio parte solo dopo una lettura valida e sotto soglia."""
    try:
        first = sensors.read()
    except SensorError as exc:
        log("preflight_failed", error=str(exc))
        return Outcome(EXIT_PREFLIGHT, None, sensors.peaks)
    if _hot(sensors, first):
        log("preflight_hot", readings=first, limits=sensors.limits)
        return Outcome(EXIT_PREFLIGHT, None, sensors.peaks)
    proc = subprocess.Popen(argv, env=env, start_new_session=True)
    log("child_started", pid=proc.pid, limits=sensors.limits, readings=first)
    tripped, how = None, ""
    try:
        while proc.poll() is None:
            try:
                readings = sensors.read()
            except SensorError as exc:
                tripped = {"reason": "sensor_lost", "error": str(exc)}
                break
            if _hot(sensors, readings):
                tripped = {"reason": "thermal_limit", "readings": readings}
                break
    finally:  # anche su KeyboardInterrupt: nessun gruppo orfano in giro
        if tripped or _group_alive(proc.pid):
            how = terminate_group(proc)
    if tripped:
        log("thermal_stop", **tripped, limits=sensors.limits, how=how)
        unload()
        return Outcome(EXIT_THERMAL, tripped, sensors.peaks, how)
    return Outcome(proc.returncode, None, sensors.peaks, how)
