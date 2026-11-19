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
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

HWMON = Path("/sys/class/hwmon")
PROC = Path("/proc")
EXIT_THERMAL = 75  # arresto termico o sensore perso: figlio terminato, modelli scaricati (EX_TEMPFAIL)
EXIT_PREFLIGHT = 78  # soglie illeggibili o gia' superate: il figlio non e' mai partito (EX_CONFIG)
EXIT_SHUTDOWN_FAILED = 70  # arresto termico NON verificato: container ancora attivo, modello ancora caricato o controllo illeggibile (EX_SOFTWARE)
COMPOSE_FILE = Path(__file__).resolve().parent.parent / "docker-compose.yml"
RAG_SERVICE = "rag"  # il servizio del compose che, con una query di retrieve in volo, continua a chiamare ollama
# LIMITE di sicurezza: dopo SIGTERM il figlio ha 5 s per uscire; oltre, SIGKILL al gruppo. Non e' un parametro di
# misura: limita quanto a lungo la GPU puo' restare caricata dopo che i sensori hanno dato l'allarme.
TERM_GRACE_S = 5
# LIMITE di risorsa: nvidia-smi che non risponde entro 10 s e' telemetria persa (fail-closed), non un'attesa da ripetere.
SENSOR_READ_TIMEOUT_S = 10
# LIMITE di risorsa: l'intero scarico dei modelli (/api/ps e un POST per modello) finisce entro 30 s o si esce comunque.
UNLOAD_TIMEOUT_S = 30
# LIMITE di risorsa: `docker compose stop` = SIGTERM, attesa PREDEFINITA di Docker (10 s, o stop_grace_period del compose: nessun -t
# qui) e SIGKILL. 60 s coprono il caso normale con margine; oltre e' un daemon appeso, e l'arresto e' un errore (fail-closed).
DOCKER_TIMEOUT_S = 60
# LIMITE di risorsa: ollama completa lo scarico in modo asincrono, per cui la verifica "nessun modello caricato" ripete /api/ps
# fino a 10 s; un modello ancora residente dopo di allora e' un arresto non riuscito, non un'attesa da ripetere.
VERIFY_TIMEOUT_S = 10
# LIMITE di reattivita': verifica che il gruppo sia sparito 20 volte al secondo, costo nullo.
GROUP_POLL_S = 0.05
# LIMITE di reattivita': il registro e' diagnostica, non sicurezza; se stderr non accetta la riga entro 1 s si prosegue senza.
LOG_WAIT_S = 1.0


class SensorError(Exception):
    """Sensore assente, illeggibile o fuori dominio."""


@dataclass
class Outcome:
    code: int
    tripped: dict | None  # {"reason": "thermal_limit"|"sensor_lost", ...} se il supervisore ha fermato il figlio
    peaks: dict
    how: str = ""  # "SIGTERM" o "SIGKILL": come si e' chiuso il gruppo
    shutdown: dict | None = None  # esito di thermal_shutdown (container fermo, modelli scaricati, tempi) se c'e' stato un arresto


def log(event: str, **fields) -> None:
    """Una riga JSON su stderr, ma mai a spese della sicurezza: la scrittura va in un thread e si attende al piu' LOG_WAIT_S
    (pipe piena e lettore fermo bloccano print() senza sollevare), un errore di scrittura non salta arresto e scarico."""
    line = json.dumps({"t": round(time.time(), 3), "event": event, **fields}, sort_keys=True)
    done = threading.Event()

    def write() -> None:
        try:
            print(line, file=sys.stderr, flush=True)
        except OSError:
            pass  # stderr chiuso o disco pieno
        finally:
            done.set()

    threading.Thread(target=write, daemon=True).start()
    done.wait(LOG_WAIT_S)


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
    """Un membro vivo del gruppo (gli zombie non contano: sono morti, aspettano solo di essere raccolti). Linux: /proc/<pid>/stat."""
    for stat in PROC.glob("[0-9]*/stat"):
        try:
            state, _ppid, group = stat.read_text().rsplit(")", 1)[1].split()[:3]  # dopo ")" perche' il nome puo' contenere spazi
        except (OSError, ValueError):
            continue  # il processo e' sparito mentre lo si leggeva
        if int(group) != pgid:
            continue
        if state != "Z":
            return True
        for task in (stat.parent / "task").glob("[0-9]*/stat"):  # leader zombie: i suoi thread possono girare ancora
            try:
                if task.read_text().rsplit(")", 1)[1].split()[0] != "Z":
                    return True
            except (OSError, IndexError):
                continue
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
    try:
        proc.wait(TERM_GRACE_S)  # un leader in I/O non interrompibile non deve impedire lo scarico dei modelli
    except subprocess.TimeoutExpired:
        log("child_exit_unconfirmed", pid=proc.pid)
        return "SIGKILL_UNCONFIRMED"
    return "SIGKILL"


def _ollama(host: str, method: str, path: str, body: dict | None = None) -> tuple[int, bytes]:
    url = urlparse(host)
    conn = http.client.HTTPConnection(url.hostname or "127.0.0.1", url.port or 80, timeout=UNLOAD_TIMEOUT_S)
    try:
        conn.request(method, path, json.dumps(body) if body else None, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, resp.read() if method == "GET" else b""  # dello scarico basta lo stato
    finally:
        conn.close()


def _loaded(host: str, strict: bool = True) -> list[str]:
    """Nomi dei modelli residenti (`/api/ps`); ValueError se l'elenco non e' valido. `strict`: anche una voce senza nome e' un errore
    (la verifica non puo' dire "nessun modello"); lo scarico invece ferma tutto cio' che sa identificare."""
    status, raw = _ollama(host, "GET", "/api/ps")
    listing = json.loads(raw)
    if status != 200 or not isinstance(listing, dict) or not isinstance(listing.get("models"), list):
        raise ValueError(f"HTTP {status}, elenco non valido: {raw[:120]!r}")
    names = [m["name"] for m in listing["models"] if isinstance(m, dict) and isinstance(m.get("name"), str)]
    if strict and len(names) != len(listing["models"]):  # una voce non riconosciuta e' un modello che non si sa scaricare: mai "nessun modello"
        raise ValueError(f"voce senza nome nell'elenco: {raw[:120]!r}")
    return names


def _unload(host: str) -> None:
    try:
        models = _loaded(host, strict=False)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        log("thermal_model_stop_failed", error=f"/api/ps: {exc}")
        return
    def stop(name: str) -> None:
        try:
            status = _ollama(host, "POST", "/api/generate", {"model": name, "keep_alive": 0})[0]
            log("thermal_model_stop" if status == 200 else "thermal_model_stop_failed", model=name, status=status)
        except (OSError, http.client.HTTPException) as exc:
            log("thermal_model_stop_failed", model=name, error=str(exc))

    # Un thread per modello: uno scarico che non risponde non consuma il tempo degli altri.
    threads = [threading.Thread(target=stop, args=(name,), daemon=True) for name in models]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def unload_models(host: str) -> None:
    """Scarica ogni modello residente (`/api/ps` + keep_alive=0): l'equivalente di `ollama stop` senza conoscere il nome.
    Tetto di tempo sull'insieme (un ollama che gocciola un corpo senza fine non deve trattenere il supervisore)."""
    worker = threading.Thread(target=_unload, args=(host,), daemon=True)
    worker.start()
    worker.join(UNLOAD_TIMEOUT_S)
    if worker.is_alive():
        log("thermal_model_stop_failed", error=f"scarico non concluso entro {UNLOAD_TIMEOUT_S} s")


def _compose(compose_file: Path, *args: str) -> subprocess.CompletedProcess:
    """`docker compose` con i permessi Docker che l'utente ha gia': niente sudo, nessun privilegio nuovo."""
    return subprocess.run(["docker", "compose", "-f", str(compose_file), *args], capture_output=True, text=True, check=False, timeout=DOCKER_TIMEOUT_S)


def stop_rag_service(compose_file: Path = COMPOSE_FILE, service: str = RAG_SERVICE) -> bool:
    """Ferma il container: la query di retrieve ancora in volo muore con lui. Uno stop manuale non fa scattare `unless-stopped`."""
    try:
        done = _compose(compose_file, "stop", service)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log("rag_stop_failed", service=service, error=str(exc))
        return False
    if done.returncode != 0:
        log("rag_stop_failed", service=service, exit=done.returncode, stderr=done.stderr[-200:])
    return done.returncode == 0


def rag_service_running(compose_file: Path = COMPOSE_FILE, service: str = RAG_SERVICE) -> bool | None:
    """True se il container gira, False se e' fermo (o non esiste), None se non si riesce a saperlo."""
    try:
        done = _compose(compose_file, "ps", "--status", "running", "--quiet", service)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log("rag_check_failed", service=service, error=str(exc))
        return None
    if done.returncode != 0:
        log("rag_check_failed", service=service, exit=done.returncode, stderr=done.stderr[-200:])
        return None
    return bool(done.stdout.strip())


def _models_loaded(host: str) -> list[str] | None:
    """Modelli ancora residenti dopo lo scarico: ripete /api/ps finche' e' vuoto, al piu' VERIFY_TIMEOUT_S sull'insieme (un ollama che
    gocciola la risposta non deve trattenere il supervisore). Ultimo elenco letto, None se illeggibile o mai arrivato."""
    seen: list[list[str] | None] = [None]
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            try:
                seen[0] = _loaded(host)
            except (OSError, ValueError, http.client.HTTPException) as exc:
                seen[0] = None
                log("thermal_model_check_failed", error=str(exc))
                return
            if not seen[0]:
                return
            time.sleep(GROUP_POLL_S)

    worker = threading.Thread(target=poll, daemon=True)
    worker.start()
    worker.join(VERIFY_TIMEOUT_S)
    if worker.is_alive():
        last = seen[0]  # istantanea: una risposta tardiva non deve trasformare il timeout in "nessun modello"
        stop.set()
        log("thermal_model_check_failed", error=f"verifica non conclusa entro {VERIFY_TIMEOUT_S} s")
        return last or None
    return seen[0]


def thermal_shutdown(host: str, compose_file: Path = COMPOSE_FILE, service: str = RAG_SERVICE) -> dict:
    """Arresto dopo la morte del figlio: PRIMA il container rag-service (una query viva ricaricherebbe i modelli), POI lo scarico di
    ollama, infine i due controlli. `ok` solo se il container e' verificato fermo e nessun modello e' caricato: il resto e' un errore.
    Il servizio resta fermo (restart `unless-stopped` non scatta dopo uno stop manuale) fino a `restart_rag_service`."""
    t0 = time.monotonic()
    stopped = stop_rag_service(compose_file, service)
    t1 = time.monotonic()
    unload_models(host)  # anche se lo stop e' fallito: la GPU va protetta comunque
    t2 = time.monotonic()
    running = rag_service_running(compose_file, service)
    loaded = _models_loaded(host)
    container_stopped = False if not stopped else None if running is None else not running
    return {"ok": container_stopped is True and loaded == [], "stop_s": round(t1 - t0, 3), "unload_s": round(t2 - t1, 3),
            "container_stopped": container_stopped, "models_loaded": loaded}


def restart_rag_service(sensors, compose_file: Path = COMPOSE_FILE, service: str = RAG_SERVICE) -> bool:
    """Unico modo di rialzare il rag-service dopo un arresto termico: azione esplicita, solo con i sensori sotto le soglie derivate
    da loro (le stesse del supervisore), registrata. Sensore illeggibile o temperatura alta = rifiuto."""
    try:
        readings = sensors.read()
    except SensorError as exc:
        log("rag_restart_refused", error=str(exc))
        return False
    if _hot(sensors, readings):
        log("rag_restart_refused", readings=readings, limits=sensors.limits)
        return False
    try:
        done = _compose(compose_file, "start", service)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log("rag_restart_failed", error=str(exc))
        return False
    log("rag_restart" if done.returncode == 0 else "rag_restart_failed", readings=readings, limits=sensors.limits, exit=done.returncode)
    return done.returncode == 0


def supervise(argv: list[str], sensors, unload, env: dict | None = None) -> Outcome:
    """Esegue `argv` sotto controllo termico. Il figlio parte solo dopo una lettura valida e sotto soglia.
    Un SIGTERM o SIGINT al supervisore (scheduler, operatore) vale come Ctrl-C: il gruppo del figlio, che sta in una
    sessione propria e non lo riceverebbe, viene terminato e l'interruzione risale. All'arresto `unload()` ferma il rag-service, scarica i modelli e ne verifica l'esito (dict con `ok`). Un segnale che arriva prima che il
    figlio sia registrato per la pulizia resta in sospeso e scatta appena lo e'; durante pulizia e scarico e' ignorato."""
    try:
        first = sensors.read()
    except SensorError as exc:
        log("preflight_failed", error=str(exc))
        return Outcome(EXIT_PREFLIGHT, None, sensors.peaks)
    if _hot(sensors, first):
        log("preflight_hot", readings=first, limits=sensors.limits)
        return Outcome(EXIT_PREFLIGHT, None, sensors.peaks)
    armed, pending = [False], [False]

    def interrupt(signum, frame) -> None:
        if not armed[0]:
            pending[0] = True
            return
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    tripped, how = None, ""
    try:
        proc = subprocess.Popen(argv, env=env, start_new_session=True)
        try:
            armed[0] = True
            if pending[0]:
                raise KeyboardInterrupt
            log("child_started", pid=proc.pid, limits=sensors.limits, readings=first)
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
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)  # un secondo segnale non deve interrompere pulizia e scarico
            if tripped or _group_alive(proc.pid):
                how = terminate_group(proc)
        if tripped:
            log("thermal_stop", **tripped, limits=sensors.limits, how=how)
            shutdown = unload()
            ok = isinstance(shutdown, dict) and shutdown.get("ok") is True  # niente successo presunto: senza i controlli e' un errore
            log("thermal_shutdown" if ok else "thermal_shutdown_failed", reason=tripped["reason"], readings=tripped.get("readings"),
                limits=sensors.limits, decision="stopped" if ok else "failed", alarm=not ok, **(shutdown if isinstance(shutdown, dict) else {}))
            return Outcome(EXIT_THERMAL if ok else EXIT_SHUTDOWN_FAILED, tripped, sensors.peaks, how, shutdown)
        return Outcome(proc.returncode, None, sensors.peaks, how)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
