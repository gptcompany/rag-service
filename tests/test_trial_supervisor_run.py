"""Supervisore termico di processo (F-775): gruppo di processi, arresto a soglia o sensore perso, scarico dei modelli.

Il figlio e' un processo vero in un gruppo proprio; solo i sensori sono finti (nessun test deve scaldare l'host).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from scripts import trial_supervisor as ts

SPAWNER = (
    "import os, subprocess, sys, time\n"
    "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "open(sys.argv[1], 'w').write(f'{os.getpid()} {g.pid}')\n"
    "time.sleep(60)\n"
)
IGNORES_TERM = "import signal, sys, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nopen(sys.argv[1], 'w').write('up')\ntime.sleep(60)\n"
HOT = {"gpu": 95.0, "cpu": 50.0, "vram_mib": 1}


class FakeSensors:
    """read() n-esimo -> step(n): un dict di letture o un'eccezione; limiti dichiarati come li darebbero i sensori."""

    def __init__(self, step):
        self.step, self.n, self.peaks, self.events = step, 0, {}, []
        self.limits = {"gpu": 90.0, "cpu": 100.0}

    def read(self):
        self.n += 1
        self.events.append("read")
        time.sleep(0.01)
        return self.step(self.n)


def cool(n=0):
    return {"gpu": 50.0, "cpu": 50.0, "vram_mib": 100}


def gone(pid, wait=3.0):
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return True
        except FileNotFoundError:
            return True
        time.sleep(0.02)
    return False


def test_il_supervisore_legge_i_sensori_prima_di_avviare_il_figlio(monkeypatch):
    sensors, order = FakeSensors(cool), []
    real = ts.subprocess.Popen
    monkeypatch.setattr(ts.subprocess, "Popen", lambda *a, **k: (order.append(list(sensors.events)), real(*a, **k))[1])
    out = ts.supervise([sys.executable, "-c", "pass"], sensors, lambda: None)
    assert order == [["read"]] and out.code == 0 and out.tripped is None


def test_sensore_illeggibile_prima_della_partenza_non_avvia_il_figlio(monkeypatch):
    def unreadable(n):
        raise ts.SensorError("nvidia-smi N/A")

    monkeypatch.setattr(ts.subprocess, "Popen", lambda *a, **k: pytest.fail("figlio avviato senza sensori"))
    assert ts.supervise(["x"], FakeSensors(unreadable), lambda: None).code == ts.EXIT_PREFLIGHT


def test_soglia_gia_superata_non_avvia_il_figlio(monkeypatch):
    monkeypatch.setattr(ts.subprocess, "Popen", lambda *a, **k: pytest.fail("figlio avviato a soglia superata"))
    assert ts.supervise(["x"], FakeSensors(lambda n: HOT), lambda: None).code == ts.EXIT_PREFLIGHT


def test_superamento_termina_figlio_e_discendenti_e_scarica_i_modelli(tmp_path, capsys):
    pids, unloaded = tmp_path / "pids", []
    out = ts.supervise([sys.executable, "-c", SPAWNER, str(pids)], FakeSensors(lambda n: HOT if pids.exists() else cool()),
                       lambda: unloaded.append(1))
    child, grandchild = map(int, pids.read_text().split())
    assert out.code == ts.EXIT_THERMAL and out.tripped["reason"] == "thermal_limit" and unloaded == [1]
    assert gone(child) and gone(grandchild), "il gruppo di processi doveva morire per intero"
    err = capsys.readouterr().err
    assert '"event": "thermal_stop"' in err and '"gpu": 95.0' in err  # l'evento porta le letture


def test_sensore_che_sparisce_a_meta_prova_ferma_la_prova(tmp_path, capsys):
    pids, unloaded = tmp_path / "pids", []

    def step(n):
        if pids.exists():
            raise ts.SensorError("temp1_input sparito")
        return cool()

    out = ts.supervise([sys.executable, "-c", SPAWNER, str(pids)], FakeSensors(step), lambda: unloaded.append(1))
    child, grandchild = map(int, pids.read_text().split())
    assert out.code == ts.EXIT_THERMAL and out.tripped["reason"] == "sensor_lost" and unloaded == [1]
    assert gone(child) and gone(grandchild)
    assert "temp1_input sparito" in capsys.readouterr().err


def test_sigkill_oltre_la_scadenza_se_il_figlio_ignora_sigterm(tmp_path, monkeypatch):
    up = tmp_path / "up"
    monkeypatch.setattr(ts, "TERM_GRACE_S", 0.3)
    out = ts.supervise([sys.executable, "-c", IGNORES_TERM, str(up)], FakeSensors(lambda n: HOT if up.exists() else cool()), lambda: None)
    assert out.code == ts.EXIT_THERMAL and out.how == "SIGKILL"


def test_uscita_normale_propaga_il_codice_e_non_scarica_nulla():
    unloaded = []
    out = ts.supervise([sys.executable, "-c", "raise SystemExit(3)"], FakeSensors(cool), lambda: unloaded.append(1))
    assert out.code == 3 and out.tripped is None and unloaded == []


def test_interruzione_dell_operatore_non_lascia_il_gruppo_in_vita(tmp_path):
    pids = tmp_path / "pids"

    def step(n):
        if pids.exists():
            raise KeyboardInterrupt
        return cool()

    with pytest.raises(KeyboardInterrupt):
        ts.supervise([sys.executable, "-c", SPAWNER, str(pids)], FakeSensors(step), lambda: None)
    assert all(gone(int(p)) for p in pids.read_text().split())


def test_scarico_di_ogni_modello_residente_e_un_errore_non_ferma_gli_altri(capsys):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self._send(200, {"models": [{"name": "a"}, {"name": "b"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            self._send(500 if body["model"] == "a" else 200, {})

        def _send(self, status, obj):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ts.unload_models(f"http://127.0.0.1:{srv.server_port}")
    finally:
        srv.shutdown()
        srv.server_close()
    assert calls == [{"model": "a", "keep_alive": 0}, {"model": "b", "keep_alive": 0}]
    err = capsys.readouterr().err
    assert err.count("thermal_model_stop_failed") == 1 and err.count('"event": "thermal_model_stop"') == 1


def test_ollama_irraggiungibile_allo_scarico_si_registra_e_non_solleva(capsys):
    ts.unload_models("http://127.0.0.1:9")
    assert "thermal_model_stop_failed" in capsys.readouterr().err
