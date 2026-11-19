"""Supervisore termico (F-775), difetti della review: SIGTERM al supervisore, log che non si scrive, zombie, scarico con tetto."""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts import trial_supervisor as ts
from tests.test_trial_supervisor_run import HOT, SPAWNER, FakeSensors, cool, gone


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_sigterm_al_supervisore_ferma_il_gruppo(tmp_path):
    """Lo SIGTERM arriva al pid del supervisore (scheduler, operatore): il figlio, in una sessione propria, non lo riceverebbe."""
    pids, seen = tmp_path / "pids", []
    signal.signal(signal.SIGTERM, lambda *_: seen.append(1))  # rete di sicurezza: senza gestore del supervisore non muore pytest

    def step(n):
        if pids.exists():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.5)  # con il gestore del supervisore il segnale interrompe l'attesa con KeyboardInterrupt
            if seen:
                raise ts.SensorError("lo SIGTERM non ha fermato il supervisore")
        return cool()

    before = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(KeyboardInterrupt):
            ts.supervise([sys.executable, "-c", SPAWNER, str(pids)], FakeSensors(step), lambda: None)
        assert all(gone(int(p)) for p in pids.read_text().split()) and signal.getsignal(signal.SIGTERM) == before
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_un_log_che_non_si_scrive_non_salta_arresto_e_scarico(tmp_path, monkeypatch):
    class Broken:
        def write(self, text):
            raise OSError("stderr chiuso")

        def flush(self):
            raise OSError("stderr chiuso")

    pids, unloaded = tmp_path / "pids", []
    monkeypatch.setattr(sys, "stderr", Broken())
    sensors = FakeSensors(lambda n: HOT if pids.exists() else cool())
    out = ts.supervise([sys.executable, "-c", SPAWNER, str(pids)], sensors, lambda: unloaded.append(1))
    assert out.code == ts.EXIT_THERMAL and unloaded == [1] and all(gone(int(p)) for p in pids.read_text().split())


def test_uno_zombie_non_tiene_vivo_il_gruppo(tmp_path, monkeypatch):
    for pid, state, pgrp in [(100, "Z", 77), (101, "S", 88), (102, "S", 77)]:
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_text(f"{pid} (nome con ) e spazi) {state} 1 {pgrp} {pgrp} 0 -1 4194560\n")
    monkeypatch.setattr(ts, "PROC", tmp_path)
    assert ts._group_alive(77) is True  # 102 e' vivo
    (tmp_path / "102" / "stat").write_text("102 (x) Z 1 77 77 0 -1 0\n")
    assert ts._group_alive(77) is False  # restano solo uno zombie e un processo di un altro gruppo


def test_scarico_con_corpo_che_non_finisce_ha_un_tetto_di_tempo(monkeypatch, capsys):
    class Slow(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            for _ in range(200):  # un byte ogni 50 ms: ogni singola lettura risponde, il corpo no
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(0.05)

    srv = _serve(Slow)
    monkeypatch.setattr(ts, "UNLOAD_TIMEOUT_S", 0.5)
    t0 = time.monotonic()
    try:
        ts.unload_models(f"http://127.0.0.1:{srv.server_port}")
    finally:
        srv.shutdown()
        srv.server_close()
    assert time.monotonic() - t0 < 3 and "thermal_model_stop_failed" in capsys.readouterr().err


@pytest.mark.parametrize("status, body", [(503, {"error": "busy"}), (200, {"models": None}), (200, {"models": "x"}), (200, [1])])
def test_elenco_modelli_non_valido_si_registra_e_non_solleva(status, body, capsys):
    class H(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = _serve(H)
    try:
        ts.unload_models(f"http://127.0.0.1:{srv.server_port}")
    finally:
        srv.shutdown()
        srv.server_close()
    assert "thermal_model_stop_failed" in capsys.readouterr().err
