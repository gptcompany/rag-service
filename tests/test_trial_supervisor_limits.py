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


# ---- secondo giro della review interna ------------------------------------------------------------------------

def _raising_handler(*_):
    raise KeyboardInterrupt


def test_un_annullamento_subito_dopo_la_nascita_del_figlio_non_lo_lascia_orfano(tmp_path, monkeypatch):
    """Il segnale arriva dopo Popen e prima che il supervisore registri la pulizia: il figlio, in un'altra sessione, non cadrebbe."""
    pids, real = tmp_path / "pids", ts.subprocess.Popen
    made = []

    def popen_then_signal(*args, **kwargs):
        proc = real(*args, **kwargs)
        made.append(proc.pid)
        os.kill(os.getpid(), signal.SIGTERM)
        return proc

    monkeypatch.setattr(ts.subprocess, "Popen", popen_then_signal)
    signal.signal(signal.SIGTERM, _raising_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            ts.supervise([sys.executable, "-c", SPAWNER, str(pids)], FakeSensors(lambda n: cool()), lambda: None)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    assert gone(made[0])


def test_uno_sigterm_durante_lo_scarico_non_lo_interrompe(tmp_path):
    pids, done = tmp_path / "pids", []

    def unload():
        os.kill(os.getpid(), signal.SIGTERM)  # lo scheduler insiste mentre i modelli si stanno scaricando
        time.sleep(0.2)
        done.append(1)

    signal.signal(signal.SIGTERM, _raising_handler)
    try:
        out = ts.supervise([sys.executable, "-c", SPAWNER, str(pids)], FakeSensors(lambda n: HOT if pids.exists() else cool()), unload)
    except KeyboardInterrupt:
        pytest.fail("lo SIGTERM ha interrotto lo scarico dei modelli")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    assert out.code == ts.EXIT_THERMAL and done == [1]


def test_il_leader_che_non_muore_dopo_sigkill_non_blocca_il_supervisore(monkeypatch):
    class Stuck:
        pid = 4242_4242

        def poll(self):
            return None

        def wait(self, timeout=None):
            assert timeout is not None, "attesa illimitata dopo SIGKILL"
            raise ts.subprocess.TimeoutExpired("x", timeout)

    monkeypatch.setattr(ts.os, "killpg", lambda *a: None)
    monkeypatch.setattr(ts, "_group_alive", lambda pgid: True)
    monkeypatch.setattr(ts, "TERM_GRACE_S", 0.1)
    assert ts.terminate_group(Stuck()) == "SIGKILL_UNCONFIRMED"


def test_uno_scarico_che_non_risponde_non_affama_gli_altri_modelli(monkeypatch):
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            data = json.dumps({"models": [{"name": "a"}, {"name": "b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body["model"])
            if body["model"] == "a":
                time.sleep(3)  # il primo non risponde: il secondo va tentato lo stesso
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    srv = _serve(H)
    monkeypatch.setattr(ts, "UNLOAD_TIMEOUT_S", 1.0)
    try:
        ts.unload_models(f"http://127.0.0.1:{srv.server_port}")
    finally:
        srv.shutdown()
        srv.server_close()
    assert sorted(calls) == ["a", "b"]


def test_un_leader_zombie_con_thread_vivi_tiene_vivo_il_gruppo(tmp_path, monkeypatch):
    """Su Linux il leader puo' essere zombie mentre altri suoi thread girano ancora: il gruppo non e' morto."""
    (tmp_path / "100" / "task" / "100").mkdir(parents=True)
    (tmp_path / "100" / "task" / "101").mkdir()
    (tmp_path / "100" / "stat").write_text("100 (py) Z 1 77 77 0 -1 0\n")
    (tmp_path / "100" / "task" / "100" / "stat").write_text("100 (py) Z 1 77 77 0 -1 0\n")
    (tmp_path / "100" / "task" / "101" / "stat").write_text("101 (py) S 1 77 77 0 -1 0\n")
    monkeypatch.setattr(ts, "PROC", tmp_path)
    assert ts._group_alive(77) is True
    (tmp_path / "100" / "task" / "101" / "stat").write_text("101 (py) Z 1 77 77 0 -1 0\n")
    assert ts._group_alive(77) is False


def test_uno_stderr_bloccato_non_ferma_arresto_e_scarico(tmp_path, monkeypatch):
    """Pipe piena e lettore fermo: print() non solleva, aspetta. Il registro e' diagnostica e non puo' trattenere la sicurezza."""
    release = threading.Event()

    class Stalled:
        def write(self, text):
            release.wait(30)

        def flush(self):
            pass

    pids, unloaded = tmp_path / "pids", []
    monkeypatch.setattr(sys, "stderr", Stalled())
    monkeypatch.setattr(ts, "LOG_WAIT_S", 0.2)
    t0 = time.monotonic()
    try:
        out = ts.supervise([sys.executable, "-c", SPAWNER, str(pids)], FakeSensors(lambda n: HOT if pids.exists() else cool()), lambda: unloaded.append(1))
    finally:
        release.set()
    assert out.code == ts.EXIT_THERMAL and unloaded == [1] and time.monotonic() - t0 < 10
    assert all(gone(int(p)) for p in pids.read_text().split())
