"""Prova modelli (F-775, decisione A): un solo supervisore termico di processo, padre della prova.

Il figlio e' un processo vero (`model_trial` rilanciato dal supervisore in un gruppo proprio) che parla con un ollama o un
rag-service finti su 127.0.0.1; solo i sensori sono finti. Nessun watcher per richiesta: l'arresto e' del supervisore.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import trial_supervisor as ts
from tests import test_model_trial_transport as transport
from tests.test_model_trial_run import _args, mt
from tests.test_model_trial_transport import (
    hang_before_headers,
    send_json,
    stream,
    write_scratch,
)

ollama = transport.ollama  # la fixture del server ollama finto, condivisa con i test di trasporto
REPO = Path(__file__).resolve().parent.parent
# Il figlio vero con lo scratch di test: SCRATCH_ROOT e' una costante di sicurezza (mai /tmp), qui la si sposta solo nel figlio.
PRELUDE = (
    "import os, sys, pathlib\n"
    f"sys.path.insert(0, {str(REPO)!r})\n"
    "import scripts.model_trial as mt\n"
    "mt.SCRATCH_ROOT = pathlib.Path(os.environ['TEST_SCRATCH'])\n"
    "mt.main(sys.argv[1:])\n"
)


class Sensors:
    """Sensori finti: `step(n)` -> letture o eccezione; `events` registra l'ordine rispetto al resto del sistema."""

    def __init__(self, step, events=None):
        self.step, self.n, self.events = step, 0, events if events is not None else []
        self.limits, self.peaks = {"gpu": 90.0, "cpu": 100.0}, {"vram_mib": 7, "gpu_c": 61.0, "cpu_c": 55.0}

    def read(self):
        self.n += 1
        self.events.append("sensor_read")
        time.sleep(0.01)
        return self.step(self.n)


COOL = {"gpu": 50.0, "cpu": 50.0, "vram_mib": 1}
HOT = {"gpu": 95.0, "cpu": 50.0, "vram_mib": 1}


@pytest.fixture
def supervised(tmp_path, monkeypatch):
    """Esegue main() come l'operatore: sensori finti nel supervisore, figlio vero con lo scratch di test."""
    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    monkeypatch.setenv("TEST_SCRATCH", str(tmp_path))
    monkeypatch.delenv("F775_SUPERVISOR_PID")
    monkeypatch.setattr(mt, "child_command", lambda raw: [sys.executable, "-c", PRELUDE, *raw])

    def go(sensors, *argv):
        monkeypatch.setattr(mt.trial_supervisor, "Sensors", (lambda: sensors) if hasattr(sensors, "read") else sensors)
        with pytest.raises(SystemExit) as stop:
            mt.main(["--scratch", str(tmp_path), *argv])
        return stop.value.code

    return go


def result(tmp_path):
    return json.loads((tmp_path / "result.json").read_text())


def test_thermal_before_headers_il_supervisore_ferma_il_figlio_in_attesa_delle_intestazioni(tmp_path, monkeypatch, ollama, supervised):
    """Il caso R1 del report: il POST non ha ancora le intestazioni quando i sensori alzano `hot`."""
    srv = ollama(post__api_chat=hang_before_headers)
    write_scratch(tmp_path, monkeypatch, 1)
    sensors = Sensors(lambda n: HOT if ("POST", "/api/chat") in srv.paths() else COOL)
    t0 = time.monotonic()
    code = supervised(sensors, "run", "--models", "a", "--ollama-host", srv.host, "--guard-cmd", "true")
    assert code == ts.EXIT_THERMAL and time.monotonic() - t0 < 10
    assert srv.dropped.wait(3), "il server non ha visto cadere il client: la generazione continuerebbe"
    assert ("POST", "/api/generate", {"model": "a", "keep_alive": 0}) in srv.calls
    res = result(tmp_path)
    assert res["status"] == "PARZIALE" and res["decision"]["winner"] is None and res["stop_reason"].startswith("arresto termico")
    assert res["telemetry"]["tripped"]["reason"] == "thermal_limit" and res["telemetry"]["peaks"]["gpu_c"] == 61.0


def test_retrieve_hybrid_parte_solo_dopo_il_supervisore_e_si_ferma_a_soglia(tmp_path, monkeypatch, ollama, supervised):
    """Il residuo di #19: il retrieve hybrid chiama ollama (embedding) prima che parta qualunque watcher."""
    events = []
    rag = ollama(post__query=lambda h, s: (events.append("query"), hang_before_headers(h, s)))
    ollama_srv = ollama()
    (tmp_path / "questions.jsonl").write_text(json.dumps({"id": "q1", "paper": "p", "answer": "x", "question": "?"}) + "\n")
    sensors = Sensors(lambda n: HOT if "query" in events else COOL, events)
    code = supervised(sensors, "retrieve", "--rag-url", rag.host, "--ollama-host", ollama_srv.host)
    assert code == ts.EXIT_THERMAL and events[0] == "sensor_read" and events.index("query") > events.index("sensor_read")
    assert rag.dropped.wait(3) and ("POST", "/api/generate", {"model": "a", "keep_alive": 0}) in ollama_srv.calls


def test_sensore_che_sparisce_a_meta_prova_ferma_figlio_e_modelli(tmp_path, monkeypatch, ollama, supervised):
    srv = ollama(post__api_chat=hang_before_headers)
    write_scratch(tmp_path, monkeypatch, 2)

    def step(n):
        if ("POST", "/api/chat") in srv.paths():
            raise ts.SensorError("temp1_input sparito")
        return COOL

    code = supervised(Sensors(step), "run", "--models", "a", "--ollama-host", srv.host, "--guard-cmd", "true")
    res = result(tmp_path)
    assert code == ts.EXIT_THERMAL and res["telemetry"]["tripped"]["reason"] == "sensor_lost"
    assert res["status"] == "PARZIALE" and res["decision"]["winner"] is None and ("POST", "/api/generate", {"model": "a", "keep_alive": 0}) in srv.calls


def test_round_parziali_dopo_arresto_termico_nessun_vincitore_e_i_round_completi_restano(tmp_path, monkeypatch, ollama, supervised):
    count = {"n": 0}

    def chat(h, s):  # primo round risposta valida, secondo round bloccato
        count["n"] += 1
        if count["n"] == 1:
            return stream([json.dumps({"message": {"content": json.dumps({"answer": "x", "paper": "p0"})}}), json.dumps({"done": True})])(h, s)
        hang_before_headers(h, s)

    srv = ollama(post__api_chat=chat, get__api_ps=lambda h, s: send_json(h, {"models": [{"name": "a", "size": 10, "size_vram": 10}]}))
    write_scratch(tmp_path, monkeypatch, 2)
    code = supervised(Sensors(lambda n: HOT if count["n"] >= 2 else COOL), "run", "--models", "a", "--ollama-host", srv.host, "--guard-cmd", "true")
    res = result(tmp_path)
    assert code == ts.EXIT_THERMAL and res["rounds_completed"] == 1 and res["decision"]["winner"] is None
    assert len((tmp_path / "rounds.jsonl").read_text().splitlines()) == 1


def test_senza_sensori_leggibili_il_figlio_non_parte_e_l_esito_vecchio_sparisce(tmp_path, monkeypatch, supervised):
    (tmp_path / "result.json").write_text('{"decision": {"winner": "vecchio"}}')
    write_scratch(tmp_path, monkeypatch, 1)
    monkeypatch.setattr(mt, "child_command", lambda raw: pytest.fail("figlio avviato senza sensori"))

    def broken():
        raise ts.SensorError("soglie non leggibili")

    code = supervised(broken, "run", "--models", "a", "--guard-cmd", "true")
    assert code == ts.EXIT_PREFLIGHT and not (tmp_path / "result.json").exists()


def test_ollama_su_un_altro_host_non_avvia_nulla(tmp_path, monkeypatch, supervised):
    monkeypatch.setattr(mt, "child_command", lambda raw: pytest.fail("figlio avviato"))
    code = supervised(Sensors(lambda n: COOL), "run", "--models", "a", "--ollama-host", "http://10.0.0.5:11434", "--guard-cmd", "true")
    assert "questo host" in code


def test_sigterm_al_supervisore_ferma_il_gruppo_e_lascia_un_esito_parziale(tmp_path, monkeypatch, ollama, supervised):
    srv = ollama(post__api_chat=hang_before_headers)
    write_scratch(tmp_path, monkeypatch, 1)

    def step(n):
        if ("POST", "/api/chat") in srv.paths():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
        return COOL

    code = supervised(Sensors(step), "run", "--models", "a", "--ollama-host", srv.host, "--guard-cmd", "true")
    res = result(tmp_path)
    assert code == 130 and srv.dropped.wait(3) and res["status"] == "PARZIALE" and res["decision"]["winner"] is None


# ---- nessuna chiamata ollama fuori dal figlio supervisionato ----------------------------------------------------

def test_i_sottocomandi_che_toccano_ollama_sono_solo_quelli_supervisionati():
    assert set(mt.SUPERVISED) == {"retrieve", "run"}


@pytest.mark.parametrize("cmd", ["retrieve", "run"])
def test_il_supervisore_non_apre_alcuna_connessione_ollama_mentre_il_figlio_gira(cmd, tmp_path, monkeypatch, supervised):
    connects = []
    real = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", lambda self, addr: (connects.append(addr), real(self, addr))[1])
    monkeypatch.setattr(mt, "child_command", lambda raw: [sys.executable, "-c", "pass"])
    extra = ["--models", "a", "--guard-cmd", "true"] if cmd == "run" else []
    assert supervised(Sensors(lambda n: COOL), cmd, *extra) == 0
    assert connects == []


@pytest.mark.parametrize("fn", ["cmd_run", "cmd_retrieve"])
def test_chiamare_ollama_senza_supervisore_e_rifiutato_prima_di_ogni_connessione(fn, tmp_path, monkeypatch):
    monkeypatch.delenv("F775_SUPERVISOR_PID")
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("connessione fuori dal figlio supervisionato"))
    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    a = _args(tmp_path)
    a.rag_url, a.api_key_file, a.mode = "http://127.0.0.1:1", None, "hybrid"
    with pytest.raises(SystemExit, match="supervisore"):
        getattr(mt, fn)(a)


def test_un_valore_dell_ambiente_non_del_proprio_padre_non_basta(tmp_path, monkeypatch):
    monkeypatch.setenv("F775_SUPERVISOR_PID", str(os.getppid() + 1))
    with pytest.raises(SystemExit, match="supervisore"):
        mt.cmd_run(_args(tmp_path))


def test_lo_script_parte_anche_lanciato_direttamente():
    """Lanciato come script (`python scripts/model_trial.py`) il modulo del supervisore va trovato lo stesso."""
    done = subprocess.run([sys.executable, str(REPO / "scripts" / "model_trial.py"), "--help"], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
