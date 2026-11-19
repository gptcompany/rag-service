"""Arresto termico esteso al rag-service (F-775, decisione A del CEO, #26): prima si ferma il container, poi si scaricano i modelli.

Una query di retrieve gia' partita continua DENTRO il rag-service anche dopo la morte del figlio e, dopo lo scarico, ricarica
il modello su ollama. Qui il rag-service e' un processo vero che fa quella chiamata, ollama e' un server HTTP vero con stato,
`docker` e' un finto su PATH (tests/conftest.py); solo il test del criterio di riavvio usa il docker vero, con un'immagine
locale di busybox (nessun pull, nessuna GPU) e si salta dove docker o busybox mancano: li sostituisce il test dell'argv.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from scripts import trial_supervisor as ts
from tests import test_model_trial_supervised as sup
from tests.test_model_trial_run import mt
from tests.test_model_trial_supervised import COOL, HOT, Sensors

supervised = sup.supervised  # la fixture del supervisore con sensori finti e figlio vero, condivisa
REPO = Path(__file__).resolve().parent.parent

RAG = """
import http.server, json, os, sys, time, urllib.request

port, ollama, marker, release = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        open(marker, "w").write("in volo")
        deadline = time.monotonic() + 30  # il retrieve hybrid calcola l'embedding: la chiamata a ollama parte quando il test lo consente
        while time.monotonic() < deadline and not os.path.exists(release):
            time.sleep(0.01)
        urllib.request.urlopen(urllib.request.Request(ollama + "/api/generate", json.dumps({"model": "embed"}).encode())).read()
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"""
CHILD = "import sys, urllib.request\nurllib.request.urlopen(urllib.request.Request(sys.argv[1] + '/query', b'{}'), timeout=30).read()\n"


class StatefulOllama:
    """ollama con stato: una generazione senza keep_alive=0 carica il modello, keep_alive=0 lo scarica (se `stubborn` no)."""

    def __init__(self, loaded=(), stubborn=False):
        self.loaded, self.events, self.stubborn = dict.fromkeys(loaded, True), [], stubborn
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.reply({"models": [{"name": n} for n in outer.loaded]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                if body.get("keep_alive") == 0:
                    outer.events.append(("unload", body["model"], time.monotonic()))
                    if not outer.stubborn:
                        outer.loaded.pop(body["model"], None)
                else:
                    outer.events.append(("load", body["model"], time.monotonic()))
                    outer.loaded[body["model"]] = True
                self.reply({})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.host = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def kinds(self):
        return [(kind, model) for kind, model, _ in self.events]


@pytest.fixture
def ollama_state():
    made = []

    def make(**kw):
        made.append(StatefulOllama(**kw))
        return made[-1]

    yield make
    for srv in made:
        srv.server.shutdown()
        srv.server.server_close()


def events(err: str) -> list[dict]:
    return [json.loads(line) for line in err.splitlines() if line.startswith("{")]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hot_child(monkeypatch):
    """Figlio che resta vivo finche' il supervisore non lo ferma; i sensori scaldano appena parte."""
    monkeypatch.setattr(mt, "child_command", lambda raw: [sys.executable, "-c", "import time; time.sleep(60)"])


def _thermal_stop(supervised, ollama_srv, n_cool=2):
    return supervised(Sensors(lambda n: COOL if n <= n_cool else HOT), "retrieve", "--ollama-host", ollama_srv.host)


def test_una_query_in_volo_all_arresto_non_ricarica_il_modello(tmp_path, monkeypatch, fake_docker, ollama_state, supervised):
    """RED/GREEN: con il solo scarico dei modelli la query ancora viva nel rag-service li ricarica dopo l'arresto."""
    srv = ollama_state(loaded=["embed"])
    port, marker, release = _free_port(), tmp_path / "in_volo", tmp_path / "release"
    rag = subprocess.Popen([sys.executable, "-c", RAG, str(port), srv.host, str(marker), str(release)], start_new_session=True)
    try:
        fake_docker.attach(rag.pid)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and socket.socket().connect_ex(("127.0.0.1", port)) != 0:
            time.sleep(0.02)
        monkeypatch.setattr(mt, "child_command", lambda raw: [sys.executable, "-c", CHILD, f"http://127.0.0.1:{port}"])
        code = supervised(Sensors(lambda n: HOT if marker.exists() else COOL), "retrieve", "--rag-url", f"http://127.0.0.1:{port}", "--ollama-host", srv.host)
        assert code == ts.EXIT_THERMAL and marker.exists()
        release.write_text("x")  # ora la query, se fosse ancora viva, chiamerebbe ollama: la si aspetta (o ne si vede la morte)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and rag.poll() is None and not any(k == "load" for k, _ in srv.kinds()):
            time.sleep(0.02)
        assert srv.kinds() == [("unload", "embed")] and srv.loaded == {}, "dopo l'arresto nessuna chiamata arriva a ollama"
        stop = [c["t"] for c in fake_docker.calls() if c["argv"][3:] == ["stop", "rag"]]
        unload = [t for kind, _, t in srv.events if kind == "unload"]
        assert stop and unload and stop[0] < unload[0], "prima il container, poi i modelli"
        assert rag.poll() is not None, "il rag-service e' fermo"
    finally:
        if rag.poll() is None:
            rag.terminate()
        rag.wait()


def test_il_comando_e_docker_compose_stop_con_l_attesa_predefinita_e_senza_sudo(monkeypatch, fake_docker, ollama_state, supervised):
    _hot_child(monkeypatch)
    launched, real_run = [], subprocess.run
    monkeypatch.setattr(ts.subprocess, "run", lambda cmd, *a, **k: (launched.append(cmd[0]), real_run(cmd, *a, **k))[1])
    assert _thermal_stop(supervised, ollama_state(loaded=["m"])) == ts.EXIT_THERMAL
    assert set(launched) == {"docker"}, "solo i permessi Docker dell'utente: niente sudo"
    calls = [c["argv"] for c in fake_docker.calls()]
    # Nessun -t/--timeout: l'attesa SIGTERM -> SIGKILL e' quella predefinita di Docker (10 s, o stop_grace_period del compose).
    assert calls[1] == ["compose", "-f", str(REPO / "docker-compose.yml"), "stop", "rag"]
    assert calls[2][3:] == ["ps", "--status", "running", "--quiet", "rag"]


def test_il_log_dell_arresto_ha_temperatura_soglia_decisione_tempi_ed_esito_dei_due_controlli(monkeypatch, capsys, ollama_state, supervised):
    _hot_child(monkeypatch)
    assert _thermal_stop(supervised, ollama_state(loaded=["m"])) == ts.EXIT_THERMAL
    done = [e for e in events(capsys.readouterr().err) if e["event"] == "thermal_shutdown"]
    assert len(done) == 1
    e = done[0]
    assert e["readings"]["gpu"] == 95.0 and e["limits"] == {"gpu": 90.0, "cpu": 100.0} and e["decision"] == "stopped"
    assert e["stop_s"] >= 0 and e["unload_s"] >= 0 and e["container_stopped"] is True and e["models_loaded"] == []


def test_stop_fallito_e_un_errore_con_allarme_non_un_arresto_riuscito(monkeypatch, capsys, fake_docker, ollama_state, supervised):
    monkeypatch.setenv("F775_FAKE_DOCKER_FAIL", "stop")
    _hot_child(monkeypatch)
    srv = ollama_state(loaded=["m"])
    assert _thermal_stop(supervised, srv) == ts.EXIT_SHUTDOWN_FAILED
    assert srv.kinds() == [("unload", "m")], "i modelli si scaricano comunque: la GPU va protetta anche se il container resiste"
    alarm = [e for e in events(capsys.readouterr().err) if e["event"] == "thermal_shutdown_failed"]
    assert len(alarm) == 1 and alarm[0]["alarm"] is True and alarm[0]["decision"] == "failed"
    assert alarm[0]["readings"]["gpu"] == 95.0 and alarm[0]["container_stopped"] is False


def test_controllo_del_container_illeggibile_non_e_un_successo(monkeypatch, capsys, ollama_state, supervised):
    monkeypatch.setenv("F775_FAKE_DOCKER_FAIL", "ps_after_stop")
    _hot_child(monkeypatch)
    assert _thermal_stop(supervised, ollama_state(loaded=["m"])) == ts.EXIT_SHUTDOWN_FAILED
    alarm = [e for e in events(capsys.readouterr().err) if e["event"] == "thermal_shutdown_failed"]
    assert alarm and alarm[0]["container_stopped"] is None


def test_modello_ancora_caricato_dopo_lo_scarico_e_un_errore_con_allarme(monkeypatch, capsys, ollama_state, supervised):
    monkeypatch.setattr(ts, "VERIFY_TIMEOUT_S", 0.3, raising=False)
    _hot_child(monkeypatch)
    assert _thermal_stop(supervised, ollama_state(loaded=["m"], stubborn=True)) == ts.EXIT_SHUTDOWN_FAILED
    alarm = [e for e in events(capsys.readouterr().err) if e["event"] == "thermal_shutdown_failed"]
    assert alarm and alarm[0]["models_loaded"] == ["m"] and alarm[0]["container_stopped"] is True


def test_nessun_riavvio_automatico_se_la_temperatura_non_rientra(monkeypatch, capsys, fake_docker):
    monkeypatch.setattr(mt.trial_supervisor, "Sensors", lambda: Sensors(lambda n: HOT))
    with pytest.raises(SystemExit) as stop:
        mt.main(["rag-restart"])
    assert stop.value.code == ts.EXIT_PREFLIGHT and not [c for c in fake_docker.calls() if "start" in c["argv"]]
    assert "rag_restart_refused" in capsys.readouterr().err


def test_il_riavvio_esplicito_sotto_soglia_e_registrato(monkeypatch, capsys, fake_docker):
    monkeypatch.setattr(mt.trial_supervisor, "Sensors", lambda: Sensors(lambda n: COOL))
    mt.main(["rag-restart"])
    assert [c["argv"] for c in fake_docker.calls()] == [["compose", "-f", str(REPO / "docker-compose.yml"), "start", "rag"]]
    done = [e for e in events(capsys.readouterr().err) if e["event"] == "rag_restart"]
    assert len(done) == 1 and done[0]["readings"]["gpu"] == 50.0 and done[0]["limits"] == {"gpu": 90.0, "cpu": 100.0}


# ---- la restart policy del compose non rialza il servizio fermato -----------------------------------------------------

def _real_docker(monkeypatch) -> str:
    path = os.pathsep.join(p for p in os.environ["PATH"].split(os.pathsep) if "fake_docker" not in p)
    monkeypatch.setenv("PATH", path)
    monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)  # prevarrebbe sul `name:` del compose di prova e toccherebbe un deployment vero
    busybox = shutil.which("busybox", path=path)
    if not shutil.which("docker", path=path) or not busybox:
        pytest.skip("docker o busybox mancano: coperto dal test dell'argv di `docker compose stop`")
    ldd = subprocess.run(["ldd", busybox], capture_output=True, text=True, check=False)
    ready = all(subprocess.run(["docker", *cmd], capture_output=True, check=False).returncode == 0 for cmd in (["info"], ["compose", "version"]))
    if not ready or "not a dynamic" not in ldd.stdout + ldd.stderr:
        pytest.skip("docker, il plugin compose o un busybox statico mancano: coperto dal test dell'argv di `docker compose stop`")
    return busybox


def test_la_restart_policy_unless_stopped_non_rialza_il_servizio_dopo_lo_stop(tmp_path, monkeypatch):
    """Il controllo vale sul docker vero: un'uscita non richiesta (il processo muore da solo) fa ripartire il servizio, lo stop no."""
    busybox = _real_docker(monkeypatch)
    policy = yaml.safe_load((REPO / "docker-compose.yml").read_text())["services"]["rag"]["restart"]
    tag, project = f"f775-probe:{uuid.uuid4().hex[:8]}", f"f775probe{uuid.uuid4().hex[:8]}"
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w") as t:
        t.add(busybox, arcname="busybox")
    subprocess.run(["docker", "import", "--change", 'CMD ["/busybox","sh","-c","sleep 2; exit 1"]', "-", tag], input=tar.getvalue(), check=True, capture_output=True)
    compose = tmp_path / "docker-compose.yml"
    compose.write_text(f"name: {project}\nservices:\n  rag:\n    image: {tag}\n    network_mode: none\n    restart: {policy}\n")

    def state() -> dict:
        cid = subprocess.run(["docker", "compose", "-f", str(compose), "ps", "-a", "-q", "rag"], capture_output=True, text=True, check=True).stdout.strip()
        return json.loads(subprocess.run(["docker", "inspect", cid], capture_output=True, text=True, check=True).stdout)[0]

    def wait_for(cond, seconds=8.0) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(0.1)
        return False

    try:
        subprocess.run(["docker", "compose", "-f", str(compose), "up", "-d", "rag"], check=True, capture_output=True)
        assert wait_for(lambda: state()["RestartCount"] >= 1, 15), "controprova: un'uscita non richiesta la policy la riavvia"
        assert ts.stop_rag_service(compose, "rag") and ts.rag_service_running(compose, "rag") is False
        time.sleep(3)  # molte volte il backoff iniziale della policy (100 ms)
        assert state()["State"]["Running"] is False and ts.rag_service_running(compose, "rag") is False
    finally:
        subprocess.run(["docker", "compose", "-f", str(compose), "down", "-t", "0"], capture_output=True, check=False)
        subprocess.run(["docker", "rmi", tag], capture_output=True, check=False)


# ---- guasti dello strumento: mai un successo presunto -----------------------------------------------------------------

@pytest.mark.parametrize("exc", [FileNotFoundError("docker"), subprocess.TimeoutExpired("docker", 60)])
def test_docker_assente_o_appeso_e_un_arresto_non_riuscito(exc, monkeypatch, ollama_state):
    def broken(*args):
        raise exc

    monkeypatch.setattr(ts, "_compose", broken, raising=False)
    result = ts.thermal_shutdown(ollama_state(loaded=["m"]).host)
    assert result["ok"] is False and result["container_stopped"] is False and result["models_loaded"] == []
    assert ts.rag_service_running() is None


def test_ollama_irraggiungibile_alla_verifica_non_e_un_successo(fake_docker):
    result = ts.thermal_shutdown("http://127.0.0.1:9")
    assert result["ok"] is False and result["container_stopped"] is True and result["models_loaded"] is None


def test_riavvio_rifiutato_con_sensori_illeggibili_o_start_fallito(monkeypatch, capsys, fake_docker):
    def unreadable():
        raise ts.SensorError("coretemp assente")

    monkeypatch.setattr(mt.trial_supervisor, "Sensors", unreadable)
    with pytest.raises(SystemExit) as stop:
        mt.main(["rag-restart"])
    assert stop.value.code == ts.EXIT_PREFLIGHT and not fake_docker.calls()

    class Unreadable(Sensors):
        def read(self):
            raise ts.SensorError("temp1_input sparito")

    assert ts.restart_rag_service(Unreadable(lambda n: COOL)) is False and not fake_docker.calls()
    monkeypatch.setenv("F775_FAKE_DOCKER_FAIL", "start")
    assert ts.restart_rag_service(Sensors(lambda n: COOL)) is False
    assert {"rag_restart_refused", "rag_restart_failed"} <= {e["event"] for e in events(capsys.readouterr().err)}
    monkeypatch.setattr(ts, "_compose", lambda *a: (_ for _ in ()).throw(subprocess.TimeoutExpired("docker", 60)))
    assert ts.restart_rag_service(Sensors(lambda n: COOL)) is False


def test_retrieve_non_parte_se_il_compose_non_ha_un_rag_service_in_esecuzione(monkeypatch, capsys, fake_docker, ollama_state, supervised):
    """Un compose di un altro progetto fermerebbe 'con successo' un servizio che non e' quello in uso: il figlio non parte."""
    fake_docker.detach()
    monkeypatch.setattr(mt, "child_command", lambda raw: pytest.fail("figlio avviato senza un rag-service da fermare"))
    assert _thermal_stop(supervised, ollama_state()) == ts.EXIT_PREFLIGHT
    assert "non in esecuzione" in capsys.readouterr().err


def test_ollama_che_gocciola_la_verifica_non_trattiene_il_supervisore(monkeypatch, capsys):
    class Slow(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            for _ in range(200):  # un byte ogni 50 ms: ogni lettura risponde, il corpo no
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(0.05)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    monkeypatch.setattr(ts, "VERIFY_TIMEOUT_S", 0.3, raising=False)
    t0 = time.monotonic()
    try:
        assert ts._models_loaded(f"http://127.0.0.1:{srv.server_port}") is None
    finally:
        srv.shutdown()
        srv.server_close()
    assert time.monotonic() - t0 < 3 and "thermal_model_check_failed" in capsys.readouterr().err


def test_una_voce_senza_nome_nell_elenco_non_e_nessun_modello(monkeypatch):
    class Odd(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            data = json.dumps({"models": [{"model": "senza-nome"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Odd)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    try:
        assert ts._models_loaded(f"http://127.0.0.1:{srv.server_port}") is None
    finally:
        srv.shutdown()
        srv.server_close()


def test_lo_scarico_ferma_i_modelli_identificabili_anche_con_una_voce_malformata(monkeypatch):
    posted = []

    class Mixed(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            data = json.dumps({"models": [{"name": "a"}, {"model": "senza-nome"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            posted.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0))))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Mixed)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    try:
        ts.unload_models(f"http://127.0.0.1:{srv.server_port}")
    finally:
        srv.shutdown()
        srv.server_close()
    assert posted == [{"model": "a", "keep_alive": 0}]
