"""Prova modelli (F-775): trasporto verso ollama su uno stack HTTP reale (server finto su 127.0.0.1).

Arresto termico in ogni fase (connessione, attesa delle intestazioni, stream bloccato), errori di trasporto,
stato HTTP e decodifica normalizzati in OllamaError.
"""
from __future__ import annotations

import hashlib
import json
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.test_model_trial_run import FakeSensors, _args, mt

GOOD_LINES = [json.dumps({"message": {"content": json.dumps({"answer": "x", "paper": "p0"})}}), json.dumps({"done": True})]
PS_OK = {"models": [{"name": "a", "size": 10, "size_vram": 10}]}


def send_json(h, obj, status=200):
    body = json.dumps(obj).encode()
    h.send_response(status)
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)


def stream(lines):
    def route(h, srv):
        h.send_response(200)
        h.end_headers()
        for line in lines:
            h.wfile.write(line.encode() + b"\n")
    return route


def wait_for_client_drop(h, srv):
    """Il server vede cadere il client: e' cio' che, in ollama, annulla la generazione."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if select.select([h.connection], [], [], 0.02)[0] and not h.connection.recv(1, socket.MSG_PEEK):
                srv.dropped.set()
                return
        except OSError:  # reset dal client: anche questo e' un client caduto
            srv.dropped.set()
            return


def hang_before_headers(h, srv):
    wait_for_client_drop(h, srv)


def hang_after_headers(h, srv):
    h.send_response(200)
    h.end_headers()
    h.wfile.write(b'{"message": {"content": "{"}}\n')
    h.wfile.flush()
    wait_for_client_drop(h, srv)


class FakeOllama:
    def __init__(self, **routes):
        self.routes = {("POST", "/api/chat"): stream(GOOD_LINES), ("GET", "/api/ps"): lambda h, s: send_json(h, PS_OK),
                       ("POST", "/api/generate"): lambda h, s: send_json(h, {})}
        self.routes.update({(k.split("__")[0].upper(), "/" + k.split("__")[1].replace("_", "/")): v for k, v in routes.items()})
        self.calls, self.dropped = [], threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def serve(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
                outer.calls.append((self.command, self.path, body))
                outer.routes[(self.command, self.path)](self, outer)

            do_GET = do_POST = serve

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.host = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def paths(self):
        return [(m, p) for m, p, _ in self.calls]


@pytest.fixture
def ollama():
    made = []

    def make(**routes):
        made.append(FakeOllama(**routes))
        return made[-1]

    yield make
    for srv in made:
        srv.server.shutdown()
        srv.server.server_close()


def hot_after(seconds):
    hot = threading.Event()
    threading.Timer(seconds, hot.set).start()
    return hot


@pytest.mark.parametrize("phase", [hang_before_headers, hang_after_headers], ids=["prima_delle_intestazioni", "stream_bloccato"])
def test_arresto_termico_interrompe_la_richiesta_in_ogni_fase(ollama, monkeypatch, phase):
    """thermal_before_headers e stream fermo: il thread chiamante chiude la connessione (annulla la generazione) e scarica il modello."""
    monkeypatch.setattr(mt, "HOT_POLL_S", 0.01)
    monkeypatch.setattr(mt, "REQUEST_TIMEOUT_S", 60)  # l'arresto non dipende dal timeout di risorsa
    srv = ollama(post__api_chat=phase)
    t0 = time.monotonic()
    with pytest.raises(mt.ThermalStop):
        mt.ollama_ask(srv.host, "m", "c", "q", hot_after(0.1))
    assert time.monotonic() - t0 < 3
    assert srv.dropped.wait(3), "il server non ha visto cadere la connessione: la generazione continuerebbe"
    assert ("POST", "/api/generate", {"model": "m", "keep_alive": 0}) in srv.calls


def test_arresto_durante_la_connessione_impedisce_l_invio_della_richiesta(ollama, monkeypatch):
    """Il watcher scatta mentre connect() e' ancora in corso: a connessione ultimata nessuna inferenza deve partire."""
    import http.client

    monkeypatch.setattr(mt, "HOT_POLL_S", 0.01)
    srv = ollama()
    entered, release, original = threading.Event(), threading.Event(), http.client.HTTPConnection.connect

    def slow_connect(self):
        original(self)
        entered.set()
        release.wait(3)

    monkeypatch.setattr(http.client.HTTPConnection, "connect", slow_connect)
    hot = threading.Event()
    threading.Timer(0.05, hot.set).start()
    try:
        with pytest.raises(mt.ThermalStop):
            mt.ollama_ask(srv.host, "m", "c", "q", hot)
    finally:
        release.set()
    time.sleep(0.3)  # il worker, sbloccato, non deve inviare /api/chat
    assert ("POST", "/api/chat") not in srv.paths()


def test_fallimento_dello_stop_del_modello_non_maschera_l_arresto(ollama, monkeypatch, capsys):
    monkeypatch.setattr(mt, "HOT_POLL_S", 0.01)
    srv = ollama(post__api_chat=hang_before_headers, post__api_generate=lambda h, s: send_json(h, {}, 500))
    with pytest.raises(mt.ThermalStop):
        mt.ollama_ask(srv.host, "m", "c", "q", hot_after(0.05))
    assert "thermal_model_stop_failed" in capsys.readouterr().err


def test_flusso_completo_viene_punteggiato_con_residenza(ollama):
    reply, latency, ps = mt.ollama_ask(ollama().host, "a", "c", "q", threading.Event())
    assert reply == {"answer": "x", "paper": "p0"} and latency >= 0 and ps == {"size": 10, "size_vram": 10}


@pytest.mark.parametrize("routes, match", [
    ({"post__api_chat": stream(['{"error": "model failed to load"}'])}, "model failed"),
    ({"post__api_chat": stream([GOOD_LINES[0]])}, "senza done"),
    ({"post__api_chat": stream(["<html>"])}, ""),
    ({"post__api_chat": stream(["null"])}, "evento"),
    ({"post__api_chat": stream(['{"message": {"content": null}}'])}, "evento"),
    ({"post__api_chat": lambda h, s: send_json(h, {}, 500)}, "500"),
    ({"post__api_chat": lambda h, s: (h.send_response(302), h.send_header("Location", "http://127.0.0.1:1/"), h.end_headers())}, "302"),
    ({"post__api_chat": lambda h, s: h.connection.close()}, ""),
    ({"get__api_ps": lambda h, s: send_json(h, {}, 503)}, "503"),
    ({"get__api_ps": lambda h, s: send_json(h, {"error": "boom"})}, "models"),
    ({"get__api_ps": lambda h, s: send_json(h, ["x"])}, "models"),
    ({"get__api_ps": stream(["{not json"])}, ""),
], ids=["errore_nello_stream", "senza_done", "riga_non_json", "evento_null", "contenuto_null", "http_500", "redirect", "connessione_chiusa",
        "ps_http_503", "ps_senza_models", "ps_non_oggetto", "ps_json_non_valido"])
def test_ogni_errore_di_trasporto_stato_o_decodifica_e_un_errore_ollama(ollama, routes, match):
    with pytest.raises(mt.OllamaError, match=match or None):
        mt.ollama_ask(ollama(**routes).host, "a", "c", "q", threading.Event())


def test_connessione_rifiutata_e_un_errore_ollama(ollama):
    srv = ollama()
    srv.server.shutdown()
    srv.server.server_close()
    with pytest.raises(mt.OllamaError):
        mt.ollama_ask(srv.host, "a", "c", "q", threading.Event())


@pytest.mark.parametrize("second", [
    {"post__api_chat": "drop"}, {"get__api_ps": "ps_invalid"}, {"get__api_ps": "ps_503"}, {"get__api_ps": "ps_no_models"}],
    ids=["post_rifiutato", "ps_json_non_valido", "ps_stato_http", "ps_senza_models"])
def test_errore_di_trasporto_non_cancella_i_round_completi(tmp_path, monkeypatch, ollama, second):
    """Controesempio del report: il secondo round fallisce sul POST o su /api/ps; il primo resta nell'esito."""
    (path, kind), = second.items()
    count = {"n": 0}
    behaviours = {
        "drop": lambda h, s: h.connection.close(),
        "ps_invalid": stream(["{not json"]),
        "ps_503": lambda h, s: send_json(h, {}, 503),
        "ps_no_models": lambda h, s: send_json(h, {"error": "boom"}),
    }
    normal = {"post__api_chat": stream(GOOD_LINES), "get__api_ps": lambda h, s: send_json(h, PS_OK)}[path]

    def route(h, s):
        count["n"] += 1
        (behaviours[kind] if count["n"] == 2 else normal)(h, s)

    srv = ollama(**{path: route})
    a = write_scratch(tmp_path, monkeypatch, 2)
    a.ollama_host = srv.host
    mt.cmd_run(a)
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["rounds_completed"] == 1 and res["stop_reason"].startswith("errore ollama")


def write_scratch(tmp_path, monkeypatch, n):
    """Scratch con n domande (un paper ciascuna), contesti e impronta coerenti, sensori finti."""
    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    qs = [{"id": f"q{i}", "paper": f"p{i}", "answer": "x", "question": "?"} for i in range(n)]
    (tmp_path / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    sha = hashlib.sha1((tmp_path / "questions.jsonl").read_bytes()).hexdigest()
    (tmp_path / "retrieval.json").write_text(json.dumps({"questions_sha1": sha}))
    (tmp_path / "contexts.jsonl").write_text("".join(json.dumps({"id": q["id"], "context": "c"}) + "\n" for q in qs))
    monkeypatch.setattr(mt, "Sensors", FakeSensors)
    a = _args(tmp_path)
    a.models = ["a"]
    return a
