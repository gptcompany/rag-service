#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "numpy"]
# ///
"""Prova dei modelli di generazione su ollama (F-775, fase 2: NON eseguire senza il SI' dell'operatore).

Sottocomandi, nell'ordine:
  questions  genera le domande dal testo dei chunk gia' indicizzati (cloze, nessuna domanda a mano)
  retrieve   recupera UNA volta il contesto per ogni domanda (POST /query context_only) e lo salva
  run        stesso contesto a ogni candidato su TUTTE le domande; per ogni candidato ammesso (interamente
             in GPU) posterior Beta(1+corrette, 1+errate), P(migliore) per Monte Carlo, vince l'argmax

Lo scratch deve stare sotto /media/sam/1TB/scratch/f775/ (mai /tmp). La chiave API si legge da un
file 0600 (RAG_API_KEY_FILE) e non viene mai stampata.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

SCRATCH_ROOT = Path("/media/sam/1TB/scratch/f775")

# LIMITE di sicurezza, non un parametro: una lettura del sensore che non risponde entro 10 s equivale a
# telemetria persa e ferma la prova (fail-closed), invece di lasciarla girare senza controllo termico.
SENSOR_READ_TIMEOUT_S = 10
HWMON = Path("/sys/class/hwmon")
# LIMITE di reattivita': la connessione si chiude entro 1 s da `hot`; un risveglio al secondo costa nulla.
HOT_POLL_S = 1.0
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
# LIMITE di risorsa (un ollama appeso non deve bloccare la prova per sempre), NON il meccanismo di arresto termico:
# `hot` e' sorvegliato dal thread chiamante in ogni fase e tronca la connessione indipendentemente da questo timeout.
REQUEST_TIMEOUT_S = 600
# LIMITE di risorsa: /api/ps e l'unload del modello rispondono subito o la loro risposta non serve piu'.
SHORT_REQUEST_TIMEOUT_S = 30
# LIMITE di risorsa, non un parametro statistico: 200k campioni per candidato occupano pochi MB (3 x 1.6 MB
# in float64) e portano l'errore Monte Carlo su P(migliore) a ~sqrt(0.25/N) = 0.1%, ben sotto ogni differenza
# che cambi la scelta. Il seme fissa solo la riproducibilita' del report.
MC_SAMPLES = 200_000
MC_SEED = 775
# ASSUNZIONE dichiarata: prior uniforme Beta(1, 1) sulla proporzione di risposte corrette di ogni candidato.
PRIOR_CORRECT = PRIOR_WRONG = 1


# Segni meno tipografici (trattini Unicode U+2010..2013, meno matematico U+2212, varianti small/fullwidth) -> "-".
_MINUS = "\u2010\u2011\u2012\u2013\u2212\ufe63\uff0d"
_MINUS_TO_ASCII = str.maketrans(dict.fromkeys(_MINUS, "-"))
# Abbreviazioni di riferimento (Fig., Eq., ...): il punto non chiude la frase, altrimenti il numero resta solo.
_ABBREV = ("Fig", "Figs", "Tab", "Tabs", "Sec", "Secs", "Eq", "Eqs", "App", "Ch")
_SENT = re.compile("".join(rf"(?<!\b{a}\.)" for a in _ABBREV) + r"(?<=[.!?])\s+", re.IGNORECASE)
_NUM = re.compile(rf"(?<![\w.])[-+{_MINUS}]?\d+(?:\.\d+)?\s?%?")
_METRIC = re.compile(r"accuracy|F1|precision|recall|Sharpe|AUC|error|BLEU|loss|growth|return", re.IGNORECASE)
_REF = re.compile(r"\b(?:Fig(?:ure)?s?|Tab(?:le)?s?|Sec(?:tion)?s?|Eq(?:uation)?s?|App(?:endix)?|Ch(?:apter)?)\.?\s*\d", re.IGNORECASE)
_DATASET = re.compile(r"\b(?:the|on|using)\s+((?:[A-Z][\w-]+)(?:\s[A-Z][\w-]+){0,2})\s(?:dataset|data set|benchmark)")
_METHOD = re.compile(r"\b(?:using|via|by|with)\s+(?:the\s+)?((?:[A-Z][\w-]+)(?:\s[A-Z][\w-]+){0,2})\s(?:method|algorithm|model|estimator)")


def log(event: str, **fields) -> None:
    print(json.dumps({"t": round(time.time(), 3), "event": event, **fields}, sort_keys=True), file=sys.stderr, flush=True)


def check_scratch(path: Path) -> Path:
    path = path.resolve()
    if SCRATCH_ROOT.resolve() not in (path, *path.parents):
        raise SystemExit(f"scratch fuori da {SCRATCH_ROOT}: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_secret_file(path: str) -> str:
    st = os.stat(path)
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise SystemExit("il file chiave deve essere 0600")
    return Path(path).read_text().strip()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).translate(_MINUS_TO_ASCII)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\s+%", "%", text)
    return re.sub(r"[^a-z0-9.%+-]+", " ", text).strip().rstrip(".")


def paper_key(name: str) -> str:
    """1411.3615v1.pdf -> 1411.3615 (le versioni dello stesso paper coincidono)."""
    stem = Path(str(name)).name
    stem = re.sub(r"\.pdf$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"v\d+$", "", stem)


# ---- domande -------------------------------------------------------------

def _cloze(sentence: str, answer: str, kind: str, paper: str) -> dict | None:
    # Il valore intero e non un suo prefisso o suffisso: "94" non va svuotato dentro "194.2" o "94.2".
    whole = re.compile(r"(?<![\w.])" + re.escape(answer) + r"(?![\w]|\.\d)")
    if len(whole.findall(sentence)) != 1 or sentence.count(answer) != 1:
        return None
    qid = hashlib.sha1(f"{paper}|{sentence}|{answer}".encode()).hexdigest()[:12]
    return {
        "id": qid, "paper": paper, "kind": kind, "answer": answer.strip(),
        "question": "Which value, dataset or method fills the blank? " + sentence.replace(answer, "____"),
    }


def questions_from_chunks(chunks: dict) -> list[dict]:
    out: dict[str, dict] = {}
    for chunk in chunks.values():
        paper = paper_key(chunk.get("file_path", ""))
        if not paper:
            continue
        for sent in _SENT.split(chunk.get("content", "")):
            sent = " ".join(sent.split())
            found = []
            nums = _NUM.findall(sent)
            # Con piu' numeri l'associazione alla metrica e' ambigua: la frase non genera una domanda "value".
            # Un numero di figura, tabella o sezione non e' il valore della metrica.
            if _METRIC.search(sent) and len(nums) == 1 and not _REF.search(sent):
                found.append((nums[0].strip(), "value"))
            if m := _DATASET.search(sent):
                found.append((m.group(1), "dataset"))
            if m := _METHOD.search(sent):
                found.append((m.group(1), "method"))
            for answer, kind in found:
                q = _cloze(sent, answer, kind, paper)
                if q:
                    out.setdefault(q["id"], q)
    # Lo stesso testo con chiavi diverse (altro paper o altra risposta) non ha una risposta univoca: si scarta.
    keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for q in out.values():
        keys[q["question"]].add((q["paper"], q["answer"]))
    return sorted((q for q in out.values() if len(keys[q["question"]]) == 1), key=lambda q: q["id"])


# ---- punteggio -----------------------------------------------------------

def is_correct(question: dict, reply: dict) -> bool:
    """Risposta uguale all'attesa dopo la normalizzazione (niente alternative aggiuntive, segno e unita' inclusi)
    e paper citato esatto (versione ignorata)."""
    expected = normalize(question["answer"])
    raw = reply.get("paper", "")
    # Una stringa e' UNA citazione (un nome file puo' contenere virgole); piu' paper solo come elenco JSON.
    # Ogni citazione intera si riduce al suo nome file prima del confronto: un segmento di directory non e' un paper.
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    cited = {paper_key(t).strip().casefold() for t in map(str, items) if t.strip()}
    return bool(expected) and normalize(reply.get("answer", "")) == expected and paper_key(question["paper"]).strip().casefold() in cited

def p_best(scores: dict[str, tuple[int, int]]) -> dict[str, float]:
    """P(il candidato ha la proporzione di corrette piu' alta) campionando i posterior Beta(1+k, 1+n-k)."""
    names = sorted(scores)
    rng = np.random.default_rng(MC_SEED)
    draws = np.stack([rng.beta(PRIOR_CORRECT + scores[m][0], PRIOR_WRONG + scores[m][1] - scores[m][0], MC_SAMPLES)
                      for m in names])
    wins = np.bincount(draws.argmax(axis=0), minlength=len(names))
    return {m: float(w) / MC_SAMPLES for m, w in zip(names, wins)}


def pick_winner(scores: dict[str, tuple[int, int]], vram: dict[str, int], admitted: set[str]) -> dict:
    """Argmax di P(migliore) fra gli ammessi; solo a posterior identici (stessi k e n) vince il minore size_vram."""
    everyone = p_best(scores) if scores else {}
    pool = {m: s for m, s in scores.items() if m in admitted}
    if not pool:
        return {"winner": None, "reason": "nessun candidato ammesso (size_vram != size)", "p_best_all": everyone}
    p = p_best(pool)
    best = max(sorted(p), key=p.get)
    tied = sorted(m for m in pool if pool[m] == pool[best])
    if len(tied) > 1:
        return {"winner": min(tied, key=lambda m: (vram[m], m)), "reason": "parita' esatta: minore size_vram",
                "tied": tied, "p_best": p, "p_best_all": everyone}
    return {"winner": best, "reason": "argmax P(migliore)", "p_best": p, "p_best_all": everyone}


# ---- sensori e arresto ---------------------------------------------------

def gpu_slowdown_limit() -> float | None:
    out = subprocess.run(["nvidia-smi", "-q", "-d", "TEMPERATURE"], capture_output=True, text=True, check=False).stdout
    m = re.search(r"GPU Slowdown Temp\s*:\s*(\d+)", out)
    return float(m.group(1)) if m else None


def cpu_limits() -> tuple[Path, float] | None:
    for hw in sorted(HWMON.glob("hwmon*")):
        if (hw / "name").read_text().strip() == "coretemp" and (hw / "temp1_max").exists():
            return hw / "temp1_input", int((hw / "temp1_max").read_text()) / 1000
    return None


class Sensors:
    """Campiona VRAM e temperature senza pause scelte a mano e alza `hot` al limite dichiarato dai sensori."""

    def __init__(self) -> None:
        self.gpu_limit = gpu_slowdown_limit()
        cpu = cpu_limits()
        # Fail-closed: senza soglia dichiarata non si parte, la si inventerebbe a mano.
        if self.gpu_limit is None or cpu is None:
            raise SystemExit("soglie di slowdown non leggibili dai sensori: prova non avviata")
        self.cpu_input, self.cpu_limit = cpu
        self.hot = threading.Event()
        self.stop = threading.Event()
        self.peak_vram_mib = 0
        self.peak_gpu_c = 0.0
        self.peak_cpu_c = 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=SENSOR_READ_TIMEOUT_S).stdout.strip()
        mem, gpu = (float(x) for x in out.split(","))
        cpu = int(self.cpu_input.read_text()) / 1000
        self.peak_vram_mib = max(self.peak_vram_mib, int(mem))
        self.peak_gpu_c, self.peak_cpu_c = max(self.peak_gpu_c, gpu), max(self.peak_cpu_c, cpu)
        if gpu >= self.gpu_limit or cpu >= self.cpu_limit:
            log("thermal_stop", gpu=gpu, gpu_limit=self.gpu_limit, cpu=cpu, cpu_limit=self.cpu_limit)
            self.hot.set()

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                self._sample()
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                # Fail-closed: un sensore illeggibile equivale a soglia superata.
                log("sensor_unreadable", error=str(exc))
                self.hot.set()
                return

    def __enter__(self):
        self._sample()  # un primo campione valido prima di qualsiasi inferenza
        if self.hot.is_set():
            raise SystemExit("soglia termica gia' raggiunta: prova non avviata")
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def append_durable(path: Path, record: dict) -> None:
    """Una riga JSON per record, scritta per intero (il buffer completa le scritture parziali) e sincronizzata su disco."""
    with path.open("ab") as fh:
        fh.write((json.dumps(record, sort_keys=True) + "\n").encode())
        fh.flush()
        os.fsync(fh.fileno())


def guard_ok(cmd: list[str]) -> bool:
    return subprocess.run(cmd, capture_output=True, check=False).returncode == 0


# ---- ollama / rag --------------------------------------------------------

PROMPT = (
    "Answer ONLY from the context. Reply as JSON {\"answer\": \"<the missing value, dataset or method>\", "
    "\"paper\": \"<file name of the source paper as shown in the context>\"}.\n\nContext:\n%s\n\nQuestion: %s"
)


class ThermalStop(Exception):
    pass


class OllamaError(Exception):
    """Generazione fallita o incompleta: non e' una risposta sbagliata e non va contata come tale."""


def _open(host: str, box: dict, timeout: float) -> http.client.HTTPConnection:
    """Connessione http.client (nessun proxy ambientale, nessun redirect) il cui socket e' registrato in `box` per poterlo troncare."""
    url = urlparse(host)
    if url.scheme != "http" or not url.hostname:
        raise OllamaError(f"URL ollama non valido (serve http://host:porta): {host}")
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    conn.connect()
    box["sock"] = conn.sock  # a fine risposta http.client lo cede alla risposta: lo si tiene qui per troncarlo
    return conn


def _send(conn: http.client.HTTPConnection, method: str, path: str, body: dict | None = None) -> http.client.HTTPResponse:
    payload = None if body is None else json.dumps(body).encode()
    conn.request(method, path, body=payload, headers={"Content-Type": "application/json"} if payload else {})
    resp = conn.getresponse()
    if resp.status != 200:
        raise OllamaError(f"HTTP {resp.status} su {path}")
    return resp


def _sever(sock: socket.socket | None) -> None:
    """Chiude il socket sotto la lettura in corso (shutdown, che non attende i lock del buffer): il server vede
    cadere il client e annulla la generazione. Vale anche in attesa delle intestazioni."""
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # gia' chiuso dall'altra parte


def _stop_model(host: str, model: str) -> None:
    """Arresto termico: scarica il modello (equivale a `ollama stop`) dopo aver troncato la richiesta in corso."""
    try:
        conn = _open(host, {}, SHORT_REQUEST_TIMEOUT_S)
        try:
            _send(conn, "POST", "/api/generate", {"model": model, "keep_alive": 0}).read()
        finally:
            conn.close()
        log("thermal_model_stop", model=model)
    except (OllamaError, OSError, http.client.HTTPException) as exc:
        log("thermal_model_stop_failed", model=model, error=str(exc))


def _event_text(line: bytes) -> tuple[str, bool]:
    obj = json.loads(line)
    msg = obj.get("message", {}) if isinstance(obj, dict) else None
    if not isinstance(obj, dict) or not isinstance(msg, dict) or not isinstance(msg.get("content", ""), str):
        raise OllamaError(f"evento di stream non valido: {line[:120]!r}")
    if "error" in obj:
        raise OllamaError(str(obj["error"]))
    return msg.get("content", ""), bool(obj.get("done"))


def _exchange(host: str, model: str, context: str, question: str, box: dict, cancel: threading.Event) -> None:
    """Richiesta, stream e /api/ps: ogni errore di trasporto, stato HTTP o decodifica diventa OllamaError in `box`."""
    try:
        t0 = time.monotonic()
        text, finished = "", False
        conn = _open(host, box, REQUEST_TIMEOUT_S)
        if cancel.is_set():  # arresto scattato durante connect(): nessuna inferenza deve partire
            conn.close()
            return
        resp = _send(conn, "POST", "/api/chat", {
            "model": model, "stream": True, "think": False, "format": "json",
            "messages": [{"role": "user", "content": PROMPT % (context, question)}],
        })
        for line in resp:
            if cancel.is_set():
                return
            if line.strip():
                chunk, done = _event_text(line)
                text += chunk
                finished = finished or done
        conn.close()
        if cancel.is_set():
            return
        if not finished:
            raise OllamaError("stream chiuso senza done")
        latency = time.monotonic() - t0
        try:
            reply = json.loads(text)
        except ValueError:
            reply = {}
        conn = _open(host, box, SHORT_REQUEST_TIMEOUT_S)
        if cancel.is_set():
            conn.close()
            return
        body = json.loads(_send(conn, "GET", "/api/ps").read())
        conn.close()
        if not isinstance(body, dict) or not isinstance(body.get("models"), list):
            raise OllamaError(f"/api/ps senza elenco models: {str(body)[:200]}")
        # /api/ps riporta il nome canonico ("qwen3" -> "qwen3:latest").
        names = {model, model + ":latest"}
        ps = next((m for m in body["models"] if m.get("name") in names or m.get("model") in names), {})
        box["result"] = (reply if isinstance(reply, dict) else {}, latency, {"size": ps.get("size"), "size_vram": ps.get("size_vram")})
    except OllamaError as exc:
        box["error"] = exc
    except Exception as exc:  # noqa: BLE001 - il thread non deve morire senza esito: il round si scarta, i precedenti restano
        log("exchange_failed", error=repr(exc))
        box["error"] = OllamaError(f"{type(exc).__name__}: {exc}")


def ollama_ask(host: str, model: str, context: str, question: str, hot: threading.Event) -> tuple[dict, float, dict]:
    """La richiesta gira in un thread; questo thread sorveglia `hot` in ogni fase (connessione, intestazioni, stream)
    e a soglia termica tronca la connessione, scarica il modello e ferma la prova."""
    if hot.is_set():
        raise ThermalStop(model)
    box: dict = {}
    cancel, finished = threading.Event(), threading.Event()

    def work() -> None:
        try:
            _exchange(host, model, context, question, box, cancel)
        finally:
            finished.set()

    threading.Thread(target=work, daemon=True).start()
    while not finished.wait(HOT_POLL_S):
        if hot.is_set():
            break
    if hot.is_set():
        cancel.set()
        _sever(box.get("sock"))
        _stop_model(host, model)
        raise ThermalStop(model)
    if "error" in box:
        raise box["error"]
    return box["result"]


def cmd_questions(a) -> None:
    scratch = check_scratch(Path(a.scratch))
    chunks = json.loads(Path(a.chunks).read_text())
    qs = questions_from_chunks(chunks)
    (scratch / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    log("questions", n=len(qs), papers=len({q["paper"] for q in qs}))


def cmd_retrieve(a) -> None:
    import requests

    scratch = check_scratch(Path(a.scratch))
    key = read_secret_file(a.api_key_file) if a.api_key_file else ""
    out = scratch / "contexts.jsonl"
    questions = (scratch / "questions.jsonl").read_bytes()
    # L'impronta lega la cache all'insieme di domande: un nuovo `questions` non si mescola ai contesti vecchi.
    settings = {"rag_url": a.rag_url, "mode": a.mode, "questions_sha1": hashlib.sha1(questions).hexdigest()}
    meta = scratch / "retrieval.json"
    if out.exists() and (not meta.exists() or json.loads(meta.read_text()) != settings):
        raise SystemExit(f"contesti in cache recuperati con altre impostazioni (attese {settings}): usare uno scratch nuovo")
    meta.write_text(json.dumps(settings))
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    missing = 0
    if not questions.strip():
        raise SystemExit("nessuna domanda: rilanciare `questions` con i chunk indicizzati")
    for line in questions.decode().splitlines():
        q = json.loads(line)
        if q["id"] in done:
            continue
        # Niente redirect: la chiave API non deve seguire un 3xx verso un altro host.
        r = requests.post(f"{a.rag_url}/query", headers={"X-API-Key": key} if key else {}, timeout=600,
                          allow_redirects=False,
                          json={"query": q["question"], "mode": a.mode, "context_only": True})
        r.raise_for_status()
        body = r.json()
        ctx = body.get("context")
        if not body.get("success") or not isinstance(ctx, str) or not ctx.strip():
            # Mai in cache un contesto vuoto: falserebbe il confronto e la domanda non verrebbe piu' ritentata.
            log("retrieve_failed", qid=q["id"], error=body.get("error", "empty context"))
            missing += 1
            continue
        with out.open("a") as fh:
            fh.write(json.dumps({"id": q["id"], "context": ctx}) + "\n")
    if missing:
        raise SystemExit(f"{missing} domande senza contesto: rilanciare retrieve prima di `run`")


def cmd_run(a) -> None:
    # Sensori e nvidia-smi sono locali: la prova ha senso (e il gate termico vale) solo se ollama gira qui.
    if urlparse(a.ollama_host).hostname not in LOOPBACK:
        raise SystemExit(f"ollama deve girare su questo host (telemetria locale): {a.ollama_host}")
    if len(set(a.models)) != len(a.models):
        raise SystemExit(f"candidati duplicati: {a.models}")
    scratch = check_scratch(Path(a.scratch))
    (scratch / "result.json").unlink(missing_ok=True)  # un esito vecchio non deve passare per quello di questa prova
    (scratch / "rounds.jsonl").unlink(missing_ok=True)
    qs = [json.loads(line) for line in (scratch / "questions.jsonl").read_text().splitlines()]
    meta = scratch / "retrieval.json"
    fingerprint = hashlib.sha1((scratch / "questions.jsonl").read_bytes()).hexdigest()
    if not meta.exists() or json.loads(meta.read_text()).get("questions_sha1") != fingerprint:
        raise SystemExit("i contesti non sono stati recuperati per queste domande: rilanciare retrieve")
    ctx = {(d := json.loads(line))["id"]: d["context"] for line in (scratch / "contexts.jsonl").read_text().splitlines()}
    missing = [q["id"] for q in qs if not ctx.get(q["id"], "").strip()]
    if missing:
        raise SystemExit(f"{len(missing)} domande senza contesto (rilanciare retrieve): {missing[:10]}")
    by_paper = defaultdict(list)
    for q in qs:
        if q["id"] in ctx:
            by_paper[q["paper"]].append(q)
    rounds = sorted(by_paper, key=lambda p: hashlib.sha1(p.encode()).hexdigest())
    scores = {m: (0, 0) for m in a.models}
    # Ammesso = OGNI risposta contata e' girata interamente in GPU (size_vram == size > 0), non solo i massimi.
    resident = dict.fromkeys(a.models, True)
    vram = dict.fromkeys(a.models, 0)
    latency: dict[str, list[float]] = {m: [] for m in a.models}
    guard = a.guard_cmd.split()
    stop_reason = "prova non completata"
    rounds_done = 0
    sensors = Sensors()

    def snapshot(complete: bool) -> dict:
        """Esito su disco (sostituzione atomica) dopo ogni round e in uscita: un processo ucciso lascia l'ultimo round intero.
        Il vincitore esiste solo se tutte le domande sono state valutate per tutti i candidati ammessi."""
        admitted = {m for m in a.models if rounds_done and resident[m] and vram[m]}
        decision = pick_winner(scores, vram, admitted) if complete else {
            "winner": None, "reason": f"PARZIALE: {stop_reason}", "questions_evaluated": scores[a.models[0]][1], "questions_total": sum(len(by_paper[p]) for p in rounds)}
        result = {
            "status": "COMPLETO" if complete else "PARZIALE", "stop_reason": stop_reason, "rounds_completed": rounds_done,
            "scores": scores, "method": {"prior": f"Beta({PRIOR_CORRECT}, {PRIOR_WRONG})", "mc_samples": MC_SAMPLES, "mc_seed": MC_SEED},
            "admitted": sorted(admitted), "vram": vram, "latency_mean_s": {m: sum(v) / len(v) for m, v in latency.items() if v},
            "peak_vram_mib": sensors.peak_vram_mib, "peak_gpu_c": sensors.peak_gpu_c, "peak_cpu_c": sensors.peak_cpu_c,
            "decision": decision,
        }
        tmp = scratch / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
        os.replace(tmp, scratch / "result.json")
        return result

    complete = finished = False
    try:
        with sensors:
            for paper in rounds:
                if not guard_ok(guard):
                    stop_reason = "dell_guard rifiuta"
                    break
                tally = {m: [0, 0, True, 0, []] for m in a.models}  # k, n, resident, vram, latencies
                try:
                    for model in a.models:
                        for q in by_paper[paper]:
                            reply, secs, ps = ollama_ask(a.ollama_host, model, ctx[q["id"]], q["question"], sensors.hot)
                            ok = is_correct(q, reply)
                            t = tally[model]
                            t[0] += ok
                            t[1] += 1
                            t[2] = t[2] and bool(ps["size"]) and ps["size_vram"] == ps["size"]
                            t[3] = max(t[3], ps["size_vram"] or 0)
                            t[4].append(secs)
                            log("answer", model=model, qid=q["id"], correct=ok, latency=round(secs, 2), **ps)
                except ThermalStop:
                    stop_reason = "soglia termica (round incompleto scartato)"
                    break
                except OllamaError as exc:
                    log("ollama_error", error=str(exc))
                    stop_reason = f"errore ollama (round incompleto scartato): {exc}"
                    break
                if sensors.hot.is_set():
                    stop_reason = "soglia termica"
                    break
                # Si fondono solo i round completi per tutti i candidati: nessun confronto su n diseguali.
                # Il round e' su disco prima del successivo: nessun errore posteriore lo cancella.
                append_durable(scratch / "rounds.jsonl", {"paper": paper, "tally": tally})
                rounds_done += 1
                for m, (k, n, res, v, lat) in tally.items():
                    scores[m] = (scores[m][0] + k, scores[m][1] + n)
                    resident[m] = resident[m] and res
                    vram[m] = max(vram[m], v)
                    latency[m] += lat
                stop_reason = f"in corso: round {rounds_done}/{len(rounds)} completato"
                snapshot(False)
            else:
                stop_reason, finished = "domande finite", rounds_done > 0
        complete = finished  # solo a sensori chiusi: un'interruzione durante la chiusura non lascia un vincitore
    except Exception as exc:  # difetto imprevisto: l'esito dei round completi resta scritto, poi l'errore risale
        stop_reason = f"eccezione non prevista dopo {rounds_done} round: {exc!r}"
        log("run_crashed", error=repr(exc), rounds_completed=rounds_done)
        raise
    except BaseException:  # interruzione dell'operatore (SIGINT/SIGTERM): esito parziale, nessun vincitore
        stop_reason = f"interrotta dopo {rounds_done} round"
        raise
    finally:
        result = snapshot(complete)
        log("result", **result["decision"], stop_reason=stop_reason)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scratch", default=str(SCRATCH_ROOT))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("questions")
    s.add_argument("--chunks", required=True, help="kv_store_text_chunks.json")
    s.set_defaults(fn=cmd_questions)
    s = sub.add_parser("retrieve")
    s.add_argument("--rag-url", default="http://127.0.0.1:8767")
    s.add_argument("--api-key-file", default=os.environ.get("RAG_API_KEY_FILE"))
    s.add_argument("--mode", default="hybrid")
    s.set_defaults(fn=cmd_retrieve)
    s = sub.add_parser("run")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    s.add_argument("--guard-cmd", required=True, help="comando dell_guard: exit 0 = via libera")
    s.set_defaults(fn=cmd_run)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
