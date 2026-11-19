#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""Prova dei modelli di generazione su ollama (F-775, fase 2: NON eseguire senza il SI' dell'operatore).

Sottocomandi di questa parte (generazione e punteggio delle domande):
  questions  genera le domande dal testo dei chunk gia' indicizzati (cloze, nessuna domanda a mano)
  retrieve   recupera UNA volta il contesto per ogni domanda (POST /query context_only) e lo salva

Lo scratch deve stare sotto /media/sam/1TB/scratch/f775/ (mai /tmp). La chiave API si legge da un
file 0600 (RAG_API_KEY_FILE) e non viene mai stampata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

SCRATCH_ROOT = Path("/media/sam/1TB/scratch/f775")


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
    # Ogni citazione intera si riduce al suo nome file prima del confronto: un segmento di directory non e' un paper.
    items = raw if isinstance(raw, (list, tuple)) else re.split(r"\s*[,;]\s*", str(raw))
    cited = {normalize(paper_key(t)) for t in items if str(t).strip()}
    return bool(expected) and normalize(reply.get("answer", "")) == expected and normalize(question["paper"]) in cited


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
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
