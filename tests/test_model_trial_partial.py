"""Prova modelli (F-775, decisione A): un esito parziale non dichiara alcun vincitore.

Il vincitore esiste solo se tutte le domande sono state valutate per tutti i candidati ammessi; arresto termico, errore,
guardia, interruzione o difetto imprevisto lasciano un report PARZIALE (i round completi restano su disco).
"""
from __future__ import annotations

import json

import pytest

from tests.test_model_trial_run import mt
from tests.test_model_trial_transport import write_scratch

RIGHT = ({"answer": "x", "paper": ["p0", "p1", "p2"]}, 1.0, {"size": 10, "size_vram": 10})


def run_with(tmp_path, monkeypatch, ask, n=3, expect=None):
    a = write_scratch(tmp_path, monkeypatch, n)
    monkeypatch.setattr(mt, "ollama_ask", ask)
    if expect:
        with pytest.raises(expect):
            mt.cmd_run(a)
    else:
        mt.cmd_run(a)
    return json.loads((tmp_path / "result.json").read_text())


def fail_at(call, exc):
    calls = []

    def ask(*args):
        calls.append(1)
        if len(calls) == call:
            raise exc
        return RIGHT

    return ask


@pytest.mark.parametrize("exc, expect", [
    (mt.ThermalStop("a"), None), (mt.OllamaError("boom"), None), (RuntimeError("difetto"), RuntimeError), (KeyboardInterrupt(), KeyboardInterrupt),
], ids=["arresto_termico", "errore_ollama", "difetto_imprevisto", "interruzione"])
def test_round_parziali_non_dichiarano_alcun_vincitore(tmp_path, monkeypatch, exc, expect):
    res = run_with(tmp_path, monkeypatch, fail_at(2, exc), expect=expect)
    # Un round completo e un candidato interamente in GPU: i dati ci sono, ma le domande non sono finite.
    assert res["rounds_completed"] == 1 and res["admitted"] == ["a"]
    assert res["status"] == "PARZIALE" and res["decision"]["winner"] is None
    assert res["decision"]["questions_evaluated"] == 1 and res["decision"]["questions_total"] == 3


def test_guardia_che_rifiuta_a_meta_prova_e_un_esito_parziale(tmp_path, monkeypatch):
    answers = iter([True, False])
    monkeypatch.setattr(mt, "guard_ok", lambda cmd: next(answers))
    res = run_with(tmp_path, monkeypatch, lambda *a: RIGHT)
    assert res["stop_reason"] == "dell_guard rifiuta" and res["status"] == "PARZIALE" and res["decision"]["winner"] is None


def test_con_tutte_le_domande_valutate_il_vincitore_esiste(tmp_path, monkeypatch):
    res = run_with(tmp_path, monkeypatch, lambda *a: RIGHT)
    assert res["status"] == "COMPLETO" and res["rounds_completed"] == 3 and res["decision"]["winner"] == "a"


def test_l_esito_parziale_e_su_disco_dopo_ogni_round_anche_se_il_processo_muore(tmp_path, monkeypatch):
    """Un SIGKILL non lascia il tempo di scrivere nulla: l'esito del round k deve esserci gia' prima del round k+1."""
    seen = []

    def ask(*args):
        path = tmp_path / "result.json"
        seen.append(json.loads(path.read_text()) if path.exists() else None)
        return RIGHT

    run_with(tmp_path, monkeypatch, ask)
    assert seen[0] is None or seen[0]["rounds_completed"] == 0
    assert [(s["status"], s["rounds_completed"], s["decision"]["winner"]) for s in seen[1:]] == [("PARZIALE", 1, None), ("PARZIALE", 2, None)]


def test_il_parziale_conta_domande_e_non_paper(tmp_path, monkeypatch):
    """Due domande nel primo paper (completo) e una nel secondo (interrotto): 2 domande valutate su 3, non 1 su 2."""
    import hashlib

    a = write_scratch(tmp_path, monkeypatch, 3)
    qs = [json.loads(line) for line in (tmp_path / "questions.jsonl").read_text().splitlines()]
    first = min(["pA", "pB"], key=lambda p: hashlib.sha1(p.encode()).hexdigest())  # il paper che il sorteggio fa girare per primo
    second = "pB" if first == "pA" else "pA"
    for q, paper in zip(qs, [first, first, second]):
        q["paper"] = paper
    (tmp_path / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    (tmp_path / "retrieval.json").write_text(json.dumps({"questions_sha1": hashlib.sha1((tmp_path / "questions.jsonl").read_bytes()).hexdigest()}))
    monkeypatch.setattr(mt, "ollama_ask", fail_at(3, mt.OllamaError("boom")))
    mt.cmd_run(a)
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["rounds_completed"] == 1
    assert res["decision"]["questions_evaluated"] == 2 and res["decision"]["questions_total"] == 3


def test_interruzione_alla_chiusura_dei_sensori_non_lascia_un_vincitore(tmp_path, monkeypatch):
    """Ctrl-C mentre il contesto dei sensori si chiude: le domande erano finite, ma la prova non e' conclusa."""
    from tests.test_model_trial_run import FakeSensors

    class Closing(FakeSensors):
        def __exit__(self, *exc):
            raise KeyboardInterrupt

    a = write_scratch(tmp_path, monkeypatch, 2)
    monkeypatch.setattr(mt, "Sensors", Closing)
    monkeypatch.setattr(mt, "ollama_ask", lambda *args: RIGHT)
    with pytest.raises(KeyboardInterrupt):
        mt.cmd_run(a)
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["status"] == "PARZIALE" and res["decision"]["winner"] is None
