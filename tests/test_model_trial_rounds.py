"""Prova modelli (F-775): ogni round completo e' durevole e l'esito sopravvive a un difetto imprevisto."""
from __future__ import annotations

import json

import pytest

from tests.test_model_trial_run import mt
from tests.test_model_trial_transport import write_scratch


def test_ogni_round_completo_e_su_disco_prima_del_successivo(tmp_path, monkeypatch):
    seen = []

    def ask(host, model, ctx, question):
        path = tmp_path / "rounds.jsonl"
        seen.append(len(path.read_text().splitlines()) if path.exists() else 0)
        return {"answer": "x", "paper": "p"}, 1.0, {"size": 10, "size_vram": 10}

    a = write_scratch(tmp_path, monkeypatch, 3)
    monkeypatch.setattr(mt, "ollama_ask", ask)
    mt.cmd_run(a)
    assert seen == [0, 1, 2]
    assert len((tmp_path / "rounds.jsonl").read_text().splitlines()) == 3


def test_eccezione_inattesa_lascia_comunque_il_risultato_dei_round_completi(tmp_path, monkeypatch):
    calls = []

    def ask(host, model, ctx, question):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("difetto imprevisto")
        return {"answer": "x", "paper": "p"}, 1.0, {"size": 10, "size_vram": 10}

    a = write_scratch(tmp_path, monkeypatch, 3)
    monkeypatch.setattr(mt, "ollama_ask", ask)
    with pytest.raises(RuntimeError):
        mt.cmd_run(a)
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["rounds_completed"] == 1 and "difetto imprevisto" in res["stop_reason"]
