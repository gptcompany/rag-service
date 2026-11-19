"""Logica pura della prova modelli (F-775): domande, punteggio, arresto, scelta."""
from __future__ import annotations

import importlib
import json

import pytest

mt = importlib.import_module("scripts.model_trial")


def test_domande_generate_dal_testo_con_risposta_e_paper_di_origine():
    chunks = {"c1": {"file_path": "1411.3615v1.pdf", "content": (
        "Intro text. The accuracy reached 94.2% on the held-out set. "
        "We evaluate on the MNIST dataset today. Noise.")}}
    qs = mt.questions_from_chunks(chunks)
    by_kind = {q["kind"]: q for q in qs}
    assert by_kind["value"]["answer"] == "94.2%"
    assert by_kind["dataset"]["answer"] == "MNIST"
    assert all(q["paper"] == "1411.3615" and "____" in q["question"] for q in qs)
    assert "94.2%" not in by_kind["value"]["question"]


def test_domande_deterministiche_e_senza_duplicati():
    chunks = {"a": {"file_path": "x.pdf", "content": "The F1 score was 0.91 overall."},
              "b": {"file_path": "x.pdf", "content": "The F1 score was 0.91 overall."}}
    assert mt.questions_from_chunks(chunks) == mt.questions_from_chunks(chunks)
    assert len(mt.questions_from_chunks(chunks)) == 1


Q = {"answer": "94.2%", "paper": "1411.3615"}


def test_corretta_solo_con_risposta_e_paper_giusti():
    assert mt.is_correct(Q, {"answer": "94.2 %", "paper": "1411.3615v2.pdf"})
    assert mt.is_correct(Q, {"answer": "94.2%.", "paper": "1411.3615v2.pdf"})
    assert not mt.is_correct(Q, {"answer": "94.2%", "paper": "2001.00001.pdf"})
    assert not mt.is_correct(Q, {"answer": "90%", "paper": "1411.3615.pdf"})
    assert not mt.is_correct(Q, {"answer": "194.2%", "paper": "1411.3615.pdf"})
    assert not mt.is_correct(Q, {"answer": "94.2% and 95%", "paper": "1411.3615.pdf"})
    assert not mt.is_correct(Q, {"answer": "94.2%", "paper": "1411.36150.pdf"})
    assert not mt.is_correct(Q, {})


def test_scratch_fuori_da_1tb_scratch_rifiutato(tmp_path):
    with pytest.raises(SystemExit):
        mt.check_scratch(tmp_path)


def test_file_chiave_con_permessi_larghi_rifiutato(tmp_path):
    f = tmp_path / "k"
    f.write_text("secret")
    f.chmod(0o644)
    with pytest.raises(SystemExit):
        mt.read_secret_file(str(f))
    f.chmod(0o600)
    assert mt.read_secret_file(str(f)) == "secret"


def test_questions_cmd_scrive_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    chunks = tmp_path / "c.json"
    chunks.write_text(json.dumps({"c": {"file_path": "p.pdf", "content": "The F1 score was 0.91 overall."}}))
    mt.main(["--scratch", str(tmp_path / "s"), "questions", "--chunks", str(chunks)])
    assert json.loads((tmp_path / "s" / "questions.jsonl").read_text().splitlines()[0])["answer"] == "0.91"


def test_due_vuoti_nella_stessa_frase_hanno_id_distinti():
    chunks = {"a": {"file_path": "x.pdf", "content": "The accuracy on the MNIST dataset was 91%."}}
    assert {q["kind"] for q in mt.questions_from_chunks(chunks)} == {"value", "dataset"}


def test_retrieve_non_mette_in_cache_un_contesto_fallito(tmp_path, monkeypatch):
    import argparse

    import requests

    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    (tmp_path / "questions.jsonl").write_text(json.dumps({"id": "q1", "question": "?"}) + "\n")

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": False, "error": "Internal query error"}

    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    with pytest.raises(SystemExit, match="1 domande senza contesto"):
        mt.cmd_retrieve(argparse.Namespace(scratch=str(tmp_path), api_key_file=None, rag_url="u", mode="hybrid"))
    assert not (tmp_path / "contexts.jsonl").exists()


def test_numeri_con_segno_e_unita_non_si_confondono():
    q = {"answer": "0.5", "paper": "p1"}
    assert not mt.is_correct(q, {"answer": "-0.5", "paper": "p1"})
    assert not mt.is_correct({"answer": "94", "paper": "p1"}, {"answer": "94%", "paper": "p1"})
    assert mt.is_correct(q, {"answer": "0.5", "paper": "p1"})


def test_nome_file_con_spazi_citato_per_intero():
    assert mt.is_correct({"answer": "x", "paper": "A paper"}, {"answer": "x", "paper": "A paper.pdf"})


def test_frase_con_piu_numeri_non_genera_domanda_value():
    chunks = {"a": {"file_path": "x.pdf", "content": "In 2023 the model achieved 94.2% accuracy."}}
    assert mt.questions_from_chunks(chunks) == []


def test_retrieve_rifiuta_cache_con_impostazioni_diverse(tmp_path, monkeypatch):
    import argparse

    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    (tmp_path / "questions.jsonl").write_text("")
    (tmp_path / "contexts.jsonl").write_text("")
    (tmp_path / "retrieval.json").write_text(json.dumps({"rag_url": "a", "mode": "hybrid"}))
    with pytest.raises(SystemExit):
        mt.cmd_retrieve(argparse.Namespace(scratch=str(tmp_path), api_key_file=None, rag_url="b", mode="hybrid"))


def test_risposta_con_alternative_o_nome_piu_lungo_non_passa():
    q = {"answer": "MNIST", "paper": "p1"}
    assert not mt.is_correct(q, {"answer": "MNIST-M", "paper": "p1"})
    assert not mt.is_correct(q, {"answer": "MNIST or CIFAR", "paper": "p1"})
    assert mt.is_correct(q, {"answer": "mnist.", "paper": "p1"})


def test_numero_negativo_resta_intero_nella_domanda():
    chunks = {"a": {"file_path": "x.pdf", "content": "The Sharpe ratio was -0.5 overall."}}
    assert mt.questions_from_chunks(chunks)[0]["answer"] == "-0.5"


def test_meno_unicode_e_ascii_sono_lo_stesso_segno():
    assert mt.is_correct({"answer": "-0.5", "paper": "p"}, {"answer": "\u22120.5", "paper": "p"})
    assert not mt.is_correct({"answer": "0.5", "paper": "p"}, {"answer": "\u22120.5", "paper": "p"})
    assert mt.questions_from_chunks({"a": {"file_path": "x.pdf", "content": "The Sharpe ratio was \u22120.5 overall."}})[0]["answer"] == "\u22120.5"


def test_retrieve_scarta_contesto_null(tmp_path, monkeypatch):
    import argparse

    import requests

    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    (tmp_path / "questions.jsonl").write_text(json.dumps({"id": "q1", "question": "?"}) + "\n")

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": True, "context": None}

    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    with pytest.raises(SystemExit, match="1 domande senza contesto"):
        mt.cmd_retrieve(argparse.Namespace(scratch=str(tmp_path), api_key_file=None, rag_url="u", mode="hybrid"))
    assert not (tmp_path / "contexts.jsonl").exists()


def test_cache_legata_all_insieme_di_domande(tmp_path, monkeypatch):
    import argparse

    import requests

    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": True, "context": "ctx"}

    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    args = argparse.Namespace(scratch=str(tmp_path), api_key_file=None, rag_url="u", mode="hybrid")
    (tmp_path / "questions.jsonl").write_text(json.dumps({"id": "q1", "question": "?"}) + "\n")
    mt.cmd_retrieve(args)
    (tmp_path / "questions.jsonl").write_text(json.dumps({"id": "q2", "question": "??"}) + "\n")
    with pytest.raises(SystemExit, match="altre impostazioni"):
        mt.cmd_retrieve(args)


def test_citazione_con_altro_paper_che_inizia_uguale_non_passa():
    q = {"answer": "x", "paper": "Smith"}
    assert not mt.is_correct(q, {"answer": "x", "paper": "Smith et al.pdf"})
    assert mt.is_correct(q, {"answer": "x", "paper": ["Smith.pdf", "Other.pdf"]})
    assert not mt.is_correct(q, {"answer": "x", "paper": "Smith.pdf, Other.pdf"})  # una stringa e' una sola citazione


def test_prompt_identico_con_chiavi_diverse_viene_scartato():
    chunks = {"a": {"file_path": "p1.pdf", "content": "The accuracy was 94%."},
              "b": {"file_path": "p2.pdf", "content": "The accuracy was 88%."},
              "c": {"file_path": "p3.pdf", "content": "The F1 score was 0.5 overall."}}
    qs = mt.questions_from_chunks(chunks)
    assert [q["paper"] for q in qs] == ["p3"]


def test_numero_di_figura_non_e_il_valore_della_metrica():
    chunks = {"a": {"file_path": "p.pdf", "content": "The accuracy is shown in Figure 2."}}
    assert mt.questions_from_chunks(chunks) == []


def test_valore_non_svuotato_dentro_un_numero_piu_lungo():
    chunks = {"a": {"file_path": "p.pdf", "content": "The accuracy was 194.2% versus 94 in the paper."}}
    assert [q for q in mt.questions_from_chunks(chunks) if q["kind"] == "value"] == []


def test_retrieve_non_segue_i_redirect_con_la_chiave(tmp_path, monkeypatch):
    import argparse

    import requests

    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    (tmp_path / "questions.jsonl").write_text(json.dumps({"id": "q1", "question": "?"}) + "\n")
    (tmp_path / "key").write_text("segreto")
    (tmp_path / "key").chmod(0o600)
    seen = {}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": True, "context": "c"}

    def post(*a, **k):
        seen.update(k)
        return R()

    monkeypatch.setattr(requests, "post", post)
    mt.cmd_retrieve(argparse.Namespace(scratch=str(tmp_path), api_key_file=str(tmp_path / "key"), rag_url="u", mode="m"))
    assert seen["allow_redirects"] is False


def test_retrieve_senza_domande_fallisce(tmp_path, monkeypatch):
    import argparse

    monkeypatch.setattr(mt, "SCRATCH_ROOT", tmp_path)
    (tmp_path / "questions.jsonl").write_text("")
    with pytest.raises(SystemExit, match="nessuna domanda"):
        mt.cmd_retrieve(argparse.Namespace(scratch=str(tmp_path), api_key_file=None, rag_url="u", mode="m"))


@pytest.mark.parametrize("text", ["Fig. 2 shows the accuracy.", "Eq. 3 defines the loss.", "See Sec. 4 for the F1 score."])
def test_riferimento_abbreviato_non_e_il_valore_della_metrica(text):
    assert mt.questions_from_chunks({"a": {"file_path": "p.pdf", "content": text}}) == []


@pytest.mark.parametrize("minus", ["\u2013", "\u2011", "\u2010", "\u2012", "\uff0d"])
def test_ogni_meno_tipografico_resta_nel_valore_e_nel_punteggio(minus):
    qs = mt.questions_from_chunks({"a": {"file_path": "p.pdf", "content": f"The Sharpe ratio was {minus}0.5 overall."}})
    assert [q["answer"] for q in qs] == [f"{minus}0.5"]
    assert minus not in qs[0]["question"] and "0.5" not in qs[0]["question"]
    assert not mt.is_correct({"answer": "0.5", "paper": "p"}, {"answer": f"{minus}0.5", "paper": "p"})
    assert mt.is_correct({"answer": "-0.5", "paper": "p"}, {"answer": f"{minus}0.5", "paper": "p"})


def test_segmento_di_directory_non_vale_come_citazione():
    q = {"answer": "x", "paper": "Smith"}
    assert not mt.is_correct(q, {"answer": "x", "paper": "/archive/Smith/Another paper.pdf"})
    assert mt.is_correct(q, {"answer": "x", "paper": "/archive/Smith/Smith.pdf"})
    assert mt.is_correct(q, {"answer": "x", "paper": ["Other.pdf", "/archive/Smith.pdf"]})
    assert not mt.is_correct(q, {"answer": "x", "paper": ["/archive/Smith/Another paper.pdf"]})


def test_nomi_di_paper_diversi_per_punteggiatura_non_si_confondono():
    q = {"answer": "x", "paper": "Study 1"}
    assert not mt.is_correct(q, {"answer": "x", "paper": "Study_1.pdf"})
    assert mt.is_correct(q, {"answer": "x", "paper": "study 1.PDF"})


def test_nome_file_con_virgola_e_una_sola_citazione():
    assert mt.is_correct({"answer": "x", "paper": "Smith, Jr"}, {"answer": "x", "paper": "Smith, Jr.pdf"})
    assert not mt.is_correct({"answer": "x", "paper": "Jr"}, {"answer": "x", "paper": "Smith, Jr.pdf"})
    assert not mt.is_correct({"answer": "x", "paper": "Smith"}, {"answer": "x", "paper": "Smith, A Study.pdf"})
