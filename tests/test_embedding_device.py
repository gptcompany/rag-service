"""RAG_EMBEDDING_DEVICE: il device di bge-large e' configurabile (F-775)."""
from __future__ import annotations

import importlib
from unittest.mock import MagicMock

svc = importlib.import_module("scripts.raganything_service")


def test_device_unset_lascia_la_scelta_automatica_a_sentence_transformers(monkeypatch):
    monkeypatch.delenv("RAG_EMBEDDING_DEVICE", raising=False)
    assert svc._resolve_embedding_device() is None


def test_device_vuoto_e_trattato_come_non_impostato(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_DEVICE", "  ")
    assert svc._resolve_embedding_device() is None


def test_device_cpu_viene_passato_al_modello(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_DEVICE", "cpu")
    cls = MagicMock()
    svc._load_embedding_model(cls, "BAAI/bge-large-en-v1.5")
    cls.assert_called_once_with("BAAI/bge-large-en-v1.5", device="cpu")


def test_senza_device_il_modello_e_caricato_con_device_none(monkeypatch):
    monkeypatch.delenv("RAG_EMBEDDING_DEVICE", raising=False)
    cls = MagicMock()
    svc._load_embedding_model(cls, "BAAI/bge-large-en-v1.5")
    cls.assert_called_once_with("BAAI/bge-large-en-v1.5", device=None)
