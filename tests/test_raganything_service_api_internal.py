"""Tests for RAGAnythingHandler internal methods and logic."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from scripts.raganything_service import RAGAnythingHandler


@pytest.fixture
def handler():
    # Mocking BaseHTTPRequestHandler needs more than just spec
    h = MagicMock(spec=RAGAnythingHandler)
    h.rfile = MagicMock()
    h.wfile = MagicMock()
    h.headers = {}
    return h


def test_handle_process_circuit_breaker_open(handler):
    with patch("scripts.raganything_service.circuit_breaker") as mock_cb:
        mock_cb.can_proceed.return_value = False
        mock_cb.recovery_timeout = 300
        
        RAGAnythingHandler.handle_process(handler)
        
        # Verify send_json was called with 503
        handler.send_json.assert_called_once()
        args = handler.send_json.call_args[0]
        assert args[0] == 503
        assert "circuit breaker open" in args[1]["error"]


def test_handle_process_queue_full(handler):
    with patch("scripts.raganything_service.circuit_breaker.can_proceed", return_value=True), \
         patch("scripts.raganything_service.job_queue.can_accept", return_value=False):
        
        RAGAnythingHandler.handle_process(handler)
        
        handler.send_json.assert_called_once()
        args = handler.send_json.call_args[0]
        assert args[0] == 429
        assert "Too many jobs" in args[1]["error"]


def test_handle_process_missing_params(handler):
    with patch("scripts.raganything_service.circuit_breaker.can_proceed", return_value=True), \
         patch("scripts.raganything_service.job_queue.can_accept", return_value=True):
        
        handler.read_json_body.return_value = {}
        RAGAnythingHandler.handle_process(handler)
        
        handler.send_json.assert_called_once()
        args = handler.send_json.call_args[0]
        assert args[0] == 400
        assert "Missing pdf_path" in args[1]["error"]


def test_handle_query_missing_query(handler):
    handler.read_json_body.return_value = {}
    RAGAnythingHandler.handle_query(handler)
    
    handler.send_json.assert_called_once()
    assert handler.send_json.call_args[0][0] == 400


def test_handle_query_success(handler):
    handler.read_json_body.return_value = {"query": "test query"}
    handler._query_sync.return_value = {"answer": "ok"}
    RAGAnythingHandler.handle_query(handler)
    handler.send_json.assert_called_once_with(200, {"answer": "ok"})


def test_handle_process_sync_deprecated(handler):
    RAGAnythingHandler.handle_process_sync(handler)
    handler.send_json.assert_called_once()
    assert handler.send_json.call_args[0][0] == 400
    assert "deprecated" in handler.send_json.call_args[0][1]["error"]


def test_do_get_health(handler):
    handler.path = "/health"
    with patch("scripts.raganything_service.rag_instance", MagicMock()), \
         patch("scripts.raganything_service.get_hash_store") as mock_hash, \
         patch("scripts.raganything_service.job_queue.get_status", return_value={}), \
         patch("scripts.raganything_service.circuit_breaker.get_status", return_value={}):
        
        mock_hash.return_value.get_stats.return_value = {}
        RAGAnythingHandler.do_GET(handler)
        handler.send_json.assert_called_once()
        assert handler.send_json.call_args[0][0] == 200


def test_do_get_status(handler):
    handler.path = "/status"
    with patch("scripts.raganything_service.circuit_breaker.get_status", return_value={"state": "closed"}), \
         patch("scripts.raganything_service.job_queue.get_status", return_value={}):
        RAGAnythingHandler.do_GET(handler)
        handler.send_json.assert_called_once()
        assert handler.send_json.call_args[0][0] == 200


def test_do_get_job_found(handler):
    handler.path = "/jobs/123"
    mock_job = MagicMock()
    mock_job.to_dict.return_value = {"id": "123"}
    with patch("scripts.raganything_service.job_queue.get_job", return_value=mock_job):
        RAGAnythingHandler.do_GET(handler)
        handler.send_json.assert_called_once_with(200, {"id": "123"})


def test_do_get_job_not_found(handler):
    handler.path = "/jobs/999"
    with patch("scripts.raganything_service.job_queue.get_job", return_value=None), \
         patch("scripts.raganything_service.job_queue.job_history", []):
        RAGAnythingHandler.do_GET(handler)
        handler.send_json.assert_called_once()
        assert handler.send_json.call_args[0][0] == 404


def test_do_get_reset_circuit_breaker(handler):
    handler.path = "/reset-circuit-breaker"
    with patch("scripts.raganything_service.circuit_breaker.reset") as mock_reset:
        RAGAnythingHandler.do_GET(handler)
        mock_reset.assert_called_once()
        handler.send_json.assert_called_once()
        assert handler.send_json.call_args[0][0] == 200
