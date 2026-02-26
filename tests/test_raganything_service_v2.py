"""Deep coverage tests for raganything_service components."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

import pytest
from scripts.raganything_service import (
    PDFHashStore, 
    AsyncJobQueue, 
    Job, 
    JobStatus,
    RAGAnythingHandler,
    sanitize_webhook_url,
    sanitize_pdf_path
)

# ── PDFHashStore ──────────────────────────────────────────────

def test_hash_store_lifecycle(tmp_path):
    hash_file = tmp_path / "hashes.json"
    store = PDFHashStore(db_path=str(hash_file))
    assert store.hashes == {}
    
    # Mark processed
    store.mark_processed("hash1", "paper1", "/out/1", "mineru")
    assert "hash1" in store.hashes
    assert store.hashes["hash1"]["paper_id"] == "paper1"
    
    # Check persistence
    store2 = PDFHashStore(db_path=str(hash_file))
    assert "hash1" in store2.hashes
    
    # Stats
    stats = store.get_stats()
    assert stats["total_processed"] == 1
    assert stats["by_parser"]["mineru"] == 1

def test_hash_store_calculate_hash(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"pdf content")
    store = PDFHashStore()
    h = store.get_pdf_hash(str(pdf))
    assert len(h) == 64 # SHA-256
    
    # Same content, same hash
    assert h == store.get_pdf_hash(str(pdf))

# ── AsyncJobQueue ─────────────────────────────────────────────

def test_job_queue_acceptance():
    with patch("scripts.raganything_service.MAX_CONCURRENT_JOBS", 1), \
         patch("scripts.raganything_service.MAX_QUEUE_DEPTH", 1):
        queue = AsyncJobQueue(max_workers=1)
        # 1 worker + 1 queue = 2 capacity
        assert queue.can_accept() is True
        
        # Submit 2 jobs
        j1 = queue.submit_job("p1", "path1")
        j2 = queue.submit_job("p2", "path2")
        
        # Now full (assuming they are still processing/queued)
        with patch.object(queue, "get_active_count", return_value=2):
            assert queue.can_accept() is False

@patch("scripts.raganything_service.run_in_shared_loop")
def test_job_queue_process_flow(mock_run_loop):
    # Mock loop to return success
    mock_run_loop.return_value = {"success": True, "output_dir": "/tmp/out"}
    
    queue = AsyncJobQueue(max_workers=1)
    with patch("scripts.raganything_service.circuit_breaker") as mock_cb:
        job = queue.submit_job("p1", "path1")
        
        # Wait a bit for the thread pool to pick it up and complete
        # Since we mock _do_process via run_in_shared_loop, it should be fast
        timeout = time.time() + 2
        while time.time() < timeout:
            if job.status == JobStatus.COMPLETED:
                break
            time.sleep(0.1)
            
        assert job.status == JobStatus.COMPLETED
        mock_cb.record_success.assert_called()

# ── Sanitizers ────────────────────────────────────────────────

def test_sanitize_webhook_url_validation():
    # Long URL
    with pytest.raises(ValueError, match="too long"):
        sanitize_webhook_url("http://" + "a" * 2100)
    
    # Bad scheme
    with pytest.raises(ValueError, match="must use http or https"):
        sanitize_webhook_url("ftp://example.com")
        
    # Credentials
    with pytest.raises(ValueError, match="must not include credentials"):
        sanitize_webhook_url("http://user:pass@example.com")

def test_sanitize_pdf_path_validation():
    with pytest.raises(ValueError, match="must be a string"):
        sanitize_pdf_path(123)
    
    with pytest.raises(ValueError, match="empty"):
        sanitize_pdf_path("  ")
        
    with pytest.raises(ValueError, match="invalid characters"):
        sanitize_pdf_path("test\0pdf")

@pytest.fixture
def handler():
    h = MagicMock(spec=RAGAnythingHandler)
    h.rfile = MagicMock()
    h.wfile = MagicMock()
    h.headers = {}
    h.read_json_body = MagicMock()
    return h

# ── API Handlers Deep Dive ────────────────────────────────────

def test_handle_process_cached(handler):
    # Simulate directory already exists with markdown files
    handler.read_json_body.return_value = {
        "paper_id": "p1",
        "pdf_path": "/tmp/test.pdf"
    }
    with patch("os.path.exists", return_value=True), \
         patch("pathlib.Path.rglob") as mock_glob:
        mock_file = MagicMock()
        mock_file.stat.return_value.st_size = 100
        mock_glob.return_value = [mock_file]
        
        from scripts.raganything_service import RAGAnythingHandler
        RAGAnythingHandler.handle_process(handler)
        
        handler.send_json.assert_called_once()
        args = handler.send_json.call_args[0]
        assert args[0] == 200
        assert args[1]["cached"] is True

def test_handle_query_modes(handler):
    handler.read_json_body.return_value = {"query": "q", "mode": "global"}
    handler._query_sync.return_value = {"a": "1"}
    from scripts.raganything_service import RAGAnythingHandler
    RAGAnythingHandler.handle_query(handler)
    handler.send_json.assert_called_with(200, {"a": "1"})

def test_server_classes():
    from scripts.raganything_service import ReusableThreadingServer, RAGAnythingHandler
    # Just check they can be instantiated/configured
    server = ReusableThreadingServer(("127.0.0.1", 0), RAGAnythingHandler)
    assert server.allow_reuse_address is True
    server.server_close()

def test_shared_loop():
    from scripts.raganything_service import get_shared_loop
    loop = get_shared_loop()
    assert loop.is_running()

def test_initialize_rag_startup():
    with patch("scripts.raganything_service.get_rag_instance") as mock_get_rag:
        mock_rag = MagicMock()
        mock_rag._ensure_lightrag_initialized.return_value = {"success": True}
        mock_get_rag.return_value = mock_rag
        
        from scripts.raganything_service import initialize_rag_at_startup
        initialize_rag_at_startup()
        mock_rag._ensure_lightrag_initialized.assert_called()
