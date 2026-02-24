"""Lightweight integration tests for raganything_service HTTP endpoints."""

from __future__ import annotations

import http.client
import importlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


svc = importlib.import_module("scripts.raganything_service")


class _AllowAllRateLimiter:
    def allow(self, _client_ip: str) -> tuple[bool, int]:
        return True, 0


class _StubCircuitBreaker:
    recovery_timeout = 300

    def __init__(self):
        self.reset_calls = 0

    def can_proceed(self) -> bool:
        return True

    def get_status(self) -> dict:
        return {"state": "closed", "recent_failures": 0}

    def reset(self):
        self.reset_calls += 1
        return None


@dataclass
class _SubmittedJob:
    paper_id: str
    pdf_path: str
    webhook_url: str | None
    resolved_webhook_ip: str | None
    force_parser: str | None
    force_reprocess: bool


class _StubJobQueue:
    def __init__(self):
        self.submissions: list[_SubmittedJob] = []
        self.job_history = []
        self.webhook_calls: list[object] = []

    def can_accept(self) -> bool:
        return True

    def submit_job(
        self,
        paper_id: str,
        pdf_path: str,
        webhook_url=None,
        resolved_webhook_ip=None,
        force_parser=None,
        force_reprocess=False,
    ):
        self.submissions.append(
            _SubmittedJob(
                paper_id=paper_id,
                pdf_path=pdf_path,
                webhook_url=webhook_url,
                resolved_webhook_ip=resolved_webhook_ip,
                force_parser=force_parser,
                force_reprocess=force_reprocess,
            )
        )
        return SimpleNamespace(
            job_id="job12345",
            status=SimpleNamespace(value="queued"),
        )

    def get_status(self) -> dict:
        return {
            "active_jobs": len(self.submissions),
            "max_workers": 1,
            "jobs": [],
            "completed_in_history": 0,
        }

    def get_job(self, _job_id: str):
        return None

    def _call_webhook(self, job):
        self.webhook_calls.append(job)


class _StubJobRecord:
    def __init__(self, job_id: str, status: str = "completed"):
        self.job_id = job_id
        self._status = status

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self._status,
            "result": {"success": True},
        }


def _http_json_request(port: int, method: str, path: str, payload=None, headers=None):
    req_headers = dict(headers or {})
    body = None

    if payload is not None:
        body = json.dumps(payload)
        req_headers.setdefault("Content-Type", "application/json")

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(method, path, body=body, headers=req_headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        try:
            data = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            data = raw
        return resp.status, data
    finally:
        conn.close()


@pytest.fixture
def rag_api_server(monkeypatch, tmp_path):
    pdf_root = tmp_path / "pdfs"
    output_root = tmp_path / "output"
    outside_root = tmp_path / "outside"
    pdf_root.mkdir()
    output_root.mkdir()
    outside_root.mkdir()

    stub_queue = _StubJobQueue()
    monkeypatch.setattr(svc, "job_queue", stub_queue)
    monkeypatch.setattr(svc, "circuit_breaker", _StubCircuitBreaker())
    monkeypatch.setattr(svc, "ip_rate_limiter", _AllowAllRateLimiter())
    monkeypatch.setattr(svc, "OUTPUT_BASE", str(output_root))
    monkeypatch.setattr(svc, "ALLOWED_PDF_ROOTS", (pdf_root.resolve(),))
    monkeypatch.setattr(svc, "ALLOW_UNSAFE_PDF_PATHS", False)
    monkeypatch.setattr(svc, "SERVICE_API_KEY", "test-api-key")
    monkeypatch.setattr(svc, "AUTH_EXEMPT_PATHS", ("/health", "/status"))
    monkeypatch.setattr(
        svc.RAGAnythingHandler,
        "_query_sync",
        lambda self, query, mode, context_only=False: {
            "success": True,
            "query": query,
            "mode": mode,
            "context_only": context_only,
            "answer": f"stub:{query}",
        },
    )

    server = svc.ReusableThreadingServer(("127.0.0.1", 0), svc.RAGAnythingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address
    assert host == "127.0.0.1"

    try:
        yield {
            "port": port,
            "pdf_root": pdf_root,
            "outside_root": outside_root,
            "job_queue": stub_queue,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_process_endpoint_submits_job_with_sanitized_pdf_path(rag_api_server):
    pdf = rag_api_server["pdf_root"] / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    status, data = _http_json_request(
        rag_api_server["port"],
        "POST",
        "/process",
        payload={
            "pdf_path": str(pdf),
            "paper_id": "arxiv:2401.12345",
        },
        headers={"X-API-Key": "test-api-key"},
    )

    assert status == 202
    assert data["success"] is True
    assert data["job_id"] == "job12345"
    submitted = rag_api_server["job_queue"].submissions[0]
    assert submitted.paper_id == "arxiv:2401.12345"
    assert submitted.pdf_path == str(pdf.resolve())


def test_process_endpoint_blocks_path_outside_allowlist(rag_api_server):
    outside_pdf = rag_api_server["outside_root"] / "secret.pdf"
    outside_pdf.write_bytes(b"%PDF-1.4\n")
    traversal = rag_api_server["pdf_root"] / ".." / "outside" / "secret.pdf"

    status, data = _http_json_request(
        rag_api_server["port"],
        "POST",
        "/process",
        payload={
            "pdf_path": str(traversal),
            "paper_id": "arxiv:2401.99999",
        },
        headers={"Authorization": "Bearer test-api-key"},
    )

    assert status == 403
    assert data["success"] is False
    assert "outside allowed" in data["error"]


def test_process_endpoint_rejects_invalid_parser(rag_api_server):
    pdf = rag_api_server["pdf_root"] / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    status, data = _http_json_request(
        rag_api_server["port"],
        "POST",
        "/process",
        payload={
            "pdf_path": str(pdf),
            "paper_id": "arxiv:2401.12345",
            "force_parser": "evil_parser",
        },
        headers={"X-API-Key": "test-api-key"},
    )

    assert status == 400
    assert "Invalid parser" in data["error"]


def test_process_cached_result_uses_webhook_helper_with_pinned_ip(rag_api_server, monkeypatch):
    monkeypatch.setattr(svc, "_resolve_webhook_ips", lambda host, port: {"93.184.216.34"})

    paper_id = "arxiv:2401/12:345?bad*id"
    safe_id = re.sub(r"[^a-zA-Z0-9._-]", "_", paper_id.replace("arxiv:", ""))
    cached_dir = rag_api_server["pdf_root"].parent / "output" / safe_id
    cached_dir.mkdir(parents=True, exist_ok=True)
    (cached_dir / "result.md").write_text("# cached")

    # PDF path is ignored for cached flow but must be syntactically present.
    pdf = rag_api_server["pdf_root"] / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    status, data = _http_json_request(
        rag_api_server["port"],
        "POST",
        "/process",
        payload={
            "pdf_path": str(pdf),
            "paper_id": paper_id,
            "webhook_url": "https://example.com/callback",
        },
        headers={"X-API-Key": "test-api-key"},
    )

    assert status == 200
    assert data["success"] is True
    assert data["cached"] is True
    assert "?" not in data["output_dir"]
    assert "*" not in data["output_dir"]
    assert len(rag_api_server["job_queue"].webhook_calls) == 1

    webhook_job = rag_api_server["job_queue"].webhook_calls[0]
    assert webhook_job.job_id == "cached"
    assert webhook_job.webhook_url == "https://example.com/callback"
    assert webhook_job.resolved_webhook_ip == "93.184.216.34"
    assert webhook_job.result["cached"] is True


def test_query_endpoint_requires_api_key_when_enabled(rag_api_server):
    status, data = _http_json_request(
        rag_api_server["port"],
        "POST",
        "/query",
        payload={"query": "kelly criterion", "mode": "hybrid"},
    )

    assert status == 401
    assert data["success"] is False
    assert data["error"] == "Unauthorized"


def test_query_endpoint_returns_stubbed_response_with_api_key(rag_api_server):
    status, data = _http_json_request(
        rag_api_server["port"],
        "POST",
        "/query",
        payload={"query": "kelly criterion", "mode": "hybrid"},
        headers={"X-API-Key": "test-api-key"},
    )

    assert status == 200
    assert data["success"] is True
    assert data["answer"] == "stub:kelly criterion"
    assert data["mode"] == "hybrid"


def test_health_endpoint_remains_public_when_api_key_enabled(rag_api_server):
    status, data = _http_json_request(rag_api_server["port"], "GET", "/health")

    assert status == 200
    assert data["status"] == "ok"
    assert "circuit_breaker" in data


def test_jobs_endpoint_returns_queue_status_with_api_key(rag_api_server):
    rag_api_server["job_queue"].get_status = lambda: {
        "active_jobs": 1,
        "max_workers": 2,
        "jobs": [{"job_id": "job12345", "status": "queued"}],
        "completed_in_history": 3,
    }

    status, data = _http_json_request(
        rag_api_server["port"],
        "GET",
        "/jobs",
        headers={"X-API-Key": "test-api-key"},
    )

    assert status == 200
    assert data["active_jobs"] == 1
    assert data["jobs"][0]["job_id"] == "job12345"


def test_job_detail_endpoint_returns_active_job(rag_api_server):
    job = _StubJobRecord("job12345", status="processing")
    rag_api_server["job_queue"].get_job = lambda job_id: job if job_id == "job12345" else None

    status, data = _http_json_request(
        rag_api_server["port"],
        "GET",
        "/jobs/job12345",
        headers={"Authorization": "Bearer test-api-key"},
    )

    assert status == 200
    assert data["job_id"] == "job12345"
    assert data["status"] == "processing"


def test_job_detail_endpoint_falls_back_to_history(rag_api_server):
    hist_job = _StubJobRecord("hist0001", status="completed")
    rag_api_server["job_queue"].job_history = [hist_job]

    status, data = _http_json_request(
        rag_api_server["port"],
        "GET",
        "/jobs/hist0001",
        headers={"X-API-Key": "test-api-key"},
    )

    assert status == 200
    assert data["job_id"] == "hist0001"
    assert data["status"] == "completed"


def test_reset_circuit_breaker_requires_api_key(rag_api_server):
    status, data = _http_json_request(
        rag_api_server["port"],
        "GET",
        "/reset-circuit-breaker",
    )

    assert status == 401
    assert data["success"] is False
    assert data["error"] == "Unauthorized"


def test_reset_circuit_breaker_resets_when_authorized(rag_api_server):
    breaker = svc.circuit_breaker
    assert breaker.reset_calls == 0

    status, data = _http_json_request(
        rag_api_server["port"],
        "GET",
        "/reset-circuit-breaker",
        headers={"X-API-Key": "test-api-key"},
    )

    assert status == 200
    assert data["success"] is True
    assert "reset" in data["message"].lower()
    assert breaker.reset_calls == 1


def test_rate_limiter_returns_429_when_exceeded(rag_api_server, monkeypatch):
    class _BlockAllRateLimiter:
        def allow(self, _client_ip: str) -> tuple[bool, int]:
            return False, 42

    monkeypatch.setattr(svc, "ip_rate_limiter", _BlockAllRateLimiter())

    status, data = _http_json_request(
        rag_api_server["port"],
        "POST",
        "/query",
        payload={"query": "test", "mode": "hybrid"},
        headers={"X-API-Key": "test-api-key"},
    )

    assert status == 429
    assert data["success"] is False
    assert "rate limit" in data["error"].lower()
    assert data["retry_after"] == 42


def test_circuit_breaker_returns_503_when_open(rag_api_server, monkeypatch):
    class _OpenCircuitBreaker:
        recovery_timeout = 120

        def can_proceed(self) -> bool:
            return False

        def get_status(self) -> dict:
            return {"state": "open", "recent_failures": 5}

        def reset(self):
            pass

    monkeypatch.setattr(svc, "circuit_breaker", _OpenCircuitBreaker())

    pdf = rag_api_server["pdf_root"] / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    status, data = _http_json_request(
        rag_api_server["port"],
        "POST",
        "/process",
        payload={"pdf_path": str(pdf), "paper_id": "arxiv:2401.00001"},
        headers={"X-API-Key": "test-api-key"},
    )

    assert status == 503
    assert data["success"] is False
    assert "circuit breaker" in data["error"].lower()
    assert data["retry_after"] == 120
