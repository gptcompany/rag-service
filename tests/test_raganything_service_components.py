"""Tests for raganything_service.py components."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

import pytest
from scripts.raganything_service import CircuitBreaker, IpRateLimiter, RAGAnythingHandler, Job, JobStatus


def test_job_model():
    job = Job(job_id="1", paper_id="p1", pdf_path="path")
    assert job.status == JobStatus.QUEUED
    d = job.to_dict()
    assert d["job_id"] == "1"
    assert d["status"] == "queued"


def test_circuit_breaker_closed_to_open():
    cb = CircuitBreaker(failure_threshold=2)
    assert cb.can_proceed() is True
    
    cb.record_failure()
    assert cb.can_proceed() is True # Still closed or threshold not reached
    
    cb.record_failure()
    assert cb.state == "open"
    assert cb.can_proceed() is False


def test_handler_path_utils():
    # Mocking BaseHTTPRequestHandler is tricky, let's just mock the instance
    handler = MagicMock(spec=RAGAnythingHandler)
    handler.path = "/test?query=1"
    assert RAGAnythingHandler._path_without_query(handler) == "/test"
    
    handler.path = "/health"
    assert RAGAnythingHandler._path_without_query(handler) == "/health"


def test_handler_requires_auth():
    handler = MagicMock(spec=RAGAnythingHandler)
    
    with patch("scripts.raganything_service.SERVICE_API_KEY", "test-key"):
        # /health is exempt by default usually
        with patch("scripts.raganything_service.AUTH_EXEMPT_PATHS", ["/health"]):
            assert RAGAnythingHandler._requires_auth(handler, "/health") is False
            assert RAGAnythingHandler._requires_auth(handler, "/process") is True

    with patch("scripts.raganything_service.SERVICE_API_KEY", None):
        assert RAGAnythingHandler._requires_auth(handler, "/process") is False


def test_circuit_breaker_recovery():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
    cb.record_failure()
    assert cb.state == "open"
    assert cb.can_proceed() is False
    
    time.sleep(0.15)
    # can_proceed should transition to half-open
    assert cb.can_proceed() is True
    assert cb.state == "half-open"


def test_circuit_breaker_reset():
    cb = CircuitBreaker(failure_threshold=1)
    cb.record_failure()
    assert cb.state == "open"
    cb.reset()
    assert cb.state == "closed"
    assert cb.can_proceed() is True


def test_ip_rate_limiter_allow():
    limiter = IpRateLimiter(max_requests=2, window_sec=1)
    
    allowed, retry = limiter.allow("1.1.1.1")
    assert allowed is True
    
    allowed, retry = limiter.allow("1.1.1.1")
    assert allowed is True
    
    allowed, retry = limiter.allow("1.1.1.1")
    assert allowed is False
    assert retry > 0
    
    # Different IP should be allowed
    allowed, retry = limiter.allow("2.2.2.2")
    assert allowed is True


def test_ip_rate_limiter_window_slide():
    limiter = IpRateLimiter(max_requests=1, window_sec=0.1)
    limiter.allow("1.1.1.1")
    allowed, _ = limiter.allow("1.1.1.1")
    assert allowed is False
    
    time.sleep(0.15)
    allowed, _ = limiter.allow("1.1.1.1")
    assert allowed is True
