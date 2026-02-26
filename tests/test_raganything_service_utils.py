"""Tests for raganything_service.py utility functions."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest
from scripts.raganything_service import (
    _read_text_file,
    _detect_cgroup_cpu_limit,
    _detect_effective_cpu_capacity,
    _parse_optional_int_env,
    _parse_optional_bool_env,
    _auto_max_concurrent_jobs,
    _auto_queue_depth,
    _auto_cpu_threads,
    _resolve_runtime_queue_tuning,
    _apply_runtime_cpu_thread_tuning,
)


def test_read_text_file_ok(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("  hello  \n")
    assert _read_text_file(str(f)) == "hello"


def test_read_text_file_error():
    assert _read_text_file("/nonexistent/path/at/all") is None


def test_detect_cgroup_v2_limit():
    with patch("scripts.raganything_service._read_text_file") as mock_read:
        # 100000 100000 -> 1 CPU
        mock_read.side_effect = ["100000 100000", None, None]
        limit, source = _detect_cgroup_cpu_limit()
        assert limit == 1
        assert source == "cgroup-v2"


def test_detect_cgroup_v1_limit():
    with patch("scripts.raganything_service._read_text_file") as mock_read:
        # v2 fails, v1 succeeds
        mock_read.side_effect = [None, "200000", "100000"]
        limit, source = _detect_cgroup_cpu_limit()
        assert limit == 2
        assert source == "cgroup-v1"


def test_detect_cgroup_no_limit():
    with patch("scripts.raganything_service._read_text_file", return_value=None):
        limit, source = _detect_cgroup_cpu_limit()
        assert limit is None
        assert source == "none"


def test_detect_effective_cpu_capacity():
    with patch("os.cpu_count", return_value=8), \
         patch("os.sched_getaffinity", return_value={0, 1, 2, 3}), \
         patch("scripts.raganything_service._detect_cgroup_cpu_limit", return_value=(2, "cgroup-v2")):
        cap = _detect_effective_cpu_capacity()
        assert cap["effective_cpu_count"] == 2
        assert "affinity" in cap["effective_cpu_source"]
        assert "cgroup-v2" in cap["effective_cpu_source"]


def test_parse_optional_int_env():
    with patch("os.getenv", return_value="10"):
        assert _parse_optional_int_env("SOME_VAR", min_value=1) == 10
    with patch("os.getenv", return_value="0"):
        assert _parse_optional_int_env("SOME_VAR", min_value=1) is None
    with patch("os.getenv", return_value="abc"):
        assert _parse_optional_int_env("SOME_VAR", min_value=1) is None
    with patch("os.getenv", return_value=None):
        assert _parse_optional_int_env("SOME_VAR", min_value=1) is None


def test_parse_optional_bool_env():
    with patch("os.getenv", return_value="true"):
        assert _parse_optional_bool_env("VAR", default=False) is True
    with patch("os.getenv", return_value="0"):
        assert _parse_optional_bool_env("VAR", default=True) is False
    with patch("os.getenv", return_value="invalid"):
        assert _parse_optional_bool_env("VAR", default=True) is True
    with patch("os.getenv", return_value=None):
        assert _parse_optional_bool_env("VAR", default=False) is False


def test_auto_max_concurrent_jobs():
    assert _auto_max_concurrent_jobs(4) == 1
    assert _auto_max_concurrent_jobs(64) == 2
    assert _auto_max_concurrent_jobs(256) == 4


def test_auto_queue_depth():
    assert _auto_queue_depth(1) == 4
    assert _auto_queue_depth(4) == 16


def test_auto_cpu_threads():
    assert _auto_cpu_threads(8, 1) == 8
    assert _auto_cpu_threads(8, 4) == 2


from scripts.raganything_service import sanitize_webhook_url, sanitize_pdf_path

def test_sanitize_webhook_url():
    with patch("scripts.raganything_service.ALLOW_PRIVATE_WEBHOOK_HOSTS", True):
        url, ip = sanitize_webhook_url("http://localhost:8080/callback")
        assert url == "http://localhost:8080/callback"

    url, ip = sanitize_webhook_url(None)
    assert url is None
    assert ip is None


def test_sanitize_pdf_path():
    with patch("pathlib.Path.resolve") as mock_resolve, \
         patch("pathlib.Path.is_absolute", return_value=True), \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("os.access", return_value=True), \
         patch("scripts.raganything_service.ALLOW_UNSAFE_PDF_PATHS", True):
        mock_resolve.return_value = Path("/tmp/test.pdf")
        assert sanitize_pdf_path("/tmp/test.pdf") == "/tmp/test.pdf"

    with pytest.raises(ValueError):
        sanitize_pdf_path("http://malicious.com")


def test_resolve_runtime_queue_tuning():
    with patch("os.getenv", return_value="2"), \
         patch("scripts.raganything_service._detect_effective_cpu_capacity", return_value={"effective_cpu_count": 8, "host_cpu_count": 8, "affinity_cpu_count": 8, "cgroup_cpu_limit": 8, "effective_cpu_source": "test"}):
        tuning = _resolve_runtime_queue_tuning()
        assert tuning["max_concurrent_jobs"] == 2
        assert tuning["max_concurrent_jobs_source"] == "env:RAG_MAX_CONCURRENT_JOBS"


def test_apply_runtime_cpu_thread_tuning_disabled():
    with patch("os.getenv", return_value="false"):
        result = _apply_runtime_cpu_thread_tuning({"effective_cpu_count": 8, "max_concurrent_jobs": 1})
        assert result["enabled"] is False


def test_apply_runtime_cpu_thread_tuning_enabled():
    with patch("os.environ", {}), \
         patch("os.getenv", return_value=""):
        tuning = {"effective_cpu_count": 8, "max_concurrent_jobs": 1}
        result = _apply_runtime_cpu_thread_tuning(tuning)
        assert result["enabled"] is True
        assert result["recommended_threads"] == 8
