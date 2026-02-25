"""Unit tests for runtime queue auto-tuning in raganything_service."""

from __future__ import annotations

import importlib


svc = importlib.import_module("scripts.raganything_service")


def test_auto_max_concurrent_jobs_is_conservative_for_cpu_heavy_workloads():
    assert svc._auto_max_concurrent_jobs(4) == 1
    assert svc._auto_max_concurrent_jobs(40) == 1
    assert svc._auto_max_concurrent_jobs(64) == 2


def test_auto_queue_depth_scales_with_workers():
    assert svc._auto_queue_depth(1) == 4
    assert svc._auto_queue_depth(2) == 8
    assert svc._auto_queue_depth(5) == 16  # capped


def test_detect_effective_cpu_capacity_uses_affinity_and_cgroup_limit(monkeypatch):
    monkeypatch.setattr(svc.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(svc.os, "sched_getaffinity", lambda _pid: set(range(40)), raising=False)
    monkeypatch.setattr(svc, "_detect_cgroup_cpu_limit", lambda: (24, "cgroup-v2"))

    info = svc._detect_effective_cpu_capacity()

    assert info["host_cpu_count"] == 64
    assert info["affinity_cpu_count"] == 40
    assert info["cgroup_cpu_limit"] == 24
    assert info["effective_cpu_count"] == 24
    assert "affinity" in info["effective_cpu_source"]
    assert "cgroup-v2" in info["effective_cpu_source"]


def test_resolve_runtime_queue_tuning_prefers_env_overrides(monkeypatch):
    monkeypatch.setenv("RAG_MAX_CONCURRENT_JOBS", "3")
    monkeypatch.setenv("RAG_MAX_QUEUE_DEPTH", "7")
    monkeypatch.setattr(
        svc,
        "_detect_effective_cpu_capacity",
        lambda: {
            "host_cpu_count": 64,
            "affinity_cpu_count": 64,
            "cgroup_cpu_limit": None,
            "effective_cpu_count": 64,
            "effective_cpu_source": "cpu_count+affinity",
        },
    )

    tuning = svc._resolve_runtime_queue_tuning()

    assert tuning["max_concurrent_jobs"] == 3
    assert tuning["max_concurrent_jobs_source"] == "env:RAG_MAX_CONCURRENT_JOBS"
    assert tuning["max_queue_depth"] == 7
    assert tuning["max_queue_depth_source"] == "env:RAG_MAX_QUEUE_DEPTH"


def test_resolve_runtime_queue_tuning_invalid_env_falls_back_to_auto(monkeypatch, capsys):
    monkeypatch.setenv("RAG_MAX_CONCURRENT_JOBS", "abc")
    monkeypatch.setenv("RAG_MAX_QUEUE_DEPTH", "-1")
    monkeypatch.setattr(
        svc,
        "_detect_effective_cpu_capacity",
        lambda: {
            "host_cpu_count": 40,
            "affinity_cpu_count": 40,
            "cgroup_cpu_limit": None,
            "effective_cpu_count": 40,
            "effective_cpu_source": "cpu_count+affinity",
        },
    )

    tuning = svc._resolve_runtime_queue_tuning()
    captured = capsys.readouterr()

    assert tuning["max_concurrent_jobs"] == 1
    assert tuning["max_concurrent_jobs_source"] == "auto:cpu-discovery"
    assert tuning["max_queue_depth"] == 4
    assert tuning["max_queue_depth_source"] == "auto:4x-workers"
    assert "WARNING" in captured.out


def test_detect_cgroup_cpu_limit_parses_v2_cpu_max(monkeypatch):
    def _fake_read(path: str):
        if path == "/sys/fs/cgroup/cpu.max":
            return "250000 100000"
        return None

    monkeypatch.setattr(svc, "_read_text_file", _fake_read)

    limit, source = svc._detect_cgroup_cpu_limit()

    assert limit == 3
    assert source == "cgroup-v2"


def test_apply_runtime_cpu_thread_tuning_sets_defaults_when_unset(monkeypatch):
    for key in (
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_WAIT_POLICY",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "TORCH_NUM_INTEROP_THREADS",
        "TOKENIZERS_PARALLELISM",
        "RAG_AUTO_CPU_THREAD_TUNING",
    ):
        monkeypatch.delenv(key, raising=False)

    runtime_queue_tuning = {
        "effective_cpu_count": 40,
        "max_concurrent_jobs": 1,
    }
    tuning = svc._apply_runtime_cpu_thread_tuning(runtime_queue_tuning)

    assert tuning["enabled"] is True
    assert tuning["recommended_threads"] == 16
    assert tuning["recommended_torch_interop_threads"] == 2
    assert tuning["applied_env"]["OMP_NUM_THREADS"] == "16"
    assert tuning["applied_env"]["TORCH_NUM_INTEROP_THREADS"] == "2"
    assert tuning["preserved_env"] == {}


def test_apply_runtime_cpu_thread_tuning_preserves_existing_env(monkeypatch):
    for key in (
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_WAIT_POLICY",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "TORCH_NUM_INTEROP_THREADS",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "24")
    monkeypatch.setenv("TORCH_NUM_THREADS", "12")
    monkeypatch.delenv("RAG_AUTO_CPU_THREAD_TUNING", raising=False)

    tuning = svc._apply_runtime_cpu_thread_tuning(
        {"effective_cpu_count": 40, "max_concurrent_jobs": 1}
    )

    assert tuning["enabled"] is True
    assert tuning["preserved_env"]["OMP_NUM_THREADS"] == "24"
    assert tuning["preserved_env"]["TORCH_NUM_THREADS"] == "12"
    # Missing vars are still auto-filled
    assert tuning["applied_env"]["MKL_NUM_THREADS"] == "16"


def test_apply_runtime_cpu_thread_tuning_can_be_disabled(monkeypatch):
    monkeypatch.setenv("RAG_AUTO_CPU_THREAD_TUNING", "false")

    tuning = svc._apply_runtime_cpu_thread_tuning(
        {"effective_cpu_count": 40, "max_concurrent_jobs": 1}
    )

    assert tuning["enabled"] is False
    assert tuning["source"] == "env:RAG_AUTO_CPU_THREAD_TUNING=false"
