"""Security-focused unit tests for raganything_service request guards."""

from __future__ import annotations

import importlib

import pytest


svc = importlib.import_module("scripts.raganything_service")


def test_sanitize_pdf_path_allows_pdf_within_allowed_root(tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    pdf = allowed_root / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(svc, "ALLOWED_PDF_ROOTS", (allowed_root.resolve(),))
    monkeypatch.setattr(svc, "ALLOW_UNSAFE_PDF_PATHS", False)

    assert svc.sanitize_pdf_path(str(pdf)) == str(pdf.resolve())


def test_sanitize_pdf_path_blocks_file_outside_allowlist(tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    secret_root = tmp_path / "secret"
    allowed_root.mkdir()
    secret_root.mkdir()
    secret_pdf = secret_root / "secret.pdf"
    secret_pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(svc, "ALLOWED_PDF_ROOTS", (allowed_root.resolve(),))
    monkeypatch.setattr(svc, "ALLOW_UNSAFE_PDF_PATHS", False)

    with pytest.raises(PermissionError, match="outside allowed"):
        svc.sanitize_pdf_path(str(secret_pdf))


def test_sanitize_pdf_path_blocks_traversal_escape(tmp_path, monkeypatch):
    root = tmp_path / "root"
    allowed_root = root / "allowed"
    secret_root = root / "secret"
    allowed_root.mkdir(parents=True)
    secret_root.mkdir()
    secret_pdf = secret_root / "secret.pdf"
    secret_pdf.write_bytes(b"%PDF-1.4\n")

    traversal_path = allowed_root / ".." / "secret" / "secret.pdf"

    monkeypatch.setattr(svc, "ALLOWED_PDF_ROOTS", (allowed_root.resolve(),))
    monkeypatch.setattr(svc, "ALLOW_UNSAFE_PDF_PATHS", False)

    with pytest.raises(PermissionError, match="outside allowed"):
        svc.sanitize_pdf_path(str(traversal_path))


def test_sanitize_pdf_path_rejects_non_pdf_file(tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    txt = allowed_root / "notes.txt"
    txt.write_text("not a pdf")

    monkeypatch.setattr(svc, "ALLOWED_PDF_ROOTS", (allowed_root.resolve(),))
    monkeypatch.setattr(svc, "ALLOW_UNSAFE_PDF_PATHS", False)

    with pytest.raises(ValueError, match=".pdf"):
        svc.sanitize_pdf_path(str(txt))


def test_translate_container_to_host_path_uses_configured_mapping(monkeypatch):
    monkeypatch.setattr(svc, "_CONTAINER_PATH_PREFIX", "/workspace/")
    monkeypatch.setattr(svc, "_HOST_PATH_PREFIX", "/srv/default/")
    monkeypatch.setattr(
        svc,
        "_PATH_MAPPINGS",
        "/workspace/data/:/mnt/data/,/workspace/alt/:/mnt/alt/",
    )

    assert (
        svc.translate_container_to_host_path("/workspace/data/docs/file.pdf")
        == "/mnt/data/docs/file.pdf"
    )


def test_ip_rate_limiter_blocks_after_threshold():
    limiter = svc.IpRateLimiter(max_requests=2, window_sec=60)

    assert limiter.allow("127.0.0.1") == (True, 0)
    assert limiter.allow("127.0.0.1") == (True, 0)

    allowed, retry_after = limiter.allow("127.0.0.1")
    assert allowed is False
    assert retry_after > 0


def test_ip_rate_limiter_evicts_stale_buckets():
    limiter = svc.IpRateLimiter(max_requests=100, window_sec=1)

    for i in range(2001):
        limiter.allow(f"10.0.{i // 256}.{i % 256}")

    original_size = len(limiter._requests)
    assert original_size >= 2001

    for bucket in limiter._requests.values():
        for j in range(len(bucket)):
            bucket[j] = bucket[j] - 10

    limiter.allow("192.168.1.1")
    assert len(limiter._requests) < original_size


def test_extract_api_key_from_headers_supports_bearer_and_x_api_key():
    assert svc._extract_api_key({"X-API-Key": "abc123"}) == "abc123"
    assert svc._extract_api_key({"Authorization": "Bearer token-1"}) == "token-1"
    assert svc._extract_api_key({}) is None


def test_extract_client_ip_from_xff_returns_last_non_empty_entry():
    assert svc._extract_client_ip_from_xff("1.1.1.1, 2.2.2.2") == "2.2.2.2"
    assert svc._extract_client_ip_from_xff(" 1.1.1.1 ,, 3.3.3.3 ") == "3.3.3.3"
    assert svc._extract_client_ip_from_xff("   ") is None


def test_sanitize_webhook_url_blocks_localhost_by_default(monkeypatch):
    monkeypatch.setattr(svc, "ALLOW_PRIVATE_WEBHOOK_HOSTS", False)
    monkeypatch.setattr(svc, "ALLOWED_WEBHOOK_HOSTS", ())

    with pytest.raises(PermissionError, match="not allowed"):
        svc.sanitize_webhook_url("http://localhost:5678/callback")


def test_sanitize_webhook_url_allows_allowlisted_internal_host(monkeypatch):
    monkeypatch.setattr(svc, "ALLOW_PRIVATE_WEBHOOK_HOSTS", False)
    monkeypatch.setattr(svc, "ALLOWED_WEBHOOK_HOSTS", ("host.docker.internal", ".trusted.internal"))

    url, ip = svc.sanitize_webhook_url("http://host.docker.internal:5678/webhook")
    assert url == "http://host.docker.internal:5678/webhook"


def test_sanitize_webhook_url_rejects_private_resolved_ip(monkeypatch):
    monkeypatch.setattr(svc, "ALLOW_PRIVATE_WEBHOOK_HOSTS", False)
    monkeypatch.setattr(svc, "ALLOWED_WEBHOOK_HOSTS", ())
    monkeypatch.setattr(svc, "_resolve_webhook_ips", lambda host, port: {"10.0.0.4"})

    with pytest.raises(PermissionError, match="not allowed"):
        svc.sanitize_webhook_url("https://example.com/callback")


def test_sanitize_webhook_url_allows_public_resolved_ip(monkeypatch):
    monkeypatch.setattr(svc, "ALLOW_PRIVATE_WEBHOOK_HOSTS", False)
    monkeypatch.setattr(svc, "ALLOWED_WEBHOOK_HOSTS", ())
    monkeypatch.setattr(svc, "_resolve_webhook_ips", lambda host, port: {"93.184.216.34"})

    url, ip = svc.sanitize_webhook_url("https://example.com/callback")
    assert url == "https://example.com/callback"
    assert ip == "93.184.216.34"


def test_pinned_host_adapter_http_routes_to_pinned_ip():
    """HTTP adapter routes connection_from_host to the pinned IP."""
    adapter = svc.PinnedHostAdapter("1.2.3.4")
    adapter.init_poolmanager(num_pools=1, maxsize=1, block=False)

    pool = adapter.poolmanager.connection_from_host("example.com", 80, "http")
    assert pool.host == "1.2.3.4"


def test_pinned_host_adapter_http_with_hostname_does_not_inject_tls_kwargs():
    """HTTP pinning with hostname set should not break urllib3 HTTPConnection."""
    adapter = svc.PinnedHostAdapter("1.2.3.4", hostname="example.com", scheme="http")
    adapter.init_poolmanager(num_pools=1, maxsize=1, block=False)

    pool = adapter.poolmanager.connection_from_host("example.com", 80, "http")
    assert pool.host == "1.2.3.4"
    # Regression check: this raised TypeError when assert_hostname leaked into HTTP conn_kw.
    conn = pool._new_conn()
    assert conn.host == "1.2.3.4"


def test_pinned_host_adapter_https_preserves_sni_and_assert_hostname():
    """HTTPS adapter pins IP but preserves hostname for cert/SNI validation."""
    adapter = svc.PinnedHostAdapter("93.184.216.34", hostname="example.com", scheme="https")
    adapter.init_poolmanager(num_pools=1, maxsize=1, block=False)

    pool = adapter.poolmanager.connection_from_host("example.com", 443, "https")
    # TCP goes to the pinned IP
    assert pool.host == "93.184.216.34"
    # TLS cert is validated against original hostname
    assert pool.assert_hostname == "example.com"
    # SNI sends original hostname
    assert pool.conn_kw.get("server_hostname") == "example.com"


def test_pinned_host_adapter_without_hostname_has_no_sni():
    """Adapter without hostname should not set assert_hostname or server_hostname."""
    adapter = svc.PinnedHostAdapter("1.2.3.4", hostname=None, scheme="https")
    adapter.init_poolmanager(num_pools=1, maxsize=1, block=False)

    pool = adapter.poolmanager.connection_from_host("example.com", 443, "https")
    assert pool.host == "1.2.3.4"
    assert pool.assert_hostname is None
    assert "server_hostname" not in pool.conn_kw
