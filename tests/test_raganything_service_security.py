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
