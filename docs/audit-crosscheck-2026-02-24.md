# Audit Cross-Check & Hardening Status (2026-02-24)

Scope: `scripts/raganything_service.py` and related tests/docs.

## Summary

This document tracks the cross-check of the audit findings and the remediation work applied in Phase 1 / 1.5.

Implemented:

- `pdf_path` path traversal protection (canonicalization + allowlist)
- Request body size / JSON validation hardening
- Per-IP rate limiting (sliding window)
- Optional API key auth for sensitive endpoints
- Webhook URL SSRF guard (scheme/host validation + private target restrictions)
- Reduced internal error leakage to clients
- Unit tests covering the new guards

Not yet implemented:

- FastAPI migration
- Modular refactor (`jobs`, `rag`, HTTP handler split)
- Webhook signing (HMAC) / callback authentication
- End-to-end integration tests for `/process` and `/query`

## Audit Finding Cross-Check

### 1. Hardcoded Values & Config

- Local user paths in docs/examples: Confirmed
  - Status: Partially fixed (service docs updated to generic paths)
- `8767` / `0.0.0.0` values in service/scripts: Confirmed but env-driven
  - Status: Accepted for now (defaults are configurable via env)
- Static Docker/Compose templates in setup wizard: Confirmed
  - Status: Deferred

### 2. Secrets & Security

- `pdf_path` unsanitized path traversal risk: Confirmed
  - Status: Fixed
- `BaseHTTPRequestHandler` limitations (DoS/input validation): Confirmed
  - Status: Partially mitigated (rate limit + body size cap + safer parsing), architecture remains
- Secrets handling (`dotenvx`) looked correct: Confirmed
  - Status: No change required

### 3. Vulnerabilities & Infra

- Fragmented dependencies / requirements layout: Confirmed
  - Status: Deferred
- Missing per-IP rate limiting: Confirmed
  - Status: Fixed (basic in-process limiter)

### 4. Test Coverage

- Missing integration tests for `/process` and `/query`: Confirmed
  - Status: Still missing
- Missing unit tests for runtime guards/queue/breaker: Confirmed
  - Status: Partially fixed (`pdf_path`, rate limiter, webhook/auth helpers)

### 5. Code Smells & Maintainability

- God file (`raganything_service.py`): Confirmed
  - Status: Deferred
- Threading + asyncio mix: Confirmed
  - Status: Deferred
- Global state usage: Confirmed
  - Status: Deferred

## New Security Controls (Configuration)

Relevant env vars introduced/used for hardening:

- `RAG_API_KEY`
- `RAG_AUTH_EXEMPT_PATHS`
- `RAG_ALLOWED_PDF_ROOTS`
- `RAG_ALLOW_UNSAFE_PDF_PATHS`
- `RAG_MAX_REQUEST_BODY_BYTES`
- `RAG_RATE_LIMIT_MAX_REQUESTS`
- `RAG_RATE_LIMIT_WINDOW_SEC`
- `RAG_TRUST_PROXY_HEADERS`
- `RAG_ALLOW_PRIVATE_WEBHOOK_HOSTS`
- `RAG_ALLOWED_WEBHOOK_HOSTS`

See `scripts/RAGANYTHING_SERVICE_README.md` for examples and operational guidance.

## Operational Notes

- If using internal webhook callbacks (N8N, Docker bridge, internal DNS), explicitly allow the host via `RAG_ALLOWED_WEBHOOK_HOSTS`.
- If running behind a reverse proxy, set `RAG_TRUST_PROXY_HEADERS=true` only when the proxy is trusted and strips spoofed headers.
- The in-process rate limiter is intentionally simple; for stronger controls use a reverse proxy (nginx/traefik) or API gateway.
