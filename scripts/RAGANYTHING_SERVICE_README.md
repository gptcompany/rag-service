# RAGanything Service (v3.3-smart, Hardened)

HTTP service for document processing (`/process`) and semantic query (`/query`) over the local RAGAnything knowledge base.

## Quick Setup

```bash
cd /path/to/rag-service

# Install setup CLI entrypoint (editable)
pip install -e .
# or
uv pip install -e .

# Run interactive setup menu
rag-setup

# Fallback (same wizard)
.venv/bin/python3 -m scripts.setup
```

Wizard subcommands:

```bash
rag-setup deploy   # Choose deployment mode: host or Docker
rag-setup deps     # Install dependencies: Python venv, MinerU, Ollama, LibreOffice
rag-setup models   # Download/verify AI models: MinerU + Ollama
rag-setup config   # Configure service: models, parser, network
rag-setup service  # Check RAG service health and startup
rag-setup verify   # Full service verification and status display
```

If you prefer not to install the CLI entrypoint, replace `rag-setup` with `python3 -m scripts.setup`.

## Setup Wizard (step-by-step)

Running `rag-setup` (or `python -m scripts.setup`) without arguments opens an interactive menu with free navigation across the steps below:

| # | Step | Description | Skip condition |
|---|------|-------------|----------------|
| 1 | **Deploy mode** | Choose Host (systemd) or Docker (Dockerfile + compose) | — |
| 2 | **Python venv** | Verify `.venv` with dependencies installed | Skipped in Docker mode |
| 3 | **MinerU models** | Download MinerU HuggingFace models to `~/.cache/huggingface` | Skipped in Docker mode |
| 4 | **Ollama** | Verify Ollama is installed, serving, and has the configured model | Skipped in Docker + sidecar |
| 5 | **LibreOffice** | Check `libreoffice --headless` is available (for PPTX/DOCX) | Skipped in Docker mode |
| 6 | **Secrets** | Verify `OPENAI_API_KEY` is set in dotenvx-encrypted `.env` | — |
| 7 | **Configuration** | Interactive 6-section config: OpenAI model, Ollama model, embedding, reranker, parser, network | — |
| 8 | **Service** | Check if the RAG service is running on the configured port | — |
| 9 | **Verify** | End-to-end status summary | — |

In Docker mode, steps 2-5 are conditionally skipped (dependencies are handled by the container image).

### Dependencies

| Dependency | Required by | Install |
|------------|-------------|---------|
| Python 3.10+ | Core | `apt install python3.10` |
| [dotenvx](https://dotenvx.com) | Secrets management | `curl -fsS https://dotenvx.sh \| sh` |
| [Ollama](https://ollama.com) | Local LLM | `curl -fsSL https://ollama.com/install.sh \| sh` |
| LibreOffice | PPTX/DOCX conversion | `apt install libreoffice-core` |
| MinerU models | PDF parsing (default) | Auto-downloaded by wizard |
| `OPENAI_API_KEY` | GPT queries + vision | Set via `dotenvx set OPENAI_API_KEY <key> -f .env` |

### Docker mode

When "Docker" is selected in step 1, the wizard:
- Generates `Dockerfile` (if not present)
- Generates `docker-compose.yml` with chosen Ollama mode (external or sidecar)
- External mode uses `host.docker.internal:host-gateway` for Linux compatibility
- Sidecar mode adds a dedicated Ollama container with GPU reservations

### Host mode (systemd)

When "Host" is selected in step 1:
- The service is configured to run directly on the machine.
- Use the included script to automatically create/update the systemd unit:
  ```bash
  sudo bash update-systemd.sh
  ```
- This will install the `raganything.service` unit and start it.

## Endpoints

| Endpoint | Method | Auth (if `RAG_API_KEY` set) | Description |
|----------|--------|-----------------------------|-------------|
| `/health` | GET | No (default exempt) | Liveness + basic status |
| `/status` | GET | No (default exempt) | Circuit breaker + queue status |
| `/jobs` | GET | Yes | Active jobs |
| `/jobs/{id}` | GET | Yes | Job status |
| `/process` | POST | Yes | Submit PDF processing job |
| `/process/sync` | POST | Yes | Synchronous PDF processing (legacy) |
| `/query` | POST | Yes | Query knowledge graph |
| `/reset-circuit-breaker` | GET | Yes | Manual breaker reset |

## Security Hardening (Phase 1 / 1.5)

Implemented in `scripts/raganything_service.py`:

- `pdf_path` canonicalization + allowlist checks (anti path traversal)
- Request body size cap + JSON parsing validation
- Per-IP rate limiting (sliding window)
- Optional API key authentication (`X-API-Key` or `Authorization: Bearer`)
- Webhook callback URL validation (SSRF guard, host restrictions)
- Safer client-facing error messages (reduced internal leakage)

Limitations still open:

- Service still uses `BaseHTTPRequestHandler` (no FastAPI/Pydantic yet)
- No built-in TLS termination (use reverse proxy)
- No webhook signature/HMAC validation

## Usage Examples

### Process a PDF (no auth)

```bash
curl -X POST http://localhost:8767/process \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "/absolute/path/to/papers/kelly.pdf",
    "paper_id": "arxiv:2401.12345"
  }'
```

### Process a PDF (API key enabled, full options)

```bash
curl -X POST http://localhost:8767/process \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${RAG_API_KEY}" \
  -d '{
    "pdf_path": "/absolute/path/to/papers/kelly.pdf",
    "paper_id": "arxiv:2401.12345",
    "webhook_url": "https://example.com/rag-callback",
    "force_parser": "docling",
    "force_reprocess": true
  }'
```

Optional `/process` parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `webhook_url` | string | Callback URL for async result delivery |
| `force_parser` | string | Override smart router: `mineru`, `docling`, or `paddleocr` |
| `force_reprocess` | bool | Reprocess even if cached result exists (default `false`) |

### Query Knowledge Graph

```bash
curl -X POST http://localhost:8767/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RAG_API_KEY}" \
  -d '{
    "query": "What is the Kelly criterion formula?",
    "mode": "hybrid"
  }'
```

Query modes: `hybrid` (default), `local`, `global`

Optional `/query` parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `mode` | string | Query mode (default `hybrid`) |
| `context_only` | bool | Return context without LLM synthesis (default `false`) |

## PDF Path Security

`/process` now rejects unsafe `pdf_path` values unless explicitly allowed.

Checks performed:

- must be a string
- must be absolute path
- must exist and be readable
- must point to a `.pdf` file
- is resolved to canonical path (`..` traversal blocked)
- must be inside allowed directories (unless override enabled)

Primary controls:

- `RAG_ALLOWED_PDF_ROOTS` (comma-separated absolute directories)
- `RAG_ALLOW_UNSAFE_PDF_PATHS=false` (default, recommended)

If `RAG_ALLOWED_PDF_ROOTS` is not set, the service derives a default allowlist from:

- `RAG_HOST_PATH_PREFIX`
- host-side prefixes in `RAG_PATH_MAPPINGS`
- local `./data` directory

## Webhook Security (SSRF Guard)

`webhook_url` is optional but validated when present.

Allowed by default:

- `http://` or `https://`
- valid URL without embedded credentials
- public/global hosts only
- DNS-resolved target IP is validated before callback execution

Rejected by default:

- `localhost`
- private IP ranges (`10.0.0.0/8`, `192.168.0.0/16`, etc.)
- `.local`, `.internal`, loopback/link-local/reserved targets

Overrides (use carefully):

- `RAG_ALLOW_PRIVATE_WEBHOOK_HOSTS=true`
- `RAG_ALLOWED_WEBHOOK_HOSTS=host.docker.internal,.trusted.internal`

Callback delivery hardening:

- HTTP callbacks use DNS pinning to the pre-validated IP
- HTTPS callbacks use DNS pinning with SNI-aware TLS settings (certificate validation still checks original hostname)
- Cached `/process` responses reuse the same webhook helper path (no callback security bypass)

## Authentication & Rate Limiting

### API Key Auth (optional)

If `RAG_API_KEY` is set, non-exempt endpoints require auth via:

- `X-API-Key: ...`
- `Authorization: Bearer ...`

Default auth exemptions:

- `/health`
- `/status`

Override with:

- `RAG_AUTH_EXEMPT_PATHS=/health,/status`

### Rate Limiting

Simple per-IP sliding-window limiter (all endpoints except `/health` and `/status`).

- `RAG_RATE_LIMIT_MAX_REQUESTS` (default `120`)
- `RAG_RATE_LIMIT_WINDOW_SEC` (default `60`)

If behind a trusted reverse proxy:

- `RAG_TRUST_PROXY_HEADERS=true` to honor `X-Forwarded-For`
- `RAG_TRUSTED_PROXY_HOPS=1` (default) to select the Nth address from the right in `X-Forwarded-For`

## Configuration (Environment Variables)

### Core service

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_HOST` | `0.0.0.0` | Bind host |
| `RAG_PORT` | `8767` | Bind port |
| `RAG_OUTPUT_BASE` | `./data/extracted` | Extracted output directory |
| `RAG_STORAGE_DIR` | `./data/rag_knowledge_base` | Knowledge base storage |
| `RAG_PARSER_THRESHOLD` | `15` | Parser routing threshold (pages) |
| `RAG_DEFAULT_PARSER` | `mineru` | Default parser |

### Models / inference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | unset | Enables OpenAI primary LLM |
| `RAG_OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `RAG_OLLAMA_MODEL` | `qwen3:8b` | Ollama model |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `RAG_EMBEDDING_DIM` | `1024` | Embedding dimension |
| `RAG_ENABLE_VISION` | `false` | Vision processing |
| `RAG_ENABLE_RERANK` | `true` | Reranker toggle |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model |

### Path mapping / docker interop

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_HOST_PATH_PREFIX` | unset | Host path prefix for container mapping |
| `RAG_CONTAINER_PATH_PREFIX` | `/workspace/` | Container prefix |
| `RAG_PATH_MAPPINGS` | unset | CSV `container_prefix:host_prefix` mappings |

### Security / hardening

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_API_KEY` | unset | Enables API key auth on non-exempt endpoints |
| `RAG_AUTH_EXEMPT_PATHS` | `/health,/status` | CSV paths exempt from API key auth |
| `RAG_ALLOWED_PDF_ROOTS` | derived | CSV allowlist of directories for `pdf_path` |
| `RAG_ALLOW_UNSAFE_PDF_PATHS` | `false` | Disable `pdf_path` allowlist checks (not recommended) |
| `RAG_MAX_REQUEST_BODY_BYTES` | `1048576` | JSON request body size limit |
| `RAG_RATE_LIMIT_MAX_REQUESTS` | `120` | Per-IP request count |
| `RAG_RATE_LIMIT_WINDOW_SEC` | `60` | Rate-limit window (seconds) |
| `RAG_TRUST_PROXY_HEADERS` | `false` | Trust `X-Forwarded-For` |
| `RAG_TRUSTED_PROXY_HOPS` | `1` | Trusted proxy hops for XFF parsing (Nth from right) |
| `RAG_ALLOW_PRIVATE_WEBHOOK_HOSTS` | `false` | Allow internal/private webhook targets |
| `RAG_ALLOWED_WEBHOOK_HOSTS` | unset | CSV hostname allowlist (`host`, `.suffix`) |

## Smart Parser Router

Documents are automatically routed to the best parser based on page count:

- Short documents (< `RAG_PARSER_THRESHOLD` pages) → default parser (configurable)
- Long documents (>= threshold) → `docling` (better for large PDFs)

Override per-request with `force_parser` in `/process`.

## PDF Hash Deduplication

Each processed PDF is hashed (SHA-256) and tracked in `processed_pdfs.json`. Resubmitting the same content with a different `paper_id` is detected and skipped. Use `force_reprocess=true` to bypass.

## Storage

Defaults (relative to repo root):

- Knowledge graph: `data/rag_knowledge_base/`
- Extracted output: `data/extracted/`

## Integration Notes (N8N / Containers)

- Containerized clients can send container paths via `RAG_CONTAINER_PATH_PREFIX` + `RAG_PATH_MAPPINGS`.
- If you need internal callbacks (for example `host.docker.internal` or `n8n.internal`), configure:
  - `RAG_ALLOWED_WEBHOOK_HOSTS=host.docker.internal,.internal`
  - or `RAG_ALLOW_PRIVATE_WEBHOOK_HOSTS=true` (less strict)

## Troubleshooting

### `403 pdf_path is outside allowed directories`

Configure `RAG_ALLOWED_PDF_ROOTS` to include the real host directory that contains PDFs.

### `403 webhook_url host is not allowed`

Allow the hostname explicitly with `RAG_ALLOWED_WEBHOOK_HOSTS`, or (less safe) set `RAG_ALLOW_PRIVATE_WEBHOOK_HOSTS=true`.

### `401 Unauthorized`

Either:

- provide `X-API-Key` / `Authorization: Bearer ...`
- or unset `RAG_API_KEY`

### `429 Rate limit exceeded`

Increase limits or put the service behind a reverse proxy / queue:

- `RAG_RATE_LIMIT_MAX_REQUESTS`
- `RAG_RATE_LIMIT_WINDOW_SEC`
