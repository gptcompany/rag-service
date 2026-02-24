# RAGanything Service (v3.3, Hardened)

HTTP service for document processing (`/process`) and semantic query (`/query`) over the local RAGAnything knowledge base.

## Quick Setup

```bash
cd /path/to/rag-service
.venv/bin/python3 -m scripts.setup
```

Wizard subcommands:

```bash
python3 -m scripts.setup deps
python3 -m scripts.setup models
python3 -m scripts.setup service
python3 -m scripts.setup verify
```

## Endpoints

| Endpoint | Method | Auth (if `RAG_API_KEY` set) | Description |
|----------|--------|-----------------------------|-------------|
| `/health` | GET | No (default exempt) | Liveness + basic status |
| `/status` | GET | No (default exempt) | Circuit breaker + queue status |
| `/jobs` | GET | Yes | Active jobs |
| `/jobs/{id}` | GET | Yes | Job status |
| `/process` | POST | Yes | Submit PDF processing job |
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

### Process a PDF (API key enabled)

```bash
curl -X POST http://localhost:8767/process \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${RAG_API_KEY}" \
  -d '{
    "pdf_path": "/absolute/path/to/papers/kelly.pdf",
    "paper_id": "arxiv:2401.12345",
    "webhook_url": "https://example.com/rag-callback"
  }'
```

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

Query modes:

- `hybrid` (default)
- `local`
- `global`

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

Rejected by default:

- `localhost`
- private IP ranges (`10.0.0.0/8`, `192.168.0.0/16`, etc.)
- `.local`, `.internal`, loopback/link-local/reserved targets

Overrides (use carefully):

- `RAG_ALLOW_PRIVATE_WEBHOOK_HOSTS=true`
- `RAG_ALLOWED_WEBHOOK_HOSTS=host.docker.internal,.trusted.internal`

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
| `RAG_ALLOW_PRIVATE_WEBHOOK_HOSTS` | `false` | Allow internal/private webhook targets |
| `RAG_ALLOWED_WEBHOOK_HOSTS` | unset | CSV hostname allowlist (`host`, `.suffix`) |

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
