# RAGanything Service v3.1

Full RAG (Retrieval Augmented Generation) service for academic papers.

## Quick Setup

```bash
cd /media/sam/1TB/rag-service
.venv/bin/python3 -m scripts.setup    # Interactive guided setup
```

The wizard checks Python, MinerU models, Ollama + qwen3:8b, LibreOffice (optional), OPENAI_API_KEY, and service health.

Subcommands: `python3 -m scripts.setup deps | models | service | verify`

**Requirements**: OPENAI_API_KEY (via dotenvx), Ollama with qwen3:8b model, MinerU models (~2GB).

## What's New in v3.1

- **Local Embeddings**: Uses `bge-large-en-v1.5` (sentence-transformers) - no OpenAI dependency
- **Hybrid LLM**: OpenAI `gpt-4o-mini` with fallback to Ollama `qwen3:8b`
- **Vision Toggle**: Disabled by default to save costs (GPT-4o)
- **GSM Integration**: OPENAI_API_KEY loaded from Google Secret Manager

## Architecture

```
┌─────────────────────┐     ┌────────────────────────────────┐
│   N8N (Docker)      │────▶│   RAGanything Service (HOST)   │
│   localhost:5678    │     │   localhost:8767               │
│                     │     │                                │
│  host.docker.internal:8767│   Embeddings: bge-large-en-v1.5│
└─────────────────────┘     │   LLM: OpenAI → Ollama fallback│
                            │   Parser: MinerU               │
                            │   Storage: LightRAG KG         │
                            └────────────────────────────────┘
```

## Service Management

```bash
# Start
sudo systemctl start raganything

# Stop
sudo systemctl stop raganything

# Restart
sudo systemctl restart raganything

# Status
sudo systemctl status raganything

# Logs
journalctl -u raganything -f
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health + status |
| `/status` | GET | Circuit breaker + jobs |
| `/process` | POST | Process PDF → Knowledge Graph |
| `/query` | POST | Semantic search |

## Usage Examples

### Process a PDF

```bash
curl -X POST http://localhost:8767/process \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "/media/sam/1TB/papers/kelly.pdf",
    "paper_id": "arxiv:2401.12345"
  }'
```

### Query Knowledge Graph

```bash
curl -X POST http://localhost:8767/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Kelly criterion formula?",
    "mode": "hybrid"
  }'
```

Query modes:
- `hybrid` - Combines local + global (default, recommended)
- `local` - Specific context chunks
- `global` - Summary across papers

## Storage

- **Knowledge Graph**: `/media/sam/1TB/N8N_dev/rag_knowledge_base/`
- **Extracted Output**: `/media/sam/1TB/N8N_dev/extracted/raganything/`

## Configuration

### Environment Variables

| Variable | Source | Description |
|----------|--------|-------------|
| `OPENAI_API_KEY` | GSM | Optional - enables OpenAI LLM (fallback to Ollama if not set) |
| `RAG_ENABLE_VISION` | .env | Enable vision processing (default: `false`) |
| `OLLAMA_HOST` | .env | Ollama endpoint (default: `http://localhost:11434`) |

### Model Configuration

| Component | Model | Dimension | Source |
|-----------|-------|-----------|--------|
| **Embeddings** | `bge-large-en-v1.5` | 1024 | Local (sentence-transformers) |
| **LLM Primary** | `gpt-4o-mini` | - | OpenAI (if API key available) |
| **LLM Fallback** | `qwen3:8b` | - | Ollama (local) |
| **Vision** | `gpt-4o` | - | OpenAI (disabled by default) |

### Service Settings (in script)

```python
PORT = 8767
PROCESS_TIMEOUT = 600  # 10 minutes
MAX_CONCURRENT_JOBS = 2
```

## Files

| File | Purpose |
|------|---------|
| `raganything_service.py` | Main service script |
| `raganything_start.sh` | Wrapper script (loads GSM secrets) |
| `/etc/systemd/system/raganything.service` | Systemd unit file |

## Systemd Service

Features:
- Auto-start on boot
- Auto-restart on crash (10s delay)
- Loads secrets from Google Secret Manager via wrapper script
- Logs to journald

## Cost Optimization

| Feature | Default | Cost Impact |
|---------|---------|-------------|
| Embeddings | Local | Free |
| LLM | OpenAI with Ollama fallback | Low (gpt-4o-mini is cheap) |
| Vision | Disabled | $0 unless enabled |

To enable vision processing (for image-heavy papers):
```bash
# Add to .env
RAG_ENABLE_VISION=true
```

## Integration with N8N

From N8N container, call:
```
http://host.docker.internal:8767/process
http://host.docker.internal:8767/query
```

## Claude Command

Use `/research-papers "query"` to query the knowledge graph from Claude Code.

## Troubleshooting

### Knowledge base empty after dimension change
If you change embedding models, clear the knowledge base:
```bash
rm -rf /media/sam/1TB/N8N_dev/rag_knowledge_base/*
```

### OpenAI API key not loaded
Check GSM is working:
```bash
gcloud secrets versions access latest --secret="openai-api-key"
```

### Ollama not responding
Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```
