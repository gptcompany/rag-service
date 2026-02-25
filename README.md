# RAG Service

This project provides a hardened HTTP service for document processing and semantic query over a RAGAnything knowledge base.

## Documentation

The primary documentation is located at [scripts/RAGANYTHING_SERVICE_README.md](scripts/RAGANYTHING_SERVICE_README.md).

## Quick Start

```bash
# Install the local setup CLI (editable)
python3 -m pip install -e .
# or
uv pip install -e .

# Run the setup wizard (interactive menu)
rag-setup

# Fallback (same wizard entry point)
python3 -m scripts.setup
```

If the CLI was installed into the project venv without activating it, use `.venv/bin/rag-setup`.

## Audit & Status

See [docs/audit-crosscheck-2026-02-24.md](docs/audit-crosscheck-2026-02-24.md) for the latest security hardening status.
