# RAG Service

![CI](https://github.com/gptcompany/rag-service/actions/workflows/ci.yml/badge.svg?branch=main)
![Sandbox Validation](https://github.com/gptcompany/rag-service/actions/workflows/sandbox-validate.yml/badge.svg?branch=main)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/gptprojectmanager/ac39e6516b7114f96b84ba445b8e7a83/raw/rag-service-coverage.json)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)
![RAG](https://img.shields.io/badge/RAG-RAGAnything-blueviolet?style=flat-square)
![License](https://img.shields.io/github/license/gptcompany/rag-service?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/gptcompany/rag-service?style=flat-square)
![Issues](https://img.shields.io/github/issues/gptcompany/rag-service?style=flat-square)
![Lines of Code](https://sloc.xyz/github/gptcompany/rag-service)

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

## Performance (CPU / MinerU)

The service now includes plug-and-play CPU auto-tuning for MinerU workloads:

- Auto-detects visible CPU capacity (host cores, CPU affinity, cgroup quota)
- Chooses a conservative default job concurrency (`RAG_MAX_CONCURRENT_JOBS`) for CPU-heavy parsing
- Chooses a bounded queue depth (`RAG_MAX_QUEUE_DEPTH`)
- Applies safe thread defaults (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `TORCH_NUM_THREADS`, etc.) **only if not already set**

You can inspect the effective values at runtime via `GET /health` (`runtime_tuning` and `jobs` fields).

Power-user overrides (optional):
- `RAG_MAX_CONCURRENT_JOBS`
- `RAG_MAX_QUEUE_DEPTH`
- `RAG_AUTO_CPU_THREAD_TUNING=false` (disable auto thread defaults)

## Audit & Status

See [docs/audit-crosscheck-2026-02-24.md](docs/audit-crosscheck-2026-02-24.md) for the latest security hardening status.
