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

Wizard notes:
- In Docker mode, host-only dependency steps are marked as "Handled by selected mode" (expected behavior).
- You can re-run any step manually even when it is already `OK`.
- `Service configuration` now supports custom Ollama endpoint URL (default `http://localhost:11434`), custom/free port selection, and pre-checks for port conflicts.
- In local mode, the Ollama step can attempt automatic installation if Ollama is missing.

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

## Prova modelli (F-775): arresto termico del rag-service

`scripts/model_trial.py` (`retrieve`, `run`) gira sotto il supervisore termico (`scripts/trial_supervisor.py`). All'arresto, a soglia
superata o sensore perso, il supervisore: (1) ferma il figlio, (2) **ferma il container** con `docker compose stop rag` (attesa
SIGTERM→SIGKILL predefinita di Docker, nessun `-t`), (3) scarica i modelli di ollama, (4) verifica che il container sia fermo e che
`/api/ps` sia vuoto. Il container va fermato PRIMA: una query di retrieve gia' partita continua dentro il rag-service e, dopo lo
scarico, ricaricherebbe il modello. Se un controllo fallisce l'esito e' `thermal_shutdown_failed` (`alarm: true`) e il codice di uscita
e' 70 invece di 75. Il registro (stderr, JSON) riporta temperatura letta, soglie, decisione, tempi di stop e scarico e l'esito dei due
controlli. Usa solo i permessi Docker dell'utente: niente sudo. `retrieve` parte solo se il compose indicato (`--compose-file`, di default quello del
repository; il nome del progetto deriva dalla cartella) ha il rag-service in esecuzione: un compose di un altro progetto fermerebbe
"con successo" un servizio diverso da quello in uso.

Precondizione non verificata dal codice (TODO): `--rag-url`, `--compose-file` e `--ollama-host` devono indicare lo stesso deployment.

**Nessun riavvio automatico.** Lo stop manuale non fa scattare `restart: unless-stopped` (provato sul docker vero in
`tests/test_trial_supervisor_shutdown.py`). Il rag-service resta fermo finche' non lo si rialza con l'azione esplicita e registrata,
che rifiuta se i sensori sono sopra le soglie da essi derivate o illeggibili:

```
python scripts/model_trial.py rag-restart   # --compose-file / --rag-service per un compose o servizio diversi
```

**Assunzione dichiarata.** Oggi rag-service non ha altri consumatori che l'arresto danneggi (la prova gira prima della produzione).
In produzione l'interruzione del servizio all'arresto termico va dichiarata qui e nel runbook; se serve continuita', l'alternativa
(difesa dentro il servizio) si riapre come issue separata.
