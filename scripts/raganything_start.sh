#!/bin/bash
# RAGanything Service Wrapper - Loads dotenvx secrets before starting
#
# MIGRATION (2026-01-17): GSM → SOPS
# MIGRATION (2026-02-02): SOPS → dotenvx

set -e

DOTENVX="/home/sam/.local/bin/dotenvx"
ENV_FILE="/media/sam/1TB/.env"

# Activate virtual environment
export VIRTUAL_ENV="/media/sam/1TB/rag-service/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[RAGanything] ERROR: $ENV_FILE not found"
    exit 1
fi

# Verify critical secret
OPENAI_KEY=$($DOTENVX get OPENAI_API_KEY -f "$ENV_FILE" 2>/dev/null)
if [[ -z "$OPENAI_KEY" ]]; then
    echo "[RAGanything] CRITICAL: OPENAI_API_KEY not available!"
    echo "[RAGanything] RAG features will fail. Check dotenvx configuration."
else
    echo "[RAGanything] ✅ OPENAI_API_KEY loaded"
fi

echo "[RAGanything] Starting service..."
exec $DOTENVX run -f "$ENV_FILE" -- /media/sam/1TB/rag-service/.venv/bin/python3 /media/sam/1TB/rag-service/scripts/raganything_service.py
