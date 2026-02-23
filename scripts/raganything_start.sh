#!/bin/bash
# RAGanything Service Wrapper - Loads dotenvx secrets before starting
#
# MIGRATION (2026-01-17): GSM -> SOPS
# MIGRATION (2026-02-02): SOPS -> dotenvx
# MIGRATION (2026-02-23): Removed hardcoded paths for public publication

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_ROOT="$(dirname "$SCRIPT_DIR")"

DOTENVX="${DOTENVX:-dotenvx}"
ENV_FILE="${RAG_ENV_FILE:-$SERVICE_ROOT/.env}"

# Activate virtual environment
export VIRTUAL_ENV="${RAG_VENV:-$SERVICE_ROOT/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[RAGanything] ERROR: $ENV_FILE not found"
    exit 1
fi

# Verify critical secret exists (never assign secret value to a variable)
OPENAI_KEY_EXISTS=$($DOTENVX get OPENAI_API_KEY -f "$ENV_FILE" 2>/dev/null | grep -q . && echo "yes" || echo "no")
if [[ "$OPENAI_KEY_EXISTS" != "yes" ]]; then
    echo "[RAGanything] CRITICAL: OPENAI_API_KEY not available!"
    echo "[RAGanything] RAG features will fail. Check dotenvx configuration."
else
    echo "[RAGanything] OPENAI_API_KEY loaded"
fi

echo "[RAGanything] Starting service..."
exec $DOTENVX run -f "$ENV_FILE" -- "$VIRTUAL_ENV/bin/python3" "$SCRIPT_DIR/raganything_service.py"
