#!/usr/bin/env bash
# Update RAGAnything systemd unit to point to rag-service location.
# Run: sudo bash /path/to/rag-service/update-systemd.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_ROOT="${RAG_SERVICE_ROOT:-$SCRIPT_DIR}"
SERVICE_USER="${RAG_SERVICE_USER:-$(whoami)}"
ENV_FILE="${RAG_ENV_FILE:-$SERVICE_ROOT/.env}"
RAG_VENV_ROOT="${RAG_VENV:-$SERVICE_ROOT/.venv}"
RAG_PORT_VALUE="${RAG_PORT:-8767}"
START_SCRIPT="$SERVICE_ROOT/scripts/raganything_start.sh"

if [[ ! -x "$START_SCRIPT" ]]; then
  echo "[rag-service] ERROR: start script not found at $START_SCRIPT" >&2
  exit 1
fi

cat > /etc/systemd/system/raganything.service <<EOF
[Unit]
Description=RAGanything Service - Full RAG for Academic Papers
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${SERVICE_ROOT}
Environment=RAG_ENV_FILE=${ENV_FILE}
Environment=RAG_VENV=${RAG_VENV_ROOT}
Environment=RAG_PORT=${RAG_PORT_VALUE}
ExecStart=${START_SCRIPT}
Restart=on-failure
RestartSec=30
StartLimitIntervalSec=300
StartLimitBurst=5
RestartPreventExitStatus=1
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now raganything
sleep 3
systemctl status raganything --no-pager || true

echo ""
echo "=== Health check ==="
curl -fsS "http://localhost:${RAG_PORT_VALUE}/health" | python3 -m json.tool
