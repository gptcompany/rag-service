#!/bin/bash
# Update RAGAnything systemd unit to point to rag-service location
# Run: sudo bash /path/to/rag-service/update-systemd.sh
#
# Override SERVICE_ROOT via env var or edit default below:
#   RAG_SERVICE_ROOT=/opt/rag-service sudo bash update-systemd.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_ROOT="${RAG_SERVICE_ROOT:-$SCRIPT_DIR}"
SERVICE_USER="${RAG_SERVICE_USER:-$(whoami)}"

cat > /etc/systemd/system/raganything.service <<EOF
[Unit]
Description=RAGanything Service - Full RAG for Academic Papers
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${SERVICE_ROOT}
ExecStart=${SERVICE_ROOT}/scripts/raganything_start.sh

# Anti-loop protection
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
systemctl restart raganything
sleep 3
systemctl status raganything --no-pager

echo ""
echo "=== Health check ==="
curl -s http://localhost:8767/health | python3 -m json.tool

echo ""
echo "Done! RAGAnything now runs from ${SERVICE_ROOT}/"
