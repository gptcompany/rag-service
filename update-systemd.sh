#!/bin/bash
# Update RAGAnything systemd unit to point to new rag-service location
# Run: sudo bash /media/sam/1TB/rag-service/update-systemd.sh

set -e

cat > /etc/systemd/system/raganything.service <<'EOF'
[Unit]
Description=RAGanything Service - Full RAG for Academic Papers
After=network.target

[Service]
Type=simple
User=sam
WorkingDirectory=/media/sam/1TB/rag-service
ExecStart=/media/sam/1TB/rag-service/scripts/raganything_start.sh

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
echo "Done! RAGAnything now runs from /media/sam/1TB/rag-service/"
