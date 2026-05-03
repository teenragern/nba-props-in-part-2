#!/usr/bin/env bash
# One-shot VPS setup: create user, deploy code, install systemd service.
# Run as root on a fresh Ubuntu/Debian box.
#
# Usage:
#   sudo bash deploy/install.sh

set -euo pipefail

DEPLOY_DIR="/opt/nba-prop-bot"
SERVICE_NAME="nba-prop-bot"
PYTHON="python3"

echo "==> Creating system user: propbot"
id propbot &>/dev/null || useradd --system --shell /usr/sbin/nologin --create-home propbot

echo "==> Copying files to $DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"
rsync -av --exclude='.git' --exclude='venv' --exclude='backups' --exclude='*.db' \
    "$(dirname "$(realpath "$0")")/../" "$DEPLOY_DIR/"

echo "==> Creating virtualenv and installing dependencies"
$PYTHON -m venv "$DEPLOY_DIR/venv"
"$DEPLOY_DIR/venv/bin/pip" install --quiet --upgrade pip
"$DEPLOY_DIR/venv/bin/pip" install --quiet -r "$DEPLOY_DIR/requirements.txt"

echo "==> Creating backup directory"
mkdir -p "$DEPLOY_DIR/backups"
chown -R propbot:propbot "$DEPLOY_DIR"

echo "==> Installing systemd unit"
cp "$DEPLOY_DIR/deploy/$SERVICE_NAME.service" "/etc/systemd/system/$SERVICE_NAME.service"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo ""
echo "Done. Before starting:"
echo "  1. Copy your .env file to $DEPLOY_DIR/.env"
echo "  2. Run: systemctl start $SERVICE_NAME"
echo "  3. Check logs: journalctl -u $SERVICE_NAME -f"
