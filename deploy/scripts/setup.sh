#!/usr/bin/env bash
set -euo pipefail

# ── LeanSwarm deployment bootstrap ──────────────────────────────────────────
# Idempotent first-run script for an Oracle Cloud AMD free-tier instance
# (Ubuntu 22.04+, 1 OCPU, 1 GB RAM). Safe to re-run — each step checks
# whether it has already been done before acting.
#
# Usage:  sudo bash scripts/setup.sh
# ────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[setup]${NC} $*"; }
err()  { echo -e "${RED}[setup]${NC} $*" >&2; }

# Ensure we are root
if [ "$(id -u)" -ne 0 ]; then
    err "This script must be run as root (or with sudo)."
    exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="/home/ubuntu/leanswarm-venv"
WWW_DIR="/var/www/leanswarm"
NGINX_CONF="$REPO_DIR/nginx/leanswarm.me.conf"
TARGET_CONF="/etc/nginx/sites-available/leanswarm.me.conf"
SITES_ENABLED="/etc/nginx/sites-enabled/leanswarm.me.conf"
SYSTEMD_UNIT="$REPO_DIR/systemd/leanswarm.service"
SYSTEMD_TARGET="/etc/systemd/system/leanswarm.service"
SYSTEMD_UPDATE_SERVICE="$REPO_DIR/systemd/leanswarm-update.service"
SYSTEMD_UPDATE_TARGET="/etc/systemd/system/leanswarm-update.service"
SYSTEMD_UPDATE_TIMER="$REPO_DIR/systemd/leanswarm-update.timer"
SYSTEMD_UPDATE_TIMER_TARGET="/etc/systemd/system/leanswarm-update.timer"

# ── 1. System packages ──────────────────────────────────────────────────────
log "1/9  Installing system packages..."

apt update -qq

PACKAGES=(python3.12-venv nginx certbot python3-certbot-nginx git)
MISSING=()
for pkg in "${PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        MISSING+=("$pkg")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    log "Installing: ${MISSING[*]}"
    apt install -y "${MISSING[@]}"
else
    log "All packages already installed."
fi

# ── 2. Python venv + pip install ────────────────────────────────────────────
log "2/9  Setting up Python virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
    python3.12 -m venv "$VENV_DIR"
    chown -R ubuntu:ubuntu "$VENV_DIR"
    log "Created venv at $VENV_DIR"
else
    log "Venv already exists at $VENV_DIR"
    # Ensure ubuntu owns everything — previous pip runs as root may have
    # left root-owned .pyc files that break the timer's pip (running as ubuntu).
    chown -R ubuntu:ubuntu "$VENV_DIR"
fi

log "Installing / upgrading leanswarm..."
sudo -u ubuntu "$VENV_DIR/bin/pip" install --upgrade leanswarm

# ── 3. Nginx config ─────────────────────────────────────────────────────────
log "3/9  Installing nginx configuration..."

# Only overwrite if certbot hasn't already added the HTTPS block.
# If the target file contains 'listen 443 ssl', certbot has been here.
if grep -q 'listen 443 ssl' "$TARGET_CONF" 2>/dev/null; then
    log "Nginx config already has SSL block (certbot-modified) — skipping copy."
else
    cp "$NGINX_CONF" "$TARGET_CONF"
    log "Installed nginx config."
fi

if [ ! -L "$SITES_ENABLED" ]; then
    ln -sf "$TARGET_CONF" "$SITES_ENABLED"
    log "Symlinked $TARGET_CONF → $SITES_ENABLED"
else
    log "Nginx symlink already exists."
fi

# Remove the default site if it is still enabled
if [ -L /etc/nginx/sites-enabled/default ]; then
    rm -f /etc/nginx/sites-enabled/default
    log "Removed default nginx site."
fi

# ── 4. Landing page ─────────────────────────────────────────────────────────
log "4/9  Installing landing page..."

mkdir -p "$WWW_DIR"
cp "$REPO_DIR/landing/index.html" "$WWW_DIR/index.html"
chown -R ubuntu:ubuntu "$WWW_DIR"
log "Landing page installed at $WWW_DIR"

# ── 5. systemd units ─────────────────────────────────────────────────────────
log "5/9  Installing systemd units..."

cp "$SYSTEMD_UNIT" "$SYSTEMD_TARGET"
cp "$SYSTEMD_UPDATE_SERVICE" "$SYSTEMD_UPDATE_TARGET"
cp "$SYSTEMD_UPDATE_TIMER" "$SYSTEMD_UPDATE_TIMER_TARGET"
systemctl daemon-reload

if ! systemctl is-enabled leanswarm.service &>/dev/null; then
    systemctl enable leanswarm.service
    log "leanswarm.service enabled."
else
    log "leanswarm.service already enabled."
fi

if ! systemctl is-enabled leanswarm-update.timer &>/dev/null; then
    systemctl enable --now leanswarm-update.timer
    log "leanswarm-update.timer enabled (daily PyPI check)."
else
    log "leanswarm-update.timer already enabled."
fi

# ── 6. SSL certificate via certbot ──────────────────────────────────────────
log "6/9  Checking SSL certificate..."

CERT_DIR="/etc/letsencrypt/live/leanswarm.me"
if [ -d "$CERT_DIR" ]; then
    log "SSL certificate already exists at $CERT_DIR"
else
    # Open ports 80 and 443 if iptables is blocking them (Oracle Cloud
    # images use iptables by default; cloud-level VCN rules are separate).
    if command -v iptables &>/dev/null; then
        for PORT in 80 443; do
            if ! iptables -C INPUT -p tcp --dport "$PORT" -j ACCEPT 2>/dev/null; then
                iptables -I INPUT 5 -p tcp --dport "$PORT" -j ACCEPT
                log "Opened iptables port $PORT"
            fi
        done
        if command -v netfilter-persistent &>/dev/null; then
            netfilter-persistent save 2>/dev/null || true
        fi
    fi
    warn ""
    warn "Certbot needs to obtain an SSL certificate for leanswarm.me."
    warn "Make sure the DNS A record for leanswarm.me points to this"
    warn "machine's public IP before continuing."
    warn ""

    if [ -n "${CERTBOT_EMAIL:-}" ]; then
        log "Non-interactive mode: CERTBOT_EMAIL is set."
    else
        read -r -p "Ready to proceed with certbot? [y/N] " RESPONSE
        if [[ ! "$RESPONSE" =~ ^[Yy]$ ]]; then
            err "Aborted. Re-run this script after pointing DNS."
            exit 1
        fi
    fi

    if [ -n "${CERTBOT_EMAIL:-}" ]; then
        CERT_EMAIL="$CERTBOT_EMAIL"
        log "Using CERTBOT_EMAIL from environment."
    else
        read -r -p "Email address for Let's Encrypt notifications: " CERT_EMAIL
    fi
    if [ -z "$CERT_EMAIL" ]; then
        err "Email is required for certificate registration."
        exit 1
    fi

    certbot --nginx -d leanswarm.me --non-interactive --agree-tos -m "$CERT_EMAIL"
    log "SSL certificate obtained and nginx reconfigured."
fi

# ── 7. Start / reload services ──────────────────────────────────────────────
log "7/9  Starting services..."

systemctl restart leanswarm.service
nginx -t && systemctl reload nginx

# ── 8. Done ─────────────────────────────────────────────────────────────────
log "8/9  Deployment complete!"
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  LeanSwarm is live at  https://leanswarm.me${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo ""
# ── 9. Auto-update status ────────────────────────────────────────────────────
log "9/9  Auto-updater..."
log "  Daily PyPI update check is active via leanswarm-update.timer"
log "  Check timer:  sudo systemctl status leanswarm-update.timer"
log "  Check logs:   sudo journalctl -u leanswarm-update"
log ""
log "Useful commands:"
log "  sudo systemctl status leanswarm"
log "  sudo journalctl -u leanswarm -f"
log "  sudo systemctl status leanswarm-update.timer"
log "  sudo systemctl reload nginx"
