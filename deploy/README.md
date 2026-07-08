# LeanSwarm — Deployment

One-command deployment for an Oracle Cloud AMD free-tier instance (Ubuntu
22.04+, 1 OCPU, 1 GB RAM).

## Prerequisites

- **Oracle Cloud AMD free-tier instance** running Ubuntu 22.04+
- **DNS A record** for `leanswarm.me` pointing to the instance's public IP
- **Ports 80 and 443** open in the VCN security list (ingress)
- **GitHub deploy key** on the server with read access to this repo

## Quick Start

```bash
# 1. SSH into your instance
ssh ubuntu@<your-instance-ip>

# 2. Clone the repo
git clone https://github.com/mohith-das/leanswarm.git
cd leanswarm

# 3. Run the setup script
sudo bash deploy/scripts/setup.sh
```

The script is idempotent — safe to re-run at any time.

## What gets installed

| Component   | Details                                            |
|-------------|----------------------------------------------------|
| **Python**  | 3.12 venv at `/home/ubuntu/leanswarm-venv`         |
| **Nginx**   | Reverse proxy, terminates TLS (certbot)            |
| **Systemd** | `leanswarm.service` — runs `leanswarm ui :8000`    |
| **Updater** | `leanswarm-update.timer` — checks PyPI daily, auto-restarts on new version |
| **Landing** | Static HTML at `/var/www/leanswarm/index.html`     |

## Architecture

```
Browser ── HTTPS ──► Nginx (:443) ── / ──► /var/www/leanswarm/index.html
                            │
                            └── /* ──► leanswarm ui (127.0.0.1:8000)
```

- `/` serves the static landing page directly from nginx.
- `/login`, `/api/*`, `/dashboard`, and everything else proxy to the
  `leanswarm ui` backend.
- LLM API keys are entered by each user in-browser — the server stores
  none.

## CI/CD

Push to `main` triggers a [GitHub Actions workflow](../.github/workflows/deploy.yml) that
SSHs into the instance, pulls the repo, and re-runs `setup.sh`.

**Required secrets** (set in repo Settings → Secrets and variables → Actions):

| Secret            | Description                              |
|-------------------|------------------------------------------|
| `SSH_HOST`        | Instance IP (e.g. `92.4.87.247`)         |
| `SSH_USERNAME`    | SSH user (e.g. `ubuntu`)                 |
| `SSH_KEY`         | Private SSH key for the instance         |
| `CERTBOT_EMAIL`   | Email for Let's Encrypt registration     |

The server also needs a GitHub deploy key at `~/.ssh/github_deploy` (added once
via `gh repo deploy-key add`).

## Useful commands

```bash
sudo systemctl status leanswarm               # check the service
sudo systemctl status leanswarm-update.timer   # check auto-update timer
sudo journalctl -u leanswarm -f                # tail app logs
sudo journalctl -u leanswarm-update            # check update logs
sudo systemctl reload nginx                    # reload nginx config
sudo certbot renew --dry-run                   # test cert auto-renewal
```
