#!/usr/bin/env bash
# Restrict the Skyview origin port to the IPv64 CDN64 edge servers.
#
# Without this the app on :8501 is reachable by anyone who knows the VPS
# address: plain HTTP, bypassing the CDN's TLS, WAF and per-IP limits.
#
# What it does (idempotent, safe to re-run):
#   1. allows SSH first, on every port sshd is actually listening on
#   2. allows tcp/8501 from the addresses in cdn64-edge-ips.txt
#   3. removes any "allow 8501 from anywhere" rule
#   4. sets "deny incoming / allow outgoing" and enables ufw
#   5. removes origin rules for addresses no longer in that file
#
# Loopback is never filtered by ufw, so the deploy health check and local
# curl against 127.0.0.1:8501 keep working.
#
# Usage:  sudo bash deploy/setup-firewall.sh [path/to/cdn64-edge-ips.txt]
# Undo:   sudo ufw disable
set -euo pipefail

PORT="${SKYVIEW_PORT:-8501}"
TAG="skyview-origin"
IPS_FILE="${1:-$(cd "$(dirname "$0")" && pwd)/cdn64-edge-ips.txt}"

die() { echo "[firewall] ERROR: $*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "must run as root"
command -v ufw >/dev/null || die "ufw is not installed (apt-get install ufw)"
[ -r "$IPS_FILE" ] || die "cannot read $IPS_FILE"

mapfile -t EDGE_IPS < <(sed 's/#.*//' "$IPS_FILE" | awk 'NF {print $1}')
[ "${#EDGE_IPS[@]}" -gt 0 ] || die "no addresses in $IPS_FILE - refusing to lock the origin out"

# --- 1. SSH, before anything that could block it --------------------------
# Ask sshd for its configured ports and add whatever is listening right now;
# always include 22 as a floor so a parsing failure cannot drop SSH.
mapfile -t SSH_PORTS < <(
  {
    echo 22
    sshd -T 2>/dev/null | awk '$1 == "port" {print $2}'
    ss -ltnpH 2>/dev/null | awk '/sshd/ {n = split($4, a, ":"); print a[n]}'
  } | grep -E '^[0-9]+$' | sort -un
)
for p in "${SSH_PORTS[@]}"; do
  ufw allow "${p}/tcp" comment 'ssh' >/dev/null
  echo "[firewall] ssh allowed on tcp/${p}"
done

# --- 2. origin port: edge servers only ------------------------------------
for ip in "${EDGE_IPS[@]}"; do
  ufw allow proto tcp from "$ip" to any port "$PORT" comment "$TAG" >/dev/null
done
echo "[firewall] tcp/${PORT} allowed from ${#EDGE_IPS[@]} edge addresses"

# --- 3. no blanket rule for the origin port --------------------------------
# An "allow 8501 from anywhere" rule would defeat the allowlist. Deleting by
# rule text works whether or not ufw is active; a missing rule is not an error.
for spec in "${PORT}/tcp" "${PORT}"; do
  ufw delete allow "$spec" >/dev/null 2>&1 || true
done

# --- 4. policy + enable -----------------------------------------------------
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
ufw --force enable >/dev/null

# --- 5. drop origin rules whose source is no longer listed -----------------
# Done last: "ufw status numbered" only lists rules while ufw is active, and
# the current edge addresses were already allowed above, so the edge is never
# cut off. Delete from the highest rule number down (numbers shift on delete).
while read -r num src; do
  keep=0
  for ip in "${EDGE_IPS[@]}"; do [ "$ip" = "$src" ] && keep=1 && break; done
  if [ "$keep" -eq 0 ]; then
    ufw --force delete "$num" >/dev/null
    echo "[firewall] removed stale origin rule for $src"
  fi
done < <(
  ufw status numbered 2>/dev/null \
    | grep -F "# $TAG" \
    | sed -E 's/^\[ *([0-9]+)\] .* ALLOW IN +([^ ]+).*/\1 \2/' \
    | sort -rn
)

echo "[firewall] active:"
ufw status verbose | sed 's/^/[firewall]   /'
