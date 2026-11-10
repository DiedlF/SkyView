#!/usr/bin/env bash
# Restrict the Skyview origin ports to the IPv64 CDN64 edge servers.
#
# Without this the app is reachable by anyone who knows the VPS address,
# bypassing the CDN's TLS, WAF and per-IP limits.
#
# What it does (idempotent, safe to re-run):
#   1. allows SSH first, on every port sshd is actually listening on
#   2. allows each origin port (SKYVIEW_ORIGIN_PORTS, default "8501 8443")
#      from the addresses in cdn64-edge-ips.txt
#   3. removes any "allow <origin port> from anywhere" rule
#   4. sets "deny incoming / allow outgoing" and enables ufw
#   5. removes origin rules for addresses or ports no longer configured
#
# Loopback is never filtered by ufw, so the deploy health check and local
# curl against 127.0.0.1:8501 keep working.
#
# Usage:  sudo [SKYVIEW_ORIGIN_PORTS="8443"] bash deploy/setup-firewall.sh [path/to/cdn64-edge-ips.txt]
# Undo:   sudo ufw disable
set -euo pipefail

read -r -a PORTS <<< "${SKYVIEW_ORIGIN_PORTS:-8501 8443}"
TAG="skyview-origin"
IPS_FILE="${1:-$(cd "$(dirname "$0")" && pwd)/cdn64-edge-ips.txt}"

die() { echo "[firewall] ERROR: $*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "must run as root"
command -v ufw >/dev/null || die "ufw is not installed (apt-get install ufw)"
[ -r "$IPS_FILE" ] || die "cannot read $IPS_FILE"

mapfile -t EDGE_IPS < <(sed 's/#.*//' "$IPS_FILE" | awk 'NF {print $1}')
[ "${#EDGE_IPS[@]}" -gt 0 ] || die "no addresses in $IPS_FILE - refusing to lock the origin out"
[ "${#PORTS[@]}" -gt 0 ] || die "SKYVIEW_ORIGIN_PORTS is empty"
for p in "${PORTS[@]}"; do [[ "$p" =~ ^[0-9]+$ ]] || die "bad origin port: $p"; done

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

# --- 2. origin ports: edge servers only -----------------------------------
for port in "${PORTS[@]}"; do
  for ip in "${EDGE_IPS[@]}"; do
    ufw allow proto tcp from "$ip" to any port "$port" comment "$TAG" >/dev/null
  done
  echo "[firewall] tcp/${port} allowed from ${#EDGE_IPS[@]} edge addresses"
done

# --- 3. no blanket rule for an origin port ---------------------------------
# An "allow <port> from anywhere" rule would defeat the allowlist. Deleting by
# rule text works whether or not ufw is active; a missing rule is not an error.
for port in 8501 8443 "${PORTS[@]}"; do
  for spec in "${port}/tcp" "${port}"; do
    ufw delete allow "$spec" >/dev/null 2>&1 || true
  done
done

# --- 4. policy + enable -----------------------------------------------------
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
ufw --force enable >/dev/null

# --- 5. drop origin rules whose source or port is no longer configured ------
# Done last: "ufw status numbered" only lists rules while ufw is active, and
# the current edge addresses were already allowed above, so the edge is never
# cut off. Delete from the highest rule number down (numbers shift on delete).
while read -r num port src; do
  ip_ok=0; port_ok=0
  for ip in "${EDGE_IPS[@]}"; do [ "$ip" = "$src" ] && ip_ok=1 && break; done
  for p in "${PORTS[@]}"; do [ "$p" = "$port" ] && port_ok=1 && break; done
  if [ "$ip_ok" -eq 0 ] || [ "$port_ok" -eq 0 ]; then
    ufw --force delete "$num" >/dev/null
    echo "[firewall] removed stale origin rule tcp/${port} from $src"
  fi
done < <(
  ufw status numbered 2>/dev/null \
    | grep -F "# $TAG" \
    | sed -E 's/^\[ *([0-9]+)\] +([0-9]+)(\/tcp)? +ALLOW IN +([^ ]+).*/\1 \2 \4/' \
    | sort -rn
)

# --- 6. warn about a second, independent nftables firewall -----------------
# The box also has /etc/nftables.conf (policy drop, own port list). nftables
# evaluates every base chain, so a port must be allowed there too, and its
# "flush ruleset" wipes ufw's rules if nftables.service loads after ufw at boot.
if systemctl is-enabled nftables >/dev/null 2>&1 && grep -q "flush ruleset" /etc/nftables.conf 2>/dev/null; then
  echo "[firewall] WARNING: nftables.service is enabled and /etc/nftables.conf flushes the ruleset;"
  echo "[firewall]          origin ports must also be allowed there, and a reboot may drop ufw's rules."
  for port in "${PORTS[@]}"; do
    grep -Eq "dport ${port}( |$)" /etc/nftables.conf || echo "[firewall] WARNING: tcp/${port} is not allowed in /etc/nftables.conf"
  done
fi

echo "[firewall] active:"
ufw status verbose | sed 's/^/[firewall]   /'
