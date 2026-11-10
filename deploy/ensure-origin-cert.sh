#!/usr/bin/env bash
# Create the origin TLS certificate for the CDN64 edge -> origin hop, once.
#
# Self-signed on purpose: the edge connects with "Verify the target's
# certificate" off, so the certificate's job is encryption between the edge
# servers and this box (admin Basic auth, marker tokens), not identity. A public
# CA certificate cannot be issued for the origin anyway - its public name points
# at the edge servers.
#
# Idempotent: an existing certificate is kept unless it expires within 30 days.
# Usage: sudo bash deploy/ensure-origin-cert.sh
set -euo pipefail

DIR="${SKYVIEW_TLS_DIR:-/etc/skyview/tls}"
CRT="$DIR/origin.crt"
KEY="$DIR/origin.key"

if [ -s "$CRT" ] && [ -s "$KEY" ] && openssl x509 -in "$CRT" -noout -checkend $((30 * 86400)) >/dev/null 2>&1; then
  echo "[origin-cert] valid until $(openssl x509 -in "$CRT" -noout -enddate | cut -d= -f2)"
  exit 0
fi

install -d -m 0700 "$DIR"
umask 077
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes \
  -days 3650 -subj "/CN=skyview-origin" \
  -keyout "$KEY.new" -out "$CRT.new" 2>/dev/null
mv -f "$KEY.new" "$KEY"
mv -f "$CRT.new" "$CRT"
chmod 600 "$KEY"
chmod 644 "$CRT"
echo "[origin-cert] created $CRT (valid until $(openssl x509 -in "$CRT" -noout -enddate | cut -d= -f2))"
