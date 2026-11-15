#!/bin/bash
# Turns on SSL before the official Postgres entrypoint. The key is created in the data volume
# (mode 600, owned by postgres) and is not part of the git repo. A missing key is created once.
set -euo pipefail

ssldir=/var/lib/postgresql/ssl
mkdir -p "$ssldir"
if [[ ! -s "$ssldir/server.key" || ! -s "$ssldir/server.crt" ]]; then
  openssl req -new -x509 -days 3650 -nodes \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
    -keyout "$ssldir/server.key" -out "$ssldir/server.crt"
  chown postgres:postgres "$ssldir/server.key" "$ssldir/server.crt"
  chmod 600 "$ssldir/server.key"
  chmod 644 "$ssldir/server.crt"
fi

exec /usr/local/bin/docker-entrypoint.sh "$@" \
  -c ssl=on \
  -c "ssl_cert_file=${ssldir}/server.crt" \
  -c "ssl_key_file=${ssldir}/server.key"
