#!/usr/bin/env bash
# Start the local Postgres container and apply sql/prometheus_accounts.sql.
# Passwords are written to secrets/postgres_local.txt the first time and reused after that.
#   bash scripts/setup_postgres.sh           # start, apply the schema, seed the three samples if they are missing
#   bash scripts/setup_postgres.sh --reseed  # also replace the three sample accounts (other accounts are kept)
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
secrets="$root/secrets/postgres_local.txt"
admin_file="$root/secrets/postgres_admin_password.txt"
mkdir -p "$root/secrets"
chmod 700 "$root/secrets"

if [[ ! -f "$secrets" ]]; then
  umask 077
  admin_password=$(openssl rand -hex 24)
  app_password=$(openssl rand -hex 24)
  cat > "$secrets" <<EOF
# Prometheus local Postgres. This file is gitignored. It is not an API key.
# The Account Manager reads the app_ lines. Do not copy them into .env.
# Created $(date -Iseconds). Image postgres:18, published on 127.0.0.1:5432 only.

host=127.0.0.1
port=5432
database=prometheus
admin_user=prometheus_admin
admin_password=${admin_password}
app_user=prometheus_app
app_password=${app_password}
sslmode=require
image=postgres:18
container=prometheus-postgres
EOF
  chmod 600 "$secrets"
  echo "wrote $secrets"
fi

value() {
  local key="$1"
  local found
  found=$(awk -F= -v k="$key" '$1 == k {print substr($0, index($0, "=") + 1); exit}' "$secrets")
  if [[ -z "$found" ]]; then
    echo "missing ${key} in ${secrets}" >&2
    exit 1
  fi
  printf '%s' "$found"
}

admin_user=$(value admin_user)
admin_password=$(value admin_password)
app_password=$(value app_password)
database=$(value database)
umask 077
printf '%s\n' "$admin_password" > "$admin_file"
chmod 600 "$admin_file" "$secrets"

docker compose up -d

status=starting
for _ in $(seq 1 60); do
  status=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' prometheus-postgres 2>/dev/null || true)
  if [[ "$status" == "healthy" ]]; then
    break
  fi
  sleep 1
done
if [[ "$status" != "healthy" ]]; then
  echo "postgres did not become healthy (last status: ${status})" >&2
  docker logs --tail 40 prometheus-postgres >&2 || true
  exit 1
fi

docker compose exec -T -e PGPASSWORD="$admin_password" postgres \
  psql -v ON_ERROR_STOP=1 -U "$admin_user" -d "$database" -v app_password="$app_password" \
  -f /sql/prometheus_accounts.sql

count=$(docker compose exec -T -e PGPASSWORD="$admin_password" postgres \
  psql -v ON_ERROR_STOP=1 -tA -U "$admin_user" -d "$database" \
  -c "SELECT count(*) FROM trading.accounts WHERE seed_key IS NOT NULL")
count=$(printf '%s' "$count" | tr -d '[:space:]')
if [[ "${1:-}" == "--reseed" || "$count" == "0" ]]; then
  docker compose exec -T -e PGPASSWORD="$admin_password" postgres \
    psql -v ON_ERROR_STOP=1 -U "$admin_user" -d "$database" -f /sql/seed_test_users.sql
  echo "sample accounts loaded (jacob_sample, avery_chen, blake_okonkwo)"
else
  echo "sample accounts already present (${count}); pass --reseed to replace those three"
fi

docker compose exec -T -e PGPASSWORD="$admin_password" postgres \
  psql -v ON_ERROR_STOP=1 -tA -U "$admin_user" -d "$database" -c "SELECT version()"
echo "passwords are in secrets/postgres_local.txt"
