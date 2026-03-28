#!/bin/sh
set -e

export MOTIS_BACKEND_URL=${MOTIS_BACKEND_URL:-http://localhost:80}

# MapTiler at runtime (no secret baked into the JS bundle).
MAPTILER_API_KEY_VAL="${MAPTILER_API_KEY:-${VITE_MAPTILER_API_KEY:-}}"
MAPTILER_STYLE_VAL="${MAPTILER_STYLE:-${VITE_MAPTILER_STYLE:-openstreetmap}}"
printf 'window.__MOTIS_CONFIG__ = %s;\n' "$(jq -n \
	--arg k "$MAPTILER_API_KEY_VAL" \
	--arg s "$MAPTILER_STYLE_VAL" \
	'{maptilerApiKey:$k, maptilerStyle:$s}')" \
	> /usr/share/nginx/html/runtime-config.js

# Process the shared locations snippet (envsubst for ${MOTIS_BACKEND_URL})
mkdir -p /etc/nginx/snippets
envsubst '${MOTIS_BACKEND_URL}' < /etc/nginx/snippets/motis-locations.conf.template \
  > /etc/nginx/snippets/motis-locations.conf

# Generate the server config: if SSL certs are present, enable the HTTPS server;
# otherwise fall back to HTTP-only (no redirect, no 443 listener).
if [ -f /etc/nginx/ssl/fullchain.pem ] && [ -f /etc/nginx/ssl/privkey.pem ]; then
  envsubst '' < /etc/nginx/templates/default.conf.template \
    > /etc/nginx/conf.d/default.conf
else
  # No SSL: serve directly on port 80 without redirect
  cat > /etc/nginx/conf.d/default.conf <<'NOSSL'
server {
    listen 80;
    server_name _;
    include /etc/nginx/snippets/motis-locations.conf;
}
NOSSL
fi

exec nginx -g "daemon off;"
