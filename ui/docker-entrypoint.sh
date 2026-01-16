#!/bin/sh
set -e

# Default backend URL if not set
export MOTIS_BACKEND_URL=${MOTIS_BACKEND_URL:-http://localhost:80}

# Substitute environment variables in nginx config
envsubst '${MOTIS_BACKEND_URL}' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf

# Start nginx
exec nginx -g "daemon off;"

