#!/usr/bin/env bash
set -euo pipefail

docker compose build motis-ui
docker compose up -d --no-build
