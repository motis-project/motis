#!/usr/bin/env bash
set -euo pipefail

# 1. Compile in build container
export MOTIS_UID="$(id -u)"
export MOTIS_GID="$(id -g)"
export SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-/tmp/ssh-agent}"
# rm -rf build/docker-relwithdebinfo
docker compose -f docker-compose.build.yml build motis-build
docker compose -f docker-compose.build.yml run --rm motis-build

# 2. Create runtime image from compiled binary
docker build --platform linux/amd64 -f Dockerfile.runtime -t motis:latest .

# 3. Run
docker compose up -d --no-build

# 4. Stop
docker compose down
