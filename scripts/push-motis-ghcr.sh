#!/usr/bin/env bash
set -euo pipefail

# Build Dockerfile.runtime and tag for GitHub Container Registry.
# Default: ghcr.io/flowride/motis (owner from git config github.organization, env GHCR_OWNER, or flowride).
#
# Login (use your personal GitHub username + PAT with write:packages, not the org name):
#   echo "$GITHUB_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
#
# Usage:
#   ./scripts/push-motis-ghcr.sh           # build + tag only
#   ./scripts/push-motis-ghcr.sh --push  # build + tag + docker push

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DO_PUSH=false
for a in "$@"; do
  case "$a" in
    --push) DO_PUSH=true ;;
    -h | --help)
      sed -n '1,25p' "$0"
      exit 0
      ;;
  esac
done

GHCR_OWNER="${GHCR_OWNER:-}"
if [[ -z "${GHCR_OWNER}" ]]; then
  GHCR_OWNER="$(git config --global --get github.organization 2>/dev/null || true)"
fi
GHCR_OWNER="${GHCR_OWNER:-flowride}"
GHCR_IMAGE="${GHCR_IMAGE:-motis}"

BIN="${ROOT}/build/docker-relwithdebinfo/motis"
if [[ ! -x "${BIN}" ]]; then
  echo "Erreur: binaire MOTIS introuvable ou non exécutable: ${BIN}" >&2
  echo "Compile d'abord (ex. docker compose -f docker-compose.build.yml run --rm motis-build)." >&2
  exit 1
fi

SHORT="$(git rev-parse --short HEAD)"
REF="${GHCR_OWNER}/${GHCR_IMAGE}"
FULL_BASE="ghcr.io/${REF}"

echo "Build runtime image → ${FULL_BASE}:latest et :${SHORT} (owner=${GHCR_OWNER})"
docker build -f Dockerfile.runtime -t "${FULL_BASE}:${SHORT}" -t "${FULL_BASE}:latest" .

if [[ "${DO_PUSH}" == "true" ]]; then
  docker push "${FULL_BASE}:latest"
  docker push "${FULL_BASE}:${SHORT}"
  echo "Poussé: ${FULL_BASE}:latest, ${FULL_BASE}:${SHORT}"
else
  echo "Sans --push. Pour publier: $0 --push"
fi
