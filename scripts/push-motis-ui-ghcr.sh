#!/usr/bin/env bash
set -euo pipefail

# Build ui/Dockerfile and tag for GitHub Container Registry.
# Default: ghcr.io/flowride/motis-ui (owner from git config github.organization, env GHCR_OWNER, or flowride).
#
# Login (personal GitHub username + PAT write:packages):
#   echo "$GITHUB_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
#
# Usage:
#   ./scripts/push-motis-ui-ghcr.sh           # build + tag only
#   ./scripts/push-motis-ui-ghcr.sh --push  # build + tag + docker push

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="${ROOT}/ui"
cd "${UI_DIR}"

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
GHCR_IMAGE="${GHCR_IMAGE:-motis-ui}"

if [[ ! -f "${UI_DIR}/openapi.yaml" ]]; then
  if [[ -f "${ROOT}/openapi.yaml" ]]; then
    echo "Copie de openapi.yaml depuis la racine du dépôt MOTIS..."
    cp "${ROOT}/openapi.yaml" "${UI_DIR}/openapi.yaml"
  else
    echo "Erreur: openapi.yaml introuvable (${UI_DIR}/openapi.yaml ou ${ROOT}/openapi.yaml)" >&2
    exit 1
  fi
fi

SHORT="$(git -C "${ROOT}" rev-parse --short HEAD)"
REF="${GHCR_OWNER}/${GHCR_IMAGE}"
FULL_BASE="ghcr.io/${REF}"

echo "Build MOTIS UI → ${FULL_BASE}:latest et :${SHORT} (owner=${GHCR_OWNER})"
docker build -f Dockerfile -t "${FULL_BASE}:${SHORT}" -t "${FULL_BASE}:latest" .

if [[ "${DO_PUSH}" == "true" ]]; then
  docker push "${FULL_BASE}:latest"
  docker push "${FULL_BASE}:${SHORT}"
  echo "Poussé: ${FULL_BASE}:latest, ${FULL_BASE}:${SHORT}"
else
  echo "Sans --push. Pour publier: $0 --push"
fi
