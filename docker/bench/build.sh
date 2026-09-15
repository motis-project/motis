#!/usr/bin/env bash
# Builds the benchmark image from the repo root.
#
#   docker/bench/build.sh <image:tag> [--push] [extra docker build args...]
#
# e.g. docker/bench/build.sh ghcr.io/felixguendling/motis-bench:cuda --push
#      docker/bench/build.sh motis-bench:cuda --build-arg GCC=13
#      DOCKERFILE=Dockerfile.rocm docker/bench/build.sh motis-bench:rocm
# CONTEXT overrides the source tree (default: this repo), e.g. a worktree
# whose deps/nigiri is on another branch.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "${CONTEXT:-$here/../..}" && pwd)

image=${1:?usage: build.sh <image:tag> [--push] [docker build args...]}
shift
push=0
if [ "${1:-}" = "--push" ]; then
  push=1
  shift
fi

# a docker-container builder keeps the result in its cache unless told where
# to put it: straight to the registry, or into the local image store
if [ "$push" = 1 ]; then
  out=--push
else
  out=--load
fi

docker buildx build \
  -f "$here/${DOCKERFILE:-Dockerfile}" \
  --build-context bench="$here" \
  -t "$image" \
  "$out" \
  "$@" \
  "$root"
