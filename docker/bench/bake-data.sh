#!/usr/bin/env bash
# Bakes a downloaded timetable into the benchmark image as extra layers and
# pushes the result -- with crane, no docker needed, so it runs wherever the
# data already is (the rented instance after fetch-data.sh: its uplink beats
# a home line by far).
#
#   bake-data.sh <data-dir> <base-image> <target-image> [parts]
#
#   e.g. GHCR_USER=felixguendling GHCR_TOKEN=$(gh auth token) \
#        bake-data.sh /data ghcr.io/felixguendling/motis-bench:cuda \
#                     ghcr.io/felixguendling/motis-bench:cuda-transitous-20260905
#
# <data-dir> is either what fetch-data.sh downloaded (tt.bin, tags.bin,
# config.upstream.yml) or the output directory of an own `motis import`
# (tt.bin, tags.bin, config.yml, meta/, plus queries.txt from motis
# generate); everything but tt.bin becomes one small layer. tt.bin goes in as <parts> layers, each
# under the registry's 10 GB cap on a *compressed* layer (Transitous' file
# gzips ~4:1, so 1 part = one ~5.5 GB layer that needs no join); with more
# parts fetch-data.sh joins them on first start. The slices are gzipped
# here (pigz if present) and handed to crane compressed, so the extra disk
# is one compressed slice, not a raw one.
#
# crane: https://github.com/google/go-containerregistry/releases -- e.g.
#   curl -sL https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz \
#     | tar -xz -C /usr/local/bin crane
# Login either beforehand (crane auth login ghcr.io -u USER -p TOKEN, token
# with write:packages) or via GHCR_USER/GHCR_TOKEN.
set -euo pipefail

data=${1:?usage: bake-data.sh <data-dir> <base-image> <target-image> [parts]}
base=${2:?usage: bake-data.sh <data-dir> <base-image> <target-image> [parts]}
target=${3:?usage: bake-data.sh <data-dir> <base-image> <target-image> [parts]}
parts=${4:-4}
here=$(cd "$(dirname "$0")" && pwd)

for f in tt.bin tags.bin; do
  [ -f "$data/$f" ] || { echo "missing $data/$f" >&2; exit 1; }
done
[ -f "$data/config.yml" ] || [ -f "$data/config.upstream.yml" ] || {
  echo "missing $data/config.yml (own import) or config.upstream.yml (download)" >&2
  exit 1
}
command -v crane >/dev/null || { echo "crane not on PATH (see header)" >&2; exit 1; }

if [ -n "${GHCR_TOKEN:-}" ]; then
  crane auth login "${target%%/*}" -u "${GHCR_USER:?GHCR_USER}" -p "$GHCR_TOKEN"
fi

tmp=$(mktemp -d "${TMPDIR:-/tmp}/bake.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
if [ -n "${GZ:-}" ]; then
  gz=$GZ
elif command -v pigz >/dev/null; then
  gz="pigz -1"
else
  gz="gzip -1"
fi

# everything but the timetable itself: tags.bin, the config, and for an own
# import meta/ (the hashes motis checks before loading) and queries.txt
echo "bake: metadata layer ($data minus tt.bin)"
tar -cf "$tmp/meta.tar" --owner=0 --group=0 --transform 's|^\./|data/|' \
  --exclude=./tt.bin --exclude='./tt.bin.part*' --exclude='./tt.bin.tmp' \
  -C "$data" .
crane append ${CRANE_OPTS:-} -b "$base" -f "$tmp/meta.tar" -t "$target"
rm -f "$tmp/meta.tar"

# one part fits the layer cap only if tt.bin compresses to < 10 GB -- then it
# ships under its final name and no join (= second copy on disk) is needed
for i in $(seq 0 $((parts - 1))); do
  p=$(printf '%02d' "$i")
  if [ "$parts" -eq 1 ]; then
    member=data/tt.bin
  else
    member="data/tt.bin.part$p"
  fi
  echo "bake: layer $((i + 1))/$parts ($member)"
  python3 "$here/part-tar.py" "$data/tt.bin" "$i" "$parts" "$member" | $gz > "$tmp/part.tar.gz"
  crane append ${CRANE_OPTS:-} -b "$target" -f "$tmp/part.tar.gz" -t "$target"
  rm -f "$tmp/part.tar.gz"
done

echo "bake: done -> $target"
crane manifest ${CRANE_OPTS:-} "$target" | python3 -c 'import json,sys; m=json.load(sys.stdin); print(len(m["layers"]), "layers,", sum(l["size"] for l in m["layers"])//2**20, "MB compressed")'
