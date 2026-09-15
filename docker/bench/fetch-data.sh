#!/bin/sh
# Downloads an imported timetable into DATA_DIR unless it is already there,
# writes the timetable-only config next to it and generates a fixed query set.
#
#   DATA_URL     base URL serving tt.bin, tags.bin and config.yml
#                (default: Transitous' nightly import, set in the Dockerfile)
#   DATA_DIR     target directory (default /data)
#   GPU_STATES   server.gpu_states for motis batch (default: motis' default)
#   N_QUERIES    queries.txt size for motis batch, 0 to skip (default 1000)
#
# vast.ai replaces the image entrypoint in ssh launch mode, so run this from
# the template's onstart there. Idempotent; resumes interrupted downloads.
set -e
: "${DATA_URL:=https://api.transitous.org/gtfs/data}"
: "${DATA_DIR:=/data}"
: "${N_QUERIES:=1000}"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# an image with the timetable baked in ships it as tt.bin.part* (registries
# cap a layer at 10 GB): join them once, then nothing below touches the network
if [ ! -f tt.bin ] && ls tt.bin.part* >/dev/null 2>&1; then
  echo "fetch: joining tt.bin.part* -> tt.bin"
  cat tt.bin.part* > tt.bin.tmp
  mv tt.bin.tmp tt.bin
  rm -f tt.bin.part*
fi

fetch() {
  if [ -f "$2" ]; then
    echo "fetch: $2 present"
    return
  fi
  echo "fetch: $DATA_URL/$1 -> $2"
  curl -fL --retry 5 --retry-all-errors -C - -o "$2.part" "$DATA_URL/$1"
  mv "$2.part" "$2"
}

# a data dir from our own import already has the stripped config (and meta/)
if [ ! -f config.yml ]; then
  fetch config.yml config.upstream.yml
fi
fetch tags.bin tags.bin
fetch tt.bin tt.bin

if [ ! -f config.yml ]; then
  # shellcheck disable=SC2086
  python3 /opt/motis/strip-config.py config.upstream.yml config.yml \
    ${GPU_STATES:+--gpu-states "$GPU_STATES"}
fi

if [ ! -f queries.txt ] && [ "$N_QUERIES" -gt 0 ]; then
  echo "fetch: generating $N_QUERIES queries"
  /opt/motis/motis generate -d "$DATA_DIR" -n "$N_QUERIES"
fi

echo "fetch: ready"
ls -la "$DATA_DIR"
