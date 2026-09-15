#!/bin/sh
set -e
/opt/motis/fetch-data.sh
exec "$@"
