#!/bin/bash

set -euo pipefail
set -o xtrace

SCRIPT_DIR=`dirname "$0"`
BUILD=cmake-build-relwithdebinfo

GTFS=de_DELFI.gtfs.zip
OSM=germany-latest.osm.pbf
ASSISTANCE=assistance.csv
FASTA=fasta.json

# wget https://gist.githubusercontent.com/felixguendling/43936516154ac6b4a9dfd6940db60f63/raw/4b070243498023740c03c0b7f924399884e69268/assistance.csv

mkdir -p $BUILD
pushd $BUILD
cmake .
ninja osr-extract \
	motis-prepare \
	nigiri-import \
	adr-extract \
	motis-adr-extend \
	motis-server
popd

rm -rf data
mkdir -p data

$BUILD/deps/nigiri/nigiri-import --assistance $ASSISTANCE $GTFS -o data/tt.bin
$BUILD/deps/osr/osr-extract -p true -i $OSM -o data/osr
$BUILD/deps/adr/adr-extract -i $OSM -o data/adr
$BUILD/motis-adr-extend
$BUILD/motis-prepare
mv out/* data
rm -rf out

wget https://gist.githubusercontent.com/felixguendling/7f6e839ea02ea5881feca1990496badb/raw/9ed181546c17e230ed8c3b34d81bab3edf630774/fasta.json -O data/fasta.json

$BUILD/motis-server
