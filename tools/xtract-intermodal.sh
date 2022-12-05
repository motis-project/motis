#!/bin/bash

set -x

./motis print fail/$1_intermodal-responses-{routing,nigiri}.json

rm -rf input/$1
mkdir input/$1
ln -s `pwd`/input/osm.pbf input/$1/osm.pbf

./motis xtract input/hrd input/$1 fail/$1_intermodal-responses-{routing,nigiri}.json

rm -rf input/$1/stamm
ln -s `pwd`/input/hrd/stamm input/$1/stamm

rm -rf data_$1
mkdir data_$1
ln -s `pwd`/data-full/osrm data_$1/osrm

./motis \
  -c config-intermodal.ini \
  --dataset.write_serialized=false \
  --dataset.cache_graph=false \
  --dataset.read_graph=false \
  --nigiri.no_cache=true \
  --import.paths schedule:input/$1 osm:input/osm.pbf \
  --import.data_dir data_$1 \
  --batch_input_file=fail/$1_intermodal-queries-routing.json \
  --batch_output_file=$1_intermodal_routing_response.json

./motis \
  -c config-intermodal.ini \
  --dataset.write_serialized=false \
  --dataset.cache_graph=false \
  --dataset.read_graph=false \
  --nigiri.no_cache=true \
  --import.paths schedule:input/$1 osm:input/osm.pbf \
  --import.data_dir data_$1 \
  --batch_input_file=fail/$1_intermodal-queries-nigiri.json \
  --batch_output_file=$1_intermodal_nigiri_response.json

DIR=`pwd`
cd /tmp
${DIR}/motis intermodal_compare \
    --fail "" \
    --queries \
        ${DIR}/fail/$1_intermodal-queries-routing.json \
        ${DIR}/fail/$1_intermodal-queries-nigiri.json \
    --input \
        ${DIR}/$1_intermodal_routing_response.json \
        ${DIR}/$1_intermodal_nigiri_response.json
cd $DIR

./motis print \
    $1_intermodal_routing_response.json \
    $1_intermodal_nigiri_response.json

./motis analyze $1_intermodal_routing_response.json 
./motis analyze $1_intermodal_nigiri_response.json