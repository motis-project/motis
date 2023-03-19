#!/bin/bash

set -x

./motis print fail_responses/$1_1.json fail_responses/$1_2.json

./motis rewrite --in=fail_queries/$1.json --out=fail_queries/$1_routing.json --target=/routing
./motis rewrite --in=fail_queries/$1.json --out=fail_queries/$1_nigiri.json --target=/nigiri

./motis xtract input/hrd input/$1 fail_responses/$1_1.json fail_responses/$1_2.json

rm -rf input/$1/stamm
ln -s `pwd`/input/hrd/stamm input/$1/stamm

./motis \
  --dataset.write_serialized=false \
  --dataset.cache_graph=false \
  --dataset.read_graph=false \
  --nigiri.no_cache=true \
  --import.paths schedule:input/$1 \
  --import.data_dir data_$1 \
  --batch_input_file=fail_queries/$1_routing.json \
  --batch_output_file=$1_routing_response.json

./motis \
  --dataset.write_serialized=false \
  --dataset.cache_graph=false \
  --dataset.read_graph=false \
  --nigiri.no_cache=true \
  --import.paths schedule:input/$1 \
  --import.data_dir data_$1 \
  --batch_input_file=fail_queries/$1_nigiri.json \
  --batch_output_file=$1_nigiri_response.json

DIR=`pwd`
cd /tmp
${DIR}/motis compare ${DIR}/$1_nigiri_response.json ${DIR}/$1_routing_response.json ${DIR}/fail_queries/$1_routing.json
cd $DIR

./motis print $1_nigiri_response.json $1_routing_response.json

./motis analyze $1_nigiri_response.json
./motis analyze $1_routing_response.json

./motis rewrite --in ${DIR}/$1_nigiri_response.json --out ${DIR}/$1_nigiri_response_check.json --target /cc
./motis rewrite --in ${DIR}/$1_routing_response.json --out ${DIR}/$1_routing_response_check.json --target /cc

./motis --modules cc --mode init --init ./$1_routing_response_check.json
./motis --modules cc --mode init --init ./$1_nigiri_response_check.json
