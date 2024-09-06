project(motis)
cmake_minimum_required(VERSION 3.10)

set(protocol-dir ${CMAKE_CURRENT_SOURCE_DIR}/../../protocol)
file(GLOB_RECURSE protocol-files ${protocol-dir}/*.fbs)
set(generated-headers-dir "${CMAKE_BINARY_DIR}/generated/motis/protocol")
build_flatbuffers(
  32                           # num_bits
  "--everything-required"      # commandline_options
  "${protocol-files}"          # flatbuffers_schemas
  "${protocol-dir}"            # schema_include_dirs
  generated-protocol-headers   # custom_target_name
  ""                           # additional_dependencies
  "${generated-headers-dir}"   # generated_includes_dir
  ""                           # binary_schemas_dir
  ""                           # copy_text_schemas_dir
)

set(resource-header "${generated-headers-dir}/resources.h")
compile_flatbuffers_to_resource_header(
  "${protocol-dir}/Message.fbs"
  "${resource-header}"
  "motis::module"
  "${protocol-files}"
)

file(GLOB_RECURSE motis-module-files src/*.cc)
add_library(motis-module STATIC ${motis-module-files} ${resource-header})
target_include_directories(motis-module PUBLIC include)
target_compile_features(motis-module PUBLIC cxx_std_17)
add_dependencies(motis-module generated-protocol-headers)
target_link_libraries(motis-module
  motis-core
  http-client
  flatbuffers32
  rapidjson
  ctx
  conf
  wss-client
  cista
  utl
  boost-url
  opentelemetry_trace
)
target_compile_options(motis-module PRIVATE ${MOTIS_CXX_FLAGS})
