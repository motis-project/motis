cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

set(motis-parking-dbschema-dir ${CMAKE_CURRENT_SOURCE_DIR}/dbschema)
set(motis-parking-generated-headers-dir "${CMAKE_BINARY_DIR}/generated/motis/parking/dbschema")
file(GLOB_RECURSE motis-parking-dbschema-files ${motis-parking-dbschema-dir}/*.fbs)

build_flatbuffers(
  32                                          # num_bits
  "--everything-required"                     # commandline_options
  "${motis-parking-dbschema-files}"           # flatbuffers_schemas
  "${motis-parking-dbschema-dir}"             # schema_include_dirs
  generated-motis-parking-dbschema-headers    # custom_target_name
  ""                                          # additional_dependencies
  "${motis-parking-generated-headers-dir}"    # generated_includes_dir
  ""                                          # binary_schemas_dir
  ""                                          # copy_text_schemas_dir
)

file(GLOB_RECURSE motis-parking-files src/*.cc)

add_library(motis-parking STATIC ${motis-parking-files})
target_include_directories(motis-parking PUBLIC include)
target_include_directories(motis-parking PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../ppr/include)
target_compile_features(motis-parking PUBLIC cxx_std_17)
add_dependencies(motis-parking generated-motis-parking-dbschema-headers)
target_link_libraries(motis-parking
  boost-system
  motis-module
  motis-core
  geo
  lmdb
  fmt
  ppr-routing
  osmium
  protozero
  zlibstatic
  expat
)
target_compile_options(motis-parking PRIVATE ${MOTIS_CXX_FLAGS})
