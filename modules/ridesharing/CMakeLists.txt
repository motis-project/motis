cmake_minimum_required(VERSION 3.0)
project(motis)

set(ridesharing-dbschema-dir ${CMAKE_CURRENT_SOURCE_DIR}/dbschema)
set(generated-headers-dir "${CMAKE_BINARY_DIR}/generated/motis/ridesharing/dbschema")
file(GLOB_RECURSE ridesharing-dbschema-files ${ridesharing-dbschema-dir}/*.fbs)

build_flatbuffers(
  32                                       # num_bits
  "--everything-required"                  # commandline_options
  "${ridesharing-dbschema-files}"          # flatbuffers_schemas
  "${ridesharing-dbschema-dir}"            # schema_include_dirs
  generated-ridesharing-dbschema-headers   # custom_target_name
  ""                                       # additional_dependencies
  "${generated-headers-dir}"               # generated_includes_dir
  ""                                       # binary_schemas_dir
  ""                                       # copy_text_schemas_dir
)

file(GLOB_RECURSE motis-ridesharing-files src/*.cc)
add_library(motis-ridesharing STATIC ${motis-ridesharing-files})
add_dependencies(motis-ridesharing generated-ridesharing-dbschema-headers)
target_compile_options(motis-ridesharing PRIVATE ${MOTIS_CXX_FLAGS})
target_include_directories(motis-ridesharing PUBLIC include)
target_link_libraries(motis-ridesharing
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  motis-module
  geo
  lmdb
)
