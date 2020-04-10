cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

set(bikesharing-dbschema-dir ${CMAKE_CURRENT_SOURCE_DIR}/dbschema)
set(generated-headers-dir "${CMAKE_BINARY_DIR}/generated/motis/bikesharing/dbschema")
file(GLOB_RECURSE bikesharing-dbschema-files ${bikesharing-dbschema-dir}/*.fbs)

build_flatbuffers(
  32                                       # num_bits
  "--everything-required"                  # commandline_options
  "${bikesharing-dbschema-files}"          # flatbuffers_schemas
  "${bikesharing-dbschema-dir}"            # schema_include_dirs
  generated-bikesharing-dbschema-headers   # custom_target_name
  ""                                       # additional_dependencies
  "${generated-headers-dir}"               # generated_includes_dir
  ""                                       # binary_schemas_dir
  ""                                       # copy_text_schemas_dir
)

file(GLOB_RECURSE motis-bikesharing-files src/*.cc)
add_library(motis-bikesharing STATIC ${motis-bikesharing-files})
target_include_directories(motis-bikesharing PUBLIC include)
target_compile_features(motis-bikesharing PUBLIC cxx_std_17)
add_dependencies(motis-bikesharing generated-bikesharing-dbschema-headers)
target_compile_options(motis-bikesharing PRIVATE ${MOTIS_CXX_FLAGS})
target_link_libraries(motis-bikesharing
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  motis-module
  pugixml
  geo
  lmdb
)
