cmake_minimum_required(VERSION 3.21)
project(motis)

find_package(Python3 REQUIRED
                     COMPONENTS Interpreter Development)

include_directories(include)

build_flatbuffers(
  32                                            # num_bits
  "--everything-required"                       # commandline_options
  "${motis-transfers-dbschema-files}"           # flatbuffers_schemas
  "${motis-transfers-dbschema-dir}"             # schema_include_dirs
  ""                                            # custom_target_name
  ""                                            # additional_dependencies
  "${motis-transfers-generated-headers-dir}"    # generated_includes_dir
  ""                                            # binary_schemas_dir
  ""                                            # copy_text_schemas_dir
  )

file(GLOB_RECURSE motis-transfers-files src/*.cc)
add_library(motis-transfers STATIC ${motis-transfers-files})
target_include_directories(motis-transfers PUBLIC include)

target_compile_features(motis-transfers PRIVATE cxx_std_20)
target_link_libraries(motis-transfers PRIVATE
  motis-module
  cista
  nigiri
  transfers
)

target_compile_options(motis-transfers PRIVATE ${MOTIS_CXX_FLAGS})
