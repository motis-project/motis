cmake_minimum_required(VERSION 3.10)
project(motis)

################################
# flatbuffers schema
################################
set(motis-libpath-fbs-dir ${CMAKE_CURRENT_SOURCE_DIR}/fbs)
set(generated-headers-dir "${CMAKE_BINARY_DIR}/generated/motis/path/fbs")
file(GLOB_RECURSE motis-libpath-fbs-files ${motis-libpath-fbs-dir}/*.fbs)

build_flatbuffers(
  32
  "--everything-required"              # commandline_options
  "${motis-libpath-fbs-files}"         # flatbuffers_schemas
  "${motis-libpath-fbs-dir}"           # schema_include_dirs
  generated-motis-libpath-fbs-headers  # custom_target_name
  ""                                   # additional_dependencies
  "${generated-headers-dir}"           # generated_includes_dir
  ""                                   # binary_schemas_dir
  ""                                   # copy_text_schemas_dir
)

################################
# path library
################################
file(GLOB_RECURSE motis-libpath-files src/*.cc)

add_library(motis-libpath STATIC ${motis-libpath-files})
target_include_directories(motis-libpath PUBLIC include)
target_compile_features(motis-libpath PUBLIC cxx_std_17)
target_include_directories(motis-libpath PUBLIC include)
add_dependencies(motis-libpath generated-motis-libpath-fbs-headers)
target_link_libraries(motis-libpath
  ${Boost_SYSTEM_LIBRARY}
  motis-core
  motis-module
  geo
  lmdb
  tiles
)
target_compile_options(motis-libpath PRIVATE ${MOTIS_CXX_FLAGS})
