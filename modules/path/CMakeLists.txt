cmake_minimum_required(VERSION 3.10)
project(motis)

################################
# flatbuffers schema
################################
set(path-fbs-dir ${CMAKE_CURRENT_SOURCE_DIR}/fbs)
set(generated-headers-dir "${CMAKE_BINARY_DIR}/generated/motis/path/fbs")
file(GLOB_RECURSE path-fbs-files ${path-fbs-dir}/*.fbs)

build_flatbuffers(
  32
  "--everything-required"        # commandline_options
  "${path-fbs-files}"            # flatbuffers_schemas
  "${path-fbs-dir}"              # schema_include_dirs
  generated-path-fbs-headers     # custom_target_name
  ""                             # additional_dependencies
  "${generated-headers-dir}"     # generated_includes_dir
  ""                             # binary_schemas_dir
  ""                             # copy_text_schemas_dir
)

################################
# path module
################################
file(GLOB_RECURSE motis-path-files
  src/path.cc
  src/path_database_query.cc
)

file(GLOB_RECURSE motis-path-files ${motis-path-files})
add_library(motis-path STATIC ${motis-path-files})
target_include_directories(motis-path PUBLIC include)
target_compile_features(motis-path PUBLIC cxx_std_17)
target_include_directories(motis-path PUBLIC include)
add_dependencies(motis-path generated-path-fbs-headers)
target_link_libraries(motis-path
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  motis-core
  motis-module
  geo
  lmdb
  tiles
)
target_compile_options(motis-path PRIVATE ${MOTIS_CXX_FLAGS})

file(GLOB_RECURSE motis-path-prepare-files
  src/db_builder.cc
  src/prepare/*.cc
)

################################
# path-prepare library
################################
add_library(motis-path-prepare STATIC ${motis-path-prepare-files})
target_compile_features(motis-path-prepare PUBLIC cxx_std_17)
add_dependencies(motis-path-prepare generated-path-fbs-headers)
target_include_directories(motis-path-prepare PUBLIC include)
target_link_libraries(motis-path-prepare
  motis-core
  motis-module
  motis-loader
  conf
  geo
  osrm
  osmium
  protozero
  tiles
  lmdb
  cista
  ${Boost_FILESYSTEM}
  ${Boost_THREAD_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_options(motis-path-prepare PRIVATE ${MOTIS_CXX_FLAGS})
target_link_libraries(motis-path motis-path-prepare)

################################
# path-prepare executable
################################
file(GLOB_RECURSE motis-path-prepare-main-files src/prepare.cc)

add_executable(path-prepare EXCLUDE_FROM_ALL ${motis-path-prepare-main-files})
target_compile_features(path-prepare PUBLIC cxx_std_17)
add_dependencies(path-prepare generated-path-fbs-headers)
target_include_directories(path-prepare PUBLIC include)
target_link_libraries(path-prepare motis-path-prepare)

target_compile_options(path-prepare PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(path-prepare PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
