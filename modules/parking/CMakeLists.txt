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
file(GLOB_RECURSE motis-parking-prepare-only-files
  src/prepare/*.cc
)
file(GLOB_RECURSE motis-parking-prepare-files
  src/prepare/*.cc
  src/parkings.cc
  src/database.cc
)

foreach(file ${motis-parking-prepare-only-files})
    list(REMOVE_ITEM motis-parking-files "${file}")
endforeach()

add_library(motis-parking STATIC ${motis-parking-files})
target_include_directories(motis-parking PUBLIC include)
target_compile_features(motis-parking PUBLIC cxx_std_17)
add_dependencies(motis-parking generated-motis-parking-dbschema-headers)
target_link_libraries(motis-parking
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  motis-module
  motis-core
  geo
  lmdb
  fmt
  ppr-routing
)
target_compile_options(motis-parking PRIVATE ${MOTIS_CXX_FLAGS})

add_executable(parking-prepare EXCLUDE_FROM_ALL ${motis-parking-prepare-files})
target_include_directories(parking-prepare PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../ppr/include)
target_compile_features(parking-prepare PUBLIC cxx_std_17)
add_dependencies(parking-prepare generated-motis-parking-dbschema-headers)
target_link_libraries(parking-prepare
  ${CMAKE_THREAD_LIBS_INIT}
  ${Boost_THREAD_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  conf
  zlibstatic
  expat
  geo
  lmdb
  fmt
  motis-core
  ppr-routing
  ppr-profiles
  osmium
  protozero
)
target_compile_options(parking-prepare PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(parking-prepare PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
