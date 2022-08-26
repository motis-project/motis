cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-osrm-files src/*.cc)

add_library(motis-osrm STATIC ${motis-osrm-files})
target_include_directories(motis-osrm PUBLIC include)
target_compile_features(motis-osrm PUBLIC cxx_std_20)
target_link_libraries(motis-osrm
  boost-system
  boost-filesystem
  tbb_static
  cista
  osrm
  osrm_contract
  osrm_extract
  motis-module
)
target_compile_options(motis-osrm PRIVATE ${MOTIS_CXX_FLAGS})
