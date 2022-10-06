cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-intermodal-files src/*.cc)
add_library(motis-intermodal STATIC ${motis-intermodal-files})
target_include_directories(motis-intermodal PUBLIC include)
target_compile_features(motis-intermodal PUBLIC cxx_std_17)
target_link_libraries(motis-intermodal
  boost-system
  motis-module
  motis-core
  motis-bootstrap
  geo
  ppr-routing
)
target_compile_options(motis-intermodal PRIVATE ${MOTIS_CXX_FLAGS})
add_subdirectory(test_server)