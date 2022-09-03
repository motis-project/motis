cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-intermodal-files src/*.cc test_server/*.cc)
add_library(motis-intermodal STATIC ${motis-intermodal-files})
target_include_directories(motis-intermodal PUBLIC include test_server/include)
target_compile_features(motis-intermodal PUBLIC cxx_std_17)
target_link_libraries(motis-intermodal
  boost-system
  motis-module
  motis-core
  motis-bootstrap
  geo
  ppr-routing
  motis-libjson
  http-client
)
target_compile_options(motis-intermodal PRIVATE ${MOTIS_CXX_FLAGS})