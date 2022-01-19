cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-address-files src/*.cc)
add_library(motis-address STATIC ${motis-address-files})
target_link_libraries(motis-address
  cista
  boost-system
  address-typeahead
  motis-module
  motis-core
)
target_include_directories(motis-address PUBLIC include)
target_compile_features(motis-address PUBLIC cxx_std_17)
target_compile_options(motis-address PRIVATE ${MOTIS_CXX_FLAGS})
