cmake_minimum_required(VERSION 3.21)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-transfers-files src/*.cc)
add_library(motis-transfers STATIC ${motis-transfers-files})
target_include_directories(motis-transfers PUBLIC include)

target_compile_features(motis-transfers PRIVATE cxx_std_23)
target_link_libraries(motis-transfers PRIVATE
  motis-module
  cista
  nigiri
  transfers
)

target_compile_options(motis-transfers PRIVATE ${MOTIS_CXX_FLAGS})
