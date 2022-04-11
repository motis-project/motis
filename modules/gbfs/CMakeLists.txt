cmake_minimum_required(VERSION 3.10)
project(motis)

file(GLOB_RECURSE motis-gbfs-files src/*.cc)
add_library(motis-gbfs STATIC ${motis-gbfs-files})
target_include_directories(motis-gbfs PUBLIC include)
target_compile_features(motis-gbfs PRIVATE cxx_std_20)
target_link_libraries(motis-gbfs
  motis-module
  motis-core
  geo
  motis-libjson
  http-client
)
target_compile_options(motis-gbfs PRIVATE ${MOTIS_CXX_FLAGS})
