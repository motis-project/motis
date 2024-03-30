cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-adr-files src/*.cc)
add_library(motis-adr STATIC ${motis-adr-files})
target_link_libraries(motis-adr
  adr
  motis-module
  boost-thread
)
target_include_directories(motis-adr PUBLIC include)
target_compile_features(motis-adr PUBLIC cxx_std_23)
target_compile_options(motis-adr PRIVATE ${MOTIS_CXX_FLAGS})
