cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-tripbased-files src/*.cc)

add_library(motis-tripbased STATIC ${motis-tripbased-files})
target_include_directories(motis-tripbased PUBLIC include)
target_link_libraries(motis-tripbased
  boost-system
  motis-routing
  motis-module
  motis-core
  geo
  date
)
target_compile_features(motis-tripbased PUBLIC cxx_std_17)
target_compile_options(motis-tripbased PRIVATE ${MOTIS_CXX_FLAGS})
