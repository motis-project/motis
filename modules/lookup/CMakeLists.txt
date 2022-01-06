cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-lookup-files src/*.cc)
add_library(motis-lookup STATIC ${motis-lookup-files})
target_include_directories(motis-lookup PUBLIC include)
target_compile_features(motis-lookup PUBLIC cxx_std_20)
target_compile_options(motis-lookup PRIVATE ${MOTIS_CXX_FLAGS})
target_link_libraries(motis-lookup 
  boost-system
  geo
  motis-module
  motis-core
)
