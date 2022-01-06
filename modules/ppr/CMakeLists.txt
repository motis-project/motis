cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-ppr-files src/*.cc)
add_library(motis-ppr STATIC ${motis-ppr-files})
target_include_directories(motis-ppr PUBLIC include)
target_compile_features(motis-ppr PUBLIC cxx_std_20)
target_link_libraries(motis-ppr
  boost-system
  ppr-routing
  ppr-preprocessing
  ppr-profiles
  motis-module
  motis-core
)
target_compile_options(motis-ppr PRIVATE ${MOTIS_CXX_FLAGS})
