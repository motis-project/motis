cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-railviz-files src/*.cc)
add_library(motis-railviz STATIC ${motis-railviz-files})
target_include_directories(motis-railviz PUBLIC include)
target_compile_features(motis-railviz PUBLIC cxx_std_17)
target_link_libraries(motis-railviz
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  motis-module
  motis-core
  motis-libpath
  tiles
  geo
)
target_compile_options(motis-railviz PRIVATE ${MOTIS_CXX_FLAGS})