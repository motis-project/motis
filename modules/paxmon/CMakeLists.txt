cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-paxmon-files src/*.cc)
add_library(motis-paxmon STATIC ${motis-paxmon-files})
target_include_directories(motis-paxmon PUBLIC include)
target_compile_features(motis-paxmon PUBLIC cxx_std_20)
target_link_libraries(motis-paxmon
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  motis-bootstrap
  motis-module
  motis-core
)
target_compile_options(motis-paxmon PRIVATE ${MOTIS_CXX_FLAGS})
