cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-paxassign-files src/*.cc)
add_library(motis-paxassign STATIC ${motis-paxassign-files})
target_include_directories(motis-paxassign PUBLIC include)
target_compile_features(motis-paxassign PUBLIC cxx_std_17)
target_link_libraries(motis-paxassign
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  motis-module
  motis-core
  motis-rsl
)
target_compile_options(motis-paxassign PRIVATE ${MOTIS_CXX_FLAGS})
