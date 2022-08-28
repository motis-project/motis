cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-paxforecast-files src/*.cc)
add_library(motis-paxforecast STATIC ${motis-paxforecast-files})
target_include_directories(motis-paxforecast PUBLIC include)
target_compile_features(motis-paxforecast PUBLIC cxx_std_17)
target_link_libraries(motis-paxforecast
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  motis-module
  motis-core
  motis-paxmon
  lmdb
)
target_compile_options(motis-paxforecast PRIVATE ${MOTIS_CXX_FLAGS})
