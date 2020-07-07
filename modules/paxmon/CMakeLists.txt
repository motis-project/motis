cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-paxmon-files src/*.cc)
add_library(motis-paxmon STATIC ${motis-paxmon-files})
target_include_directories(motis-paxmon PUBLIC include)
target_compile_features(motis-paxmon PUBLIC cxx_std_17)
target_link_libraries(motis-paxmon
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  motis-module
  motis-core
)
target_compile_options(motis-paxmon PRIVATE ${MOTIS_CXX_FLAGS})

file(GLOB_RECURSE motis-pax-groups-files tools/src/groups/*.cc)
add_executable(motis-pax-groups EXCLUDE_FROM_ALL ${motis-pax-groups-files})
target_compile_features(motis-pax-groups PUBLIC cxx_std_17)
target_include_directories(motis-pax-groups PUBLIC tools/include)
target_link_libraries(motis-pax-groups
  ${CMAKE_THREAD_LIBS_INIT}
  conf
  motis-core
  motis-paxmon
)
target_compile_options(motis-pax-groups PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-pax-groups PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

file(GLOB_RECURSE motis-pax-convert-files tools/src/convert/*.cc)
add_executable(motis-pax-convert EXCLUDE_FROM_ALL ${motis-pax-convert-files})
target_compile_features(motis-pax-convert PUBLIC cxx_std_17)
target_include_directories(motis-pax-convert PUBLIC tools/include)
target_link_libraries(motis-pax-convert
  ${CMAKE_THREAD_LIBS_INIT}
  conf
  motis-core
  motis-module
  motis-paxmon
)
target_compile_options(motis-pax-convert PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-pax-convert PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
