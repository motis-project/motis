cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-intermodal-files src/*.cc)
add_library(motis-intermodal STATIC ${motis-intermodal-files})
target_include_directories(motis-intermodal PUBLIC include)
target_compile_features(motis-intermodal PUBLIC cxx_std_17)
target_link_libraries(motis-intermodal
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  motis-module
  motis-core
  geo
  ppr-routing
)

target_compile_options(motis-intermodal PRIVATE ${MOTIS_CXX_FLAGS})

file(GLOB_RECURSE motis-intermodal-generator-files eval/src/generator/*.cc)
add_executable(motis-intermodal-generator EXCLUDE_FROM_ALL ${motis-intermodal-generator-files})
target_compile_features(motis-intermodal-generator PUBLIC cxx_std_17)
target_include_directories(motis-intermodal-generator PUBLIC eval/include)
target_link_libraries(motis-intermodal-generator
  ${CMAKE_THREAD_LIBS_INIT}
  motis-bootstrap
  motis-core
  motis-intermodal
  conf
  )
target_compile_options(motis-intermodal-generator PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-intermodal-generator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

file(GLOB_RECURSE motis-intermodal-comparator-files eval/src/comparator/*.cc)
add_executable(motis-intermodal-comparator EXCLUDE_FROM_ALL ${motis-intermodal-comparator-files})
target_compile_features(motis-intermodal-comparator PUBLIC cxx_std_17)
target_include_directories(motis-intermodal-comparator PUBLIC eval/include)
target_link_libraries(motis-intermodal-comparator
  ${CMAKE_THREAD_LIBS_INIT}
  motis-module
  motis-core
  motis-intermodal
  motis-bootstrap
  conf)
target_compile_options(motis-intermodal-comparator PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-intermodal-comparator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_custom_target(motis-intermodal-eval)
add_dependencies(motis-intermodal-eval motis-intermodal-generator motis-intermodal-comparator)
