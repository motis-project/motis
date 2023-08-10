cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-rt-files src/*.cc)
add_library(motis-rt STATIC ${motis-rt-files})
target_include_directories(motis-rt PUBLIC include)
target_compile_features(motis-rt PUBLIC cxx_std_20)
target_link_libraries(motis-rt boost-system motis-core motis-module)
target_compile_options(motis-rt PRIVATE ${MOTIS_CXX_FLAGS})
