cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-osr-files src/*.cc)
add_library(motis-osr STATIC ${motis-osr-files})
target_include_directories(motis-osr PUBLIC include)
target_compile_features(motis-osr PRIVATE cxx_std_20)
target_link_libraries(motis-osr boost-system boost-thread motis-module osr)
target_compile_options(motis-osr PRIVATE ${MOTIS_CXX_FLAGS})
