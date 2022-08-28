cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-cc-files src/*.cc)
add_library(motis-cc STATIC ${motis-cc-files})
target_include_directories(motis-cc PUBLIC include)
target_compile_features(motis-cc PUBLIC cxx_std_17)
target_link_libraries(motis-cc boost-system motis-module)
target_compile_options(motis-cc PRIVATE ${MOTIS_CXX_FLAGS})
