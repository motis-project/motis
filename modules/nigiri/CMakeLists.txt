cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-nigiri-files src/*.cc)
add_library(motis-nigiri STATIC ${motis-nigiri-files})
target_include_directories(motis-nigiri PUBLIC include)
target_compile_features(motis-nigiri PRIVATE cxx_std_20)
target_link_libraries(motis-nigiri boost-system boost-thread boost-filesystem motis-module nigiri tiles guess prometheus-cpp::core)
target_compile_options(motis-nigiri PRIVATE ${MOTIS_CXX_FLAGS})
