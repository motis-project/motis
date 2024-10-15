cmake_minimum_required(VERSION 3.10)
project(motis)

file(GLOB_RECURSE motis-libjson-files src/*.cc)

add_library(motis-libjson STATIC ${motis-libjson-files})
target_include_directories(motis-libjson PUBLIC include)
target_compile_features(motis-libjson PUBLIC cxx_std_17)
target_include_directories(motis-libjson PUBLIC include)
target_link_libraries(motis-libjson rapidjson utl)
target_compile_options(motis-libjson PRIVATE ${MOTIS_CXX_FLAGS})
