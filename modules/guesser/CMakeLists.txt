cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-guesser-files src/*.cc)
add_library(motis-guesser STATIC ${motis-guesser-files})
target_include_directories(motis-guesser PUBLIC include)
target_compile_features(motis-guesser PUBLIC cxx_std_20)
target_link_libraries(motis-guesser boost-system guess motis-module)
target_compile_options(motis-guesser PRIVATE ${MOTIS_CXX_FLAGS})
