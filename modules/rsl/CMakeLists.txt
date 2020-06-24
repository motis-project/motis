cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-rsl-files src/*.cc)
add_library(motis-rsl STATIC ${motis-rsl-files})
target_include_directories(motis-rsl PUBLIC include)
target_compile_features(motis-rsl PUBLIC cxx_std_17)
target_link_libraries(motis-rsl
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  motis-module
  motis-core
)
target_compile_options(motis-rsl PRIVATE ${MOTIS_CXX_FLAGS})
