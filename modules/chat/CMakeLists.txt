cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-chat-files src/*.cc)
add_library(motis-chat STATIC ${motis-chat-files})
target_link_libraries(motis-chat
  boost-system
  motis-module
  motis-core
)
target_include_directories(motis-chat PUBLIC include)
target_compile_features(motis-chat PUBLIC cxx_std_17)
target_compile_options(motis-chat PRIVATE ${MOTIS_CXX_FLAGS})
