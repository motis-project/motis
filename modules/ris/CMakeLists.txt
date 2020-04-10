cmake_minimum_required(VERSION 3.10)
project(motis)

set(gtfs-rt-header generated/motis/ris/gtfsrt.pb.h)
set(gtfs-rt-src generated/motis/ris/gtfsrt.pb.cc)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gtfsrt.pb.h
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gtfsrt.pb.cc
  COMMAND
    protoc
      --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
      --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/gtfsrt.proto
  DEPENDS protoc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gtfsrt.proto
  COMMENT "Generating GTFS RT protocol buffers C++ code"
)

file(GLOB_RECURSE motis-ris-files src/*.cc)
add_library(motis-ris STATIC
  ${motis-ris-files}
  ${CMAKE_CURRENT_BINARY_DIR}/gtfsrt.pb.h
  ${CMAKE_CURRENT_BINARY_DIR}/gtfsrt.pb.cc
)
target_include_directories(motis-ris PUBLIC include ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(motis-ris PUBLIC cxx_std_17)
target_compile_definitions(motis-ris PUBLIC PROTOBUF_LINKED=1)
target_link_libraries(motis-ris
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_FILESYSTEM}
  protobuf 
  motis-module
  motis-core
  miniz
  pugixml
  lmdb
  tar
)
target_compile_options(motis-ris PRIVATE ${MOTIS_CXX_FLAGS})
