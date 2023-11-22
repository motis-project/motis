project(motis)

set(gtfs-rt-header generated/motis/ris/gtfsrt.pb.h)
set(gtfs-rt-src generated/motis/ris/gtfsrt.pb.cc)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gtfsrt.pb.h
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gtfsrt.pb.cc
  COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:protoc>
    --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gtfsrt.proto
  DEPENDS protoc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gtfsrt.proto
  COMMENT "Generating GTFS RT protocol buffers C++ code"
)
add_library(motis-ris-gtfsrtpb STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/gtfsrt.pb.h
  ${CMAKE_CURRENT_BINARY_DIR}/gtfsrt.pb.cc
)
target_link_libraries(motis-ris-gtfsrtpb protobuf::libprotobuf)
target_compile_features(motis-ris-gtfsrtpb PUBLIC cxx_std_20)
target_include_directories(motis-ris-gtfsrtpb SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
