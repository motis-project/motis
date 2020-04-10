project(motis)
cmake_minimum_required(VERSION 3.10)

set(MOTIS_SCHEDULE_MODE "offset" CACHE STRING "Pointer type for schedule data structures")

add_library(motis-data INTERFACE)
target_compile_features(motis-data INTERFACE cxx_std_17)
target_include_directories(motis-data INTERFACE include)
target_link_libraries(motis-data INTERFACE cista)
if (MOTIS_SCHEDULE_MODE STREQUAL "raw")
  target_compile_definitions(motis-data INTERFACE MOTIS_SCHEDULE_MODE_RAW)
else()
  target_compile_definitions(motis-data INTERFACE MOTIS_SCHEDULE_MODE_OFFSET)
endif()
