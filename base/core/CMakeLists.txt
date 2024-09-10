project(motis)
cmake_minimum_required(VERSION 3.10)

file(GLOB_RECURSE motis-core-files src/*.cc)
add_library(motis-core STATIC ${motis-core-files})
target_compile_features(motis-core PUBLIC cxx_std_17)
target_include_directories(motis-core PUBLIC include)
target_compile_definitions(motis-core PUBLIC ${MOTIS_COMPILE_DEFINITIONS})
target_compile_options(motis-core PRIVATE ${MOTIS_CXX_FLAGS})
target_link_libraries(motis-core
  utl
  date
  geo
  motis-data
  motis-module
  opentelemetry_api
)
