cmake_minimum_required(VERSION 3.10)
project(motis)

set(schedule-dir ${CMAKE_CURRENT_SOURCE_DIR}/schedule-format)
set(generated-headers-dir "${CMAKE_BINARY_DIR}/generated/motis/schedule-format")
file(GLOB_RECURSE schedule-format-files ${schedule-dir}/*.fbs)
build_flatbuffers(
  64                           # num_bits
  ""                           # commandline_options
  "${schedule-format-files}"   # flatbuffers_schemas
  "${schedule-dir}"            # schema_include_dirs
  generated-schedule-headers   # custom_target_name
  ""                           # additional_dependencies
  "${generated-headers-dir}"   # generated_includes_dir
  ""                           # binary_schemas_dir
  ""                           # copy_text_schemas_dir
)

file(GLOB_RECURSE motis-loader-files src/*.cc)
add_library(motis-loader STATIC ${motis-loader-files})
target_include_directories(motis-loader PUBLIC include)
target_compile_features(motis-loader PUBLIC cxx_std_17)
add_dependencies(motis-loader generated-schedule-headers)
target_link_libraries(motis-loader
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  flatbuffers64
  motis-core
  date
  geo
  tz
)
target_compile_options(motis-loader PRIVATE ${MOTIS_CXX_FLAGS})
target_compile_definitions(motis-loader PRIVATE FLATBUFFERS_64=1)
