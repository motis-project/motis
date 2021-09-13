cmake_minimum_required(VERSION 3.10)
project(motis)

file(GLOB_RECURSE motis-launcher-files src/*.cc)
add_executable(motis ${motis-launcher-files})
target_compile_features(motis PUBLIC cxx_std_17)
target_include_directories(motis PRIVATE include)
target_link_libraries(motis
  ${CMAKE_THREAD_LIBS_INIT}
  boost-system
  boost-filesystem
  conf
  motis-bootstrap
  web-server-tls
  ianatzdb-res
  pbf_sdf_fonts_res-res
  tiles_server_res-res
)
set_target_properties(motis PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if (NOT MSVC)
  set_target_properties(motis PROPERTIES LINK_FLAGS "-Wl,-rpath,./")
endif()
if (${CMAKE_BUILD_TYPE} STREQUAL Release)
  add_custom_command(
    TARGET motis
    POST_BUILD
    COMMAND ${CMAKE_STRIP} ${CMAKE_BINARY_DIR}/motis
  )
endif()
