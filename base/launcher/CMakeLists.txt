cmake_minimum_required(VERSION 3.10)
project(motis)

file(GLOB_RECURSE motis-launcher-files src/*.cc)
add_executable(motis ${motis-launcher-files})
target_compile_features(motis PUBLIC cxx_std_17)
target_include_directories(motis PRIVATE include)
target_link_libraries(motis
  ${CMAKE_THREAD_LIBS_INIT}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${NETWORKING}
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
