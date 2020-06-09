cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-tiles-files src/*.cc)
add_library(motis-tiles STATIC ${motis-tiles-files})
target_include_directories(motis-tiles PUBLIC include)
target_compile_features(motis-tiles PUBLIC cxx_std_17)
target_link_libraries(motis-tiles 
  ${Boost_PROGRAM_OPTIONS_LIBRARY} 
  ${Boost_SYSTEM_LIBRARY} 
  motis-module
  lmdb
  tiles
  tiles-import-library
  web-server
  pbf_sdf_fonts_res
  pbf_sdf_fonts_res-res
)
target_compile_options(motis-tiles PRIVATE ${MOTIS_CXX_FLAGS})

set_property(TARGET motis-tiles tiles tiles-import-library
             APPEND PROPERTY COMPILE_DEFINITIONS TILES_GLOBAL_PROGRESS_TRACKER=1)
