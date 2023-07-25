cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-valhalla-files src/*.cc)
add_library(motis-valhalla STATIC ${motis-valhalla-files})
target_include_directories(motis-valhalla PUBLIC include)
target_compile_features(motis-valhalla PUBLIC cxx_std_17)
target_link_libraries(motis-valhalla
    valhalla
    valhalla-mjolnir
    valhalla-odin
    valhalla-baldr
    valhalla-tyr
    valhalla-midgard
    motis-module
    boost-thread)
target_compile_options(motis-valhalla PRIVATE ${MOTIS_CXX_FLAGS})
