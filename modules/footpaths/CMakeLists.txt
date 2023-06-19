cmake_minimum_required(VERSION 3.21)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-footpaths-files src/*.cc)
add_library(motis-footpaths STATIC ${motis-footpaths-files})
target_include_directories(motis-footpaths PUBLIC include)
target_compile_features(motis-footpaths PRIVATE cxx_std_20)
target_link_libraries(motis-footpaths
        boost-system
        boost-thread
        motis-module
        nigiri
        osmium
        geo
        ppr-routing
        )
target_compile_options(motis-footpaths PRIVATE ${MOTIS_CXX_FLAGS})
