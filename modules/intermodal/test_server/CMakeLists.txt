cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-ondemand-server-files src/*.cc)
add_executable(motis-ondemand-server ${motis-ondemand-server-files})
target_include_directories(motis-ondemand-server PUBLIC include)
target_compile_features(motis-ondemand-server PUBLIC cxx_std_17)
target_compile_options(motis-ondemand-server PRIVATE ${MOTIS_CXX_FLAGS})
target_compile_definitions(motis-ondemand-server PRIVATE ${MOTIS_COMPILE_DEFINITIONS})
target_link_libraries(motis-ondemand-server
        boost-system
        motis-module
        motis-core
        motis-bootstrap
        geo
        ppr-routing
        motis-libjson
        http-client
)
