cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-metrics-files src/*.cc)
add_library(motis-metrics STATIC ${motis-metrics-files})
target_include_directories(motis-metrics PUBLIC include)
target_compile_features(motis-metrics PRIVATE cxx_std_23)
target_link_libraries(motis-metrics
        boost-system
        boost-thread
        motis-module
        prometheus-cpp::core
)
target_compile_options(motis-metrics PRIVATE ${MOTIS_CXX_FLAGS})
