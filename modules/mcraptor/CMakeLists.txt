cmake_minimum_required(VERSION 3.19)
project(motis)

file(GLOB_RECURSE motis-mcraptor-files src/*.cc)
add_library(motis-mcraptor STATIC ${motis-mcraptor-files})
target_include_directories(motis-mcraptor PUBLIC include)
target_compile_features(motis-mcraptor PUBLIC cxx_std_17)
target_link_libraries(motis-mcraptor
        motis-routing
        motis-module
        motis-core
        )
target_compile_options(motis-mcraptor PRIVATE ${MOTIS_CXX_FLAGS})
target_compile_definitions(motis-mcraptor PRIVATE ${MOTIS_COMPILE_DEFINITIONS})