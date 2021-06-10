cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-isochrone-files src/*.cc)
add_library(motis-isochrone STATIC ${motis-isochrone-files})
target_include_directories(motis-isochrone PUBLIC include)
target_compile_features(motis-isochrone PUBLIC cxx_std_17)
target_link_libraries(motis-isochrone ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_SYSTEM_LIBRARY} motis-module)
target_compile_options(motis-isochrone PRIVATE ${MOTIS_CXX_FLAGS})
