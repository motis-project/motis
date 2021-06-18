cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-isochrone-files src/*.cc)
add_library(motis-isochrone STATIC ${motis-isochrone-files})
target_include_directories(motis-isochrone PUBLIC include)
target_compile_features(motis-isochrone PUBLIC cxx_std_17)
target_link_libraries(motis-isochrone ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_SYSTEM_LIBRARY} motis-module)
target_compile_options(motis-isochrone PRIVATE ${MOTIS_CXX_FLAGS})

include_directories(eval/include)

file(GLOB_RECURSE motis-isochrone-generator-files eval/src/query_generator.cc)
add_executable(motis-isochrone-generator EXCLUDE_FROM_ALL ${motis-isochrone-generator-files})
target_compile_features(motis-isochrone-generator PUBLIC cxx_std_17)
target_link_libraries(motis-isochrone-generator motis-bootstrap motis-core conf ${CMAKE_THREAD_LIBS_INIT} ianatzdb-res)
target_compile_options(motis-isochrone-generator PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-isochrone-generator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
