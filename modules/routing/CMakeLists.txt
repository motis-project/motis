cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-routing-files src/*.cc)
add_library(motis-routing STATIC ${motis-routing-files})
target_include_directories(motis-routing PUBLIC include)
target_compile_features(motis-routing PUBLIC cxx_std_17)
target_link_libraries(motis-routing
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  motis-module
  motis-core
)
if (MSVC)
    set_target_properties(motis-routing PROPERTIES COMPILE_FLAGS "${MOTIS_CXX_FLAGS} /bigobj")
else ()
    target_compile_options(motis-routing PRIVATE ${MOTIS_CXX_FLAGS})
endif ()

include_directories(eval/include)

file(GLOB_RECURSE motis-generator-files eval/src/query_generator.cc)
add_executable(motis-generator EXCLUDE_FROM_ALL ${motis-generator-files})
target_compile_features(motis-generator PUBLIC cxx_std_17)
target_link_libraries(motis-generator motis-bootstrap motis-core conf ${CMAKE_THREAD_LIBS_INIT})
target_compile_options(motis-generator PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-generator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

file(GLOB_RECURSE motis-analyzer-files eval/src/result_analyzer.cc)
add_executable(motis-analyzer EXCLUDE_FROM_ALL ${motis-analyzer-files})
target_compile_features(motis-analyzer PUBLIC cxx_std_17)
target_link_libraries(motis-analyzer conf motis-module motis-core ${CMAKE_THREAD_LIBS_INIT})
target_compile_options(motis-analyzer PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-analyzer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

file(GLOB_RECURSE motis-comparator-files eval/src/result_comparator.cc)
add_executable(motis-comparator EXCLUDE_FROM_ALL ${motis-comparator-files})
target_compile_features(motis-comparator PUBLIC cxx_std_17)
target_link_libraries(motis-comparator conf motis-module motis-core ${CMAKE_THREAD_LIBS_INIT})
target_compile_options(motis-comparator PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-comparator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

file(GLOB_RECURSE motis-xtract-files eval/src/xtract.cc)
add_executable(motis-xtract EXCLUDE_FROM_ALL ${motis-xtract-files})
target_compile_features(motis-xtract PUBLIC cxx_std_17)
target_link_libraries(motis-xtract conf motis-module motis-core ${CMAKE_THREAD_LIBS_INIT})
target_compile_options(motis-xtract PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-xtract PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

file(GLOB_RECURSE motis-query-rewrite-files eval/src/query_rewriter.cc)
add_executable(motis-query-rewrite EXCLUDE_FROM_ALL ${motis-query-rewrite-files})
target_compile_features(motis-query-rewrite PUBLIC cxx_std_17)
target_link_libraries(motis-query-rewrite conf motis-module motis-core ${CMAKE_THREAD_LIBS_INIT})
target_compile_options(motis-query-rewrite PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-query-rewrite PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

file(GLOB_RECURSE motis-print-journey-files eval/src/print.cc)
add_executable(motis-print-journey EXCLUDE_FROM_ALL ${motis-print-journey-files})
target_compile_features(motis-print-journey PUBLIC cxx_std_17)
target_link_libraries(motis-print-journey
        ${CMAKE_THREAD_LIBS_INIT}
        ${Boost_THREAD_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        motis-module
        motis-core
        )
target_compile_options(motis-print-journey PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis-print-journey PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_custom_target(motis-eval)
add_dependencies(motis-eval motis-xtract motis-generator motis-analyzer motis-comparator motis-print-journey motis-query-rewrite)
