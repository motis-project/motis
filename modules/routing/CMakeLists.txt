cmake_minimum_required(VERSION 3.10)
project(motis)

file(GLOB_RECURSE motis-routing-files src/*.cc)
add_library(motis-routing STATIC ${motis-routing-files})
target_include_directories(motis-routing PUBLIC include)
target_compile_features(motis-routing PUBLIC cxx_std_20)
target_link_libraries(motis-routing
  boost-system
  motis-bootstrap
  motis-module
  motis-core
)
if (MSVC)
    set_target_properties(motis-routing PROPERTIES COMPILE_FLAGS "${MOTIS_CXX_FLAGS} /bigobj")
else ()
    target_compile_options(motis-routing PRIVATE ${MOTIS_CXX_FLAGS})
endif ()