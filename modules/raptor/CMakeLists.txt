cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

file(GLOB_RECURSE motis-raptor-files src/*.cc)
add_library(motis-raptor STATIC ${motis-raptor-files})
target_include_directories(motis-raptor PUBLIC include)
target_compile_features(motis-raptor PUBLIC cxx_std_20)
target_link_libraries(motis-raptor
  ${Boost_SYSTEM_LIBRARY}
  motis-routing
  motis-module
  motis-core
)
target_compile_options(motis-raptor PRIVATE ${MOTIS_CXX_FLAGS})
target_compile_definitions(motis-raptor PRIVATE ${MOTIS_COMPILE_DEFINITIONS})

if (MOTIS_CUDA)
  file(GLOB_RECURSE gpu-raptor-src-files gpu-raptor/src/*cu)
  add_library(gpu-raptor SHARED ${gpu-raptor-src-files})
  set_target_properties(gpu-raptor PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
    CUDA_PROPAGATE_HOST_FLAGS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")

  target_include_directories(gpu-raptor PUBLIC include)
  target_link_libraries(motis-raptor gpu-raptor)
endif()

