cmake_minimum_required(VERSION 3.10)
project(motis)

file(GLOB_RECURSE motis-csa-files src/*.cc)
add_library(motis-csa STATIC ${motis-csa-files})
target_include_directories(motis-csa PUBLIC include)
target_compile_features(motis-csa PUBLIC cxx_std_17)
target_link_libraries(motis-csa
  boost-system
  motis-routing
  motis-module
  motis-core
  )
target_compile_options(motis-csa PRIVATE ${MOTIS_CXX_FLAGS})
target_compile_definitions(motis-csa PRIVATE ${MOTIS_COMPILE_DEFINITIONS})

if (MOTIS_CUDA)
  add_library(gpu-csa SHARED src/gpu/gpu_csa.cu)
  set_target_properties(gpu-csa PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")
  target_include_directories(gpu-csa PUBLIC include)
  set_property(TARGET gpu-csa PROPERTY CUDA_ARCHITECTURES 75 61)
  target_link_libraries(motis-csa gpu-csa)
endif ()
