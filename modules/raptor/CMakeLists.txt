cmake_minimum_required(VERSION 3.10)
project(motis)

include_directories(include)

if (MOTIS_CUDA)
    include(FindCUDAToolkit)
    include_directories(include/gpu)

    add_library(gpu-raptor SHARED src/gpu/everything.cu)
    set_target_properties(gpu-raptor PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 17
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/"
            )

    set_property(TARGET gpu-raptor PROPERTY CUDA_ARCHITECTURES 75 61 37)
    target_compile_options(gpu-raptor PRIVATE "--expt-relaxed-constexpr")
    target_link_libraries(gpu-raptor motis-core CUDA::cudart_static)
endif ()

file(GLOB_RECURSE motis-raptor-files src/*.cc)
add_library(motis-raptor STATIC ${motis-raptor-files})
target_include_directories(motis-raptor PUBLIC include)
target_compile_features(motis-raptor PUBLIC cxx_std_17)
target_link_libraries(motis-raptor
        motis-routing
        motis-module
        motis-core
        )

target_compile_options(motis-raptor PRIVATE ${MOTIS_CXX_FLAGS})
target_compile_definitions(motis-raptor PRIVATE ${MOTIS_COMPILE_DEFINITIONS})

if (MOTIS_CUDA)
    target_link_libraries(motis-raptor gpu-raptor)
endif ()