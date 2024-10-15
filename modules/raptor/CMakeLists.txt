cmake_minimum_required(VERSION 3.19)
project(motis)

file(GLOB_RECURSE motis-raptor-files src/*.cc)
add_library(motis-raptor STATIC ${motis-raptor-files})
target_include_directories(motis-raptor PUBLIC include)
target_compile_features(motis-raptor PUBLIC cxx_std_17)
target_link_libraries(motis-raptor
        motis-routing
        motis-module
        motis-core
        utl
        )
target_compile_options(motis-raptor PRIVATE ${MOTIS_CXX_FLAGS})
target_compile_definitions(motis-raptor PRIVATE ${MOTIS_COMPILE_DEFINITIONS})

if (MOTIS_CUDA)
    include(FindCUDAToolkit)

    file(GLOB_RECURSE gpu-raptor-files src/gpu/*.cu)
    add_library(gpu-raptor SHARED ${gpu-raptor-files})

    set_target_properties(gpu-raptor PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 17
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/"
            )
    target_compile_definitions(gpu-raptor PRIVATE MOTIS_CUDA)
    target_link_libraries(gpu-raptor utl motis-core)
    target_include_directories(gpu-raptor PRIVATE include)
    set_property(TARGET gpu-raptor PROPERTY CUDA_ARCHITECTURES 61 70 75 80)
    target_compile_options(gpu-raptor PRIVATE "--expt-relaxed-constexpr")
    #target_compile_options(gpu-raptor PRIVATE "-G")
    target_link_libraries(motis-raptor gpu-raptor CUDA::cudart_static)
endif ()
