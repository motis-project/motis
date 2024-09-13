cmake_minimum_required(VERSION 3.20)

project(icc)

if(NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    if(ICC_MIMALLOC)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        set(protobuf_MSVC_STATIC_RUNTIME OFF)
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        set(protobuf_MSVC_STATIC_RUNTIME ON)
    endif()
endif()

include(cmake/buildcache.cmake)
include(cmake/pkg.cmake)


# --- LINT ---
option(ICC_LINT "Run clang-tidy with the compiler." OFF)
if (ICC_LINT)
    # clang-tidy will be run on all targets defined hereafter
    include(cmake/clang-tidy.cmake)
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(icc-compile-options
            -Weverything
            -Wno-c++98-compat
            -Wno-c++98-compat-pedantic
            -Wno-newline-eof
            -Wno-missing-prototypes
            -Wno-padded
            -Wno-double-promotion
            -Wno-undef
            -Wno-undefined-reinterpret-cast
            -Wno-float-conversion
            -Wno-global-constructors
            -Wno-exit-time-destructors
            -Wno-switch-enum
            -Wno-c99-designator
            -Wno-zero-as-null-pointer-constant
            -Wno-missing-noreturn
            -Wno-undefined-func-template
            -Wno-unsafe-buffer-usage
            -Wno-c++20-compat
            -Wno-reserved-macro-identifier
            -Wno-documentation-unknown-command
            -Wno-duplicate-enum
            -Wno-ctad-maybe-unsupported
            -Wno-unknown-pragmas
            -Wno-c++20-extensions
            -Wno-switch-default
            -Werror)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(icc-compile-options -Wall -Wextra -Werror -Wno-unknown-pragmas)
elseif (MSVC)
    set(icc-compile-options /WX)
else ()
    set(icc-compile-options
            -Wall
            -Wextra
            -Wno-maybe-uninitialized)
    if (NOT CMAKE_CROSSCOMPILING)
        set(icc-compile-options ${icc-compile-options} -Werror)
    endif ()
endif ()

# --- OPENAPI ---
openapi_generate(openapi.yaml icc-api icc::api)


# --- LIB ---
file(GLOB_RECURSE icc-files src/*.cc)
add_library(icc ${icc-files})
target_include_directories(icc PUBLIC include)
target_compile_features(icc PUBLIC cxx_std_23)
target_link_libraries(icc nigiri osr boost-json adr icc-api)


# --- PREPARE ---
add_executable(icc-prepare exe/prepare.cc)
target_compile_features(icc-prepare PUBLIC cxx_std_23)
target_link_libraries(icc-prepare icc)


# --- SERVER ---
add_executable(icc-server exe/server.cc)
target_compile_features(icc-server PUBLIC cxx_std_23)
target_link_libraries(icc-server icc web-server adr fmt::fmt rtree geo cista ianatzdb-res)


# --- TEST ---
add_library(icc-generated INTERFACE)
target_include_directories(icc-generated INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dir.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/generated/test_dir.h
)
file(GLOB_RECURSE icc-test-files test/*.cc)
add_executable(icc-test ${icc-test-files})
target_link_libraries(icc-test icc gtest web-server ianatzdb-res icc-generated)
target_compile_options(icc-test PRIVATE ${icc-compile-options})
