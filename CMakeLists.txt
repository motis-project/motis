cmake_minimum_required(VERSION 3.20)

project(motis)

if (NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    if (ICC_MIMALLOC)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        set(protobuf_MSVC_STATIC_RUNTIME OFF)
    else ()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        set(protobuf_MSVC_STATIC_RUNTIME ON)
    endif ()
endif ()

include(cmake/buildcache.cmake)
include(cmake/pkg.cmake)


# --- LINT ---
option(ICC_LINT "Run clang-tidy with the compiler." OFF)
if (ICC_LINT)
    # clang-tidy will be run on all targets defined hereafter
    include(cmake/clang-tidy.cmake)
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(motis-compile-options
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
            -Wno-unused-template
            -Werror)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(motis-compile-options -Wall -Wextra -Werror -Wno-unknown-pragmas)
elseif (MSVC)
    set(motis-compile-options /WX)
else ()
    set(motis-compile-options
            -Wall
            -Wextra
            -Wno-maybe-uninitialized)
    if (NOT CMAKE_CROSSCOMPILING)
        set(motis-compile-options ${motis-compile-options} -Werror)
    endif ()
endif ()

# --- OPENAPI ---
openapi_generate(openapi.yaml motis-api motis::api)


# --- LIB ---
file(GLOB_RECURSE motis-files src/*.cc)
add_library(motis ${motis-files})
target_include_directories(motis PUBLIC include)
target_compile_features(motis PUBLIC cxx_std_23)
target_link_libraries(motis nigiri osr boost-json adr motis-api)

# --- ADR EXTEND ---
add_executable(motis-adr-extend exe/adr_extend.cc)
target_compile_features(motis-adr-extend PUBLIC cxx_std_23)
target_link_libraries(motis-adr-extend motis)

# --- PREPARE ---
add_executable(motis-prepare exe/prepare.cc)
target_compile_features(motis-prepare PUBLIC cxx_std_23)
target_link_libraries(motis-prepare motis)


# --- SERVER ---
add_executable(motis-server exe/server.cc)
target_compile_features(motis-server PUBLIC cxx_std_23)
target_link_libraries(motis-server motis web-server adr fmt::fmt rtree geo cista ianatzdb-res)


# --- TEST ---
add_library(motis-generated INTERFACE)
target_include_directories(motis-generated INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dir.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/generated/test_dir.h
)
file(GLOB_RECURSE motis-test-files test/*.cc)
add_executable(motis-test ${motis-test-files})
target_link_libraries(motis-test motis gtest web-server ianatzdb-res motis-generated)
target_compile_options(motis-test PRIVATE ${motis-compile-options})
