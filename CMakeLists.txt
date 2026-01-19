cmake_minimum_required(VERSION 3.20)

project(motis LANGUAGES C CXX ASM)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

option(MOTIS_MIMALLOC "use mimalloc" OFF)

set(MOTIS_STACKTRACE "AUTO" CACHE STRING "Enable stacktrace support (AUTO, ON, OFF)")
set_property(CACHE MOTIS_STACKTRACE PROPERTY STRINGS "AUTO;ON;OFF")
set(_MOTIS_STACKTRACE_ON "$<OR:$<STREQUAL:${MOTIS_STACKTRACE},ON>,$<AND:$<STREQUAL:${MOTIS_STACKTRACE},AUTO>,$<CONFIG:Debug>>>")

if (NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    if (MOTIS_MIMALLOC)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        set(protobuf_MSVC_STATIC_RUNTIME OFF)
    else ()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        set(protobuf_MSVC_STATIC_RUNTIME ON)
    endif ()
endif ()

if (MOTIS_MIMALLOC)
    set(CISTA_USE_MIMALLOC ON)
    set(PPR_MIMALLOC ON)
    set(ADR_MIMALLOC ON)
    set(OSR_MIMALLOC ON)
    set(TILES_MIMALLOC ON)
    if(WIN32)
        set(MI_BUILD_SHARED ON)
    endif()
endif()

include(cmake/buildcache.cmake)
include(cmake/pkg.cmake)

if (MOTIS_MIMALLOC)
    if(WIN32)
        set(motis-mimalloc-lib mimalloc)
        target_link_libraries(cista INTERFACE mimalloc)
    else()
        set(motis-mimalloc-lib mimalloc-obj)
        target_link_libraries(cista INTERFACE mimalloc-static)
    endif()
    target_compile_definitions(cista INTERFACE CISTA_USE_MIMALLOC=1)
    target_compile_definitions(boost INTERFACE BOOST_ASIO_DISABLE_STD_ALIGNED_ALLOC=1)
endif()

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
            -Wno-shadow-uncaptured-local
            -Wno-documentation-deprecated-sync
            -Wno-float-equal
            -Wno-deprecated-declarations
            -Wno-reserved-identifier
            -Wno-implicit-int-float-conversion
            -Wno-nrvo
            -Wno-thread-safety-negative
            -Wno-unused-private-field
            -Werror)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(motis-compile-options -Wall -Wextra -Werror -Wno-unknown-pragmas -Wno-deprecated-declarations)
elseif (MSVC)
    set(motis-compile-options /WX /bigobj)
else ()
    set(motis-compile-options
            -Wall
            -Wextra
            -Wno-mismatched-new-delete
            -Wno-maybe-uninitialized)
    if (NOT CMAKE_CROSSCOMPILING)
        set(motis-compile-options ${motis-compile-options} -Werror)
    endif ()
endif ()


# --- OPENAPI ---
openapi_generate(openapi.yaml motis-api motis::api)


# --- LIB ---
file(GLOB_RECURSE motislib-files src/*.cc)
add_library(motislib ${motislib-files})
target_include_directories(motislib PUBLIC include)
target_compile_features(motislib PUBLIC cxx_std_23)
target_compile_options(motislib PRIVATE ${motis-compile-options})
target_link_libraries(motislib
    nigiri
    osr
    adr
    ctx
    boost-json
    motis-api
    reflectcpp
    web-server
    tiles
    pbf_sdf_fonts_res
    ssl
    crypto
    tg
    lz4_static
    lb
    web-server
    prometheus-cpp::core
    opentelemetry_trace
    opentelemetry_exporter_otlp_http
    Boost::stacktrace
    "$<${_MOTIS_STACKTRACE_ON}:Boost::stacktrace_from_exception>"
)


# --- EXE ---
execute_process(
    COMMAND git describe --always --tags --dirty=-dirty
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE motis-git-tag
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
file(GLOB_RECURSE motis-files exe/*.cc)
add_executable(motis ${motis-files})
target_compile_features(motis PUBLIC cxx_std_23)
target_compile_options(motis PRIVATE ${motis-compile-options})
set_source_files_properties(exe/main.cc PROPERTIES COMPILE_DEFINITIONS MOTIS_VERSION="${motis-git-tag}")
target_link_libraries(motis
    motislib
    ianatzdb-res
    pbf_sdf_fonts_res-res
    tiles_server_res-res
    address_formatting_res-res
)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    target_link_options(motis PRIVATE -Wl,-no_deduplicate)
endif()

# --- TEST ---
add_library(motis-generated INTERFACE)
target_include_directories(motis-generated INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dir.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/generated/test_dir.h
)
file(GLOB_RECURSE motis-test-files test/*.cc)
add_executable(motis-test ${motis-test-files})
target_link_libraries(motis-test motislib gtest web-server ianatzdb-res motis-generated)
target_compile_options(motis-test PRIVATE ${motis-compile-options})


# --- TILES ---
set_property(
    TARGET motis tiles tiles-import-library
    APPEND PROPERTY COMPILE_DEFINITIONS TILES_GLOBAL_PROGRESS_TRACKER=1)
file (CREATE_LINK ${CMAKE_SOURCE_DIR}/deps/tiles/profile ${CMAKE_BINARY_DIR}/tiles-profiles SYMBOLIC)

# --- MIMALLOC ---
if (MOTIS_MIMALLOC)
    target_link_libraries(motis ${motis-mimalloc-lib})
    target_compile_definitions(motis PUBLIC USE_MIMALLOC=1)
    if(WIN32)
        add_custom_command(
                TARGET motis POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy
                $<TARGET_FILE:mimalloc>
                $<TARGET_FILE_DIR:motis>
                COMMENT "Copy mimalloc.dll to output directory"
        )
        add_custom_command(
                TARGET motis POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy
                "${CMAKE_SOURCE_DIR}/deps/mimalloc/bin/mimalloc-redirect.dll"
                $<TARGET_FILE_DIR:motis>
                COMMENT "Copy mimalloc-redirect.dll to output directory"
        )
        add_custom_command(
                TARGET motis POST_BUILD
                COMMAND "${CMAKE_SOURCE_DIR}/deps/mimalloc/bin/minject.exe"
                --force --inplace
                $<$<CONFIG:Debug>:--postfix=debug>
                $<TARGET_FILE:motis>
                COMMENT "Ensure mimalloc.dll is loaded first"
        )
        add_custom_command(
                TARGET motis-test POST_BUILD
                COMMAND "${CMAKE_SOURCE_DIR}/deps/mimalloc/bin/minject.exe"
                --force --inplace
                $<$<CONFIG:Debug>:--postfix=debug>
                $<TARGET_FILE:motis-test>
                COMMENT "Ensure mimalloc.dll is loaded first"
        )
    endif()
    if (MSVC)
        target_link_options(motis PUBLIC "/include:mi_version")
    endif ()
endif()


# --- UI ---
add_custom_target(motis-web-ui
        COMMAND pnpm install && pnpm -r build
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ui"
        VERBATIM
)
file (CREATE_LINK ${CMAKE_SOURCE_DIR}/ui/build ${CMAKE_BINARY_DIR}/ui SYMBOLIC)

foreach(t adr osr nigiri gtfsrt
        geo tiles tiles-import-library
        motis motis-api motislib)
    target_compile_options(${t} PUBLIC ${MOTIS_TARGET_FLAGS})
endforeach()
if (MOTIS_MIMALLOC)
    target_compile_options(mimalloc PUBLIC ${MOTIS_TARGET_FLAGS})
endif()
