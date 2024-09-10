cmake_minimum_required(VERSION 3.21)
project(motis)

enable_language(CXX C ASM)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

if (MSVC)
  # PDB debug information is not supported by buildcache.
  # Store debug info in the object files.
  option(MOTIS_DEBUG_SYMBOLS "generate debug symbols (debug builds)" ON)
  if (MOTIS_DEBUG_SYMBOLS)
    set(MOTIS_MSVC_DEBUG_FLAGS "/Z7")
  else()
    set(MOTIS_MSVC_DEBUG_FLAGS "")
  endif()
  string(REPLACE "/Zi" "${MOTIS_MSVC_DEBUG_FLAGS}" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
  string(REPLACE "/Zi" "${MOTIS_MSVC_DEBUG_FLAGS}" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  string(REPLACE "/Zi" "${MOTIS_MSVC_DEBUG_FLAGS}" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  string(REPLACE "/Zi" "${MOTIS_MSVC_DEBUG_FLAGS}" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()
include(cmake/buildcache.cmake)

if (APPLE)
  # prevents ar from invoking ranlib, let CMake do it
  set(CMAKE_C_ARCHIVE_CREATE   "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")

  # tell ranlib to ignore empty compilation units
  set(CMAKE_C_ARCHIVE_FINISH   "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
  set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")

  # Allow deprecated auto_ptr (e.g. in luabind)
  add_definitions(-D_LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR)

  # Silence linking warnings.
  add_compile_options(-fvisibility-inlines-hidden -fvisibility=hidden)
endif()

option(MOTIS_MIMALLOC "use mimalloc" OFF)
option(MOTIS_AVX "enable AVX functions" ON)
option(MOTIS_AVX2 "enable AVX2 + FMA functions" ON)
option(MOTIS_CUDA "enable CUDA functions" OFF)
option(MOTIS_LINT "enable lint (clang-tidy) target" OFF)
option(MOTIS_COV "enable coverage (coverage) target" OFF)
option(MOTIS_WITH_WEBUI "enable motis-web-ui target" OFF)

if(NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
  if(MOTIS_MIMALLOC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    set(protobuf_MSVC_STATIC_RUNTIME OFF)
  else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set(protobuf_MSVC_STATIC_RUNTIME ON)
  endif()
endif()

if (MOTIS_CUDA)
  if (MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcudadevrt")
  else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -static")
  endif()
  enable_language(CUDA)
endif()

add_library(motis-generated INTERFACE)
target_include_directories(motis-generated INTERFACE ${CMAKE_BINARY_DIR}/generated)

set(USE_OPENMP OFF CACHE BOOL openmp FORCE)


################################
# MOTIS Version
################################
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE motis_git_branch
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND git --no-pager log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE motis_git_ref
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND git describe --always --tags --dirty=-dirty
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE motis_git_tag
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

configure_file (
  "base/core/include/motis/core/version.h.in"
  "${CMAKE_BINARY_DIR}/generated/version.h"
)


################################
# Dependencies
################################
if (MOTIS_MIMALLOC)
  set(PPR_MIMALLOC ON)
  set(ADR_MIMALLOC ON)
  set(OSR_MIMALLOC ON)
  if(WIN32)
    set(MI_BUILD_SHARED ON)
  endif()
endif()

find_package(Threads)
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


################################
# Basic Compiler Flags
################################
if (NOT MSVC)
  set(MOTIS_CXX_FLAGS -Wall -Wextra -Wno-sign-compare)
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    set(MOTIS_CXX_FLAGS ${MOTIS_CXX_FLAGS} -Wno-maybe-uninitialized -Wno-unused-but-set-parameter)
  endif()
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16)
      set(MOTIS_CXX_FLAGS ${MOTIS_CXX_FLAGS} -Wno-unsafe-buffer-usage)
    endif()
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 18)
      set(MOTIS_CXX_FLAGS ${MOTIS_CXX_FLAGS} -Wno-switch-default)
    endif()
  endif()
  if (MOTIS_COV)
    set(MOTIS_CXX_FLAGS ${MOTIS_CXX_FLAGS} --coverage)
    set(CMAKE_EXE_LINKER_FLAGS  ${CMAKE_EXE_LINKER_FLAGS} --coverage)
  endif()
endif()

# Quickfix for GCC 7
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -Wno-implicit-fallthrough")
endif()

# Quickfix for MSVC
if (MSVC)
  set(MOTIS_COMPILE_DEFINITIONS
    WIN32_LEAN_AND_MEAN
    _HAS_AUTO_PTR_ETC
    _USE_MATH_DEFINES
    NOMINMAX
    ${MOTIS_COMPILE_DEFINITIONS})
endif()

# MOTIS Compile Definitions
if (MOTIS_CUDA)
  set(MOTIS_COMPILE_DEFINITIONS MOTIS_CUDA ${MOTIS_COMPILE_DEFINITIONS})
endif()
if (MOTIS_AVX)
  set(MOTIS_COMPILE_DEFINITIONS MOTIS_AVX ${MOTIS_COMPILE_DEFINITIONS})
  if (MSVC)
    set(MOTIS_CXX_FLAGS ${MOTIS_CXX_FLAGS} /arch:AVX)
  else()
    set(MOTIS_CXX_FLAGS ${MOTIS_CXX_FLAGS} -mavx)
  endif()
endif()
if (MOTIS_AVX2)
  set(MOTIS_COMPILE_DEFINITIONS MOTIS_AVX2 ${MOTIS_COMPILE_DEFINITIONS})
  if (MSVC)
    set(MOTIS_CXX_FLAGS ${MOTIS_CXX_FLAGS} /arch:AVX2)
  else()
    set(MOTIS_CXX_FLAGS ${MOTIS_CXX_FLAGS} -mavx2 -mfma -mbmi)
  endif()
endif()


################################
# Linter
################################
option(MOTIS_LINT "Run clang-tidy with the compiler." OFF)
if(MOTIS_LINT)
  # clang-tidy will be run on all targets defined hereafter
  set(MOTIS_SCHEDULE_MODE "raw")
  add_definitions(-DMOTIS_LINT=1)
  include(cmake/clang-tidy.cmake)
endif()

################################
# MOTIS
################################
add_subdirectory(base/data EXCLUDE_FROM_ALL)
add_subdirectory(base/core EXCLUDE_FROM_ALL)
add_subdirectory(base/module EXCLUDE_FROM_ALL)

file(GLOB all-modules RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR}/modules/*)
list(REMOVE_ITEM all-modules ".DS_Store")
string(REPLACE ";" " " modules-option "${all-modules}")
set(MODULES ${modules-option} CACHE STRING "Modules to load")

if(MODULES STREQUAL "all")
  set(modules ${all-modules})
else()
  string(REPLACE " " ";" modules "${MODULES}")
endif()
message(STATUS "compiling modules: ${modules}")

add_subdirectory(libs)

set(module-targets "")
set(module-test-files "")
foreach(module ${modules})
  add_subdirectory(modules/${module})
  list(APPEND module-targets "motis-${module}")
  list(APPEND module-test-files "modules/${module}/*_test.cc")
  list(APPEND module-itest-files "modules/${module}/*_itest.cc")
endforeach(module)

add_subdirectory(base/bootstrap EXCLUDE_FROM_ALL)

file(GLOB_RECURSE motis-launcher-files base/launcher/src/*.cc)
add_executable(motis ${motis-launcher-files})
target_compile_features(motis PUBLIC cxx_std_20)
target_include_directories(motis PRIVATE base/launcher/include)
target_link_libraries(motis
  ${CMAKE_THREAD_LIBS_INIT}
  boost-system
  conf
  motis-bootstrap
  web-server-tls
  ianatzdb-res
  pbf_sdf_fonts_res-res
  tiles_server_res-res
  opentelemetry_exporter_otlp_http
)
target_compile_options(motis PRIVATE ${MOTIS_CXX_FLAGS})
set_target_properties(motis PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if (NOT MSVC)
  set_target_properties(motis PROPERTIES LINK_FLAGS "-Wl,-rpath,./")

  if (${CMAKE_BUILD_TYPE} STREQUAL Release)
    add_custom_command(
      TARGET motis
      POST_BUILD
      COMMAND ${CMAKE_STRIP} ${CMAKE_BINARY_DIR}/motis
    )
  endif()
endif()

if (MOTIS_MIMALLOC)
  target_link_libraries(motis ${motis-mimalloc-lib})
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
        "${CMAKE_CURRENT_BINARY_DIR}/deps/mimalloc/mimalloc-redirect.dll"
        $<TARGET_FILE_DIR:motis>
        COMMENT "Copy mimalloc-redirect.dll to output directory"
    )
  endif()
endif()

if(MOTIS_WITH_WEBUI)
  add_subdirectory(ui/web)
endif()


################################
# Tests
################################
configure_file (
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dir.h.in
  ${CMAKE_BINARY_DIR}/generated/test_dir.h
)
file(GLOB_RECURSE motis-test-files test/src/*.cc)
file(GLOB_RECURSE motis-modules-test-files ${module-test-files})
file(GLOB_RECURSE motis-base-test-files base/*_test.cc)
add_executable(motis-test EXCLUDE_FROM_ALL
  ${motis-test-files}
  ${motis-modules-test-files}
  ${motis-base-test-files}
)
target_include_directories(motis-test PRIVATE test/include)
if (MSVC)
  target_compile_features(motis-test PUBLIC cxx_std_20)
endif()

list(FIND modules "path" MOTIS_PATH_MODULE_ENABLE)
if(${MOTIS_PATH_MODULE_ENABLE} GREATER -1)
  set(motis-test-extra-dependencies "motis-path-prepare")
endif()

target_compile_options(motis-test PRIVATE ${MOTIS_CXX_FLAGS})
target_compile_definitions(motis-test PRIVATE ${MOTIS_COMPILE_DEFINITIONS})
target_link_libraries(motis-test
  ${motis-test-extra-dependencies}
  motis-bootstrap
  motis-core
  motis-module
  conf
  ianatzdb-res
)
target_link_libraries(motis-test gtest gtest_main gmock)


################################
# Integration Tests
################################
configure_file (
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dir.h.in
  ${CMAKE_BINARY_DIR}/generated/test_dir.h
)
file(GLOB_RECURSE motis-modules-itest-files ${module-itest-files})
file(GLOB_RECURSE motis-base-itest-files base/*_itest.cc)

add_executable(motis-itest EXCLUDE_FROM_ALL
  ${motis-test-files}
  ${motis-modules-itest-files}
  ${motis-base-itest-files})
target_include_directories(motis-itest PRIVATE test/include)
if (MSVC)
  target_compile_features(motis-itest PUBLIC cxx_std_20)
endif()
target_compile_options(motis-itest PRIVATE ${MOTIS_CXX_FLAGS})
target_compile_definitions(motis-itest PRIVATE ${MOTIS_COMPILE_DEFINITIONS})
target_link_libraries(motis-itest
  ${module-targets}
  motis-bootstrap
  motis-core
  motis-module
  conf
  ianatzdb-res
  boost-filesystem
)
target_link_libraries(motis-itest gtest gtest_main gmock)
set_target_properties(motis-itest PROPERTIES VS_DEBUGGER_COMMAND_ARGUMENTS "--gtest_filter=\"ris_gtfsrt_cancel_message_itest_t0.before_cancel\"")


################################
# Lint (oclint)
################################
add_custom_target(oclint
  find
    ./modules/routing/ ./modules/guesser/ ./modules/osrm/ ./base/
    -type f
    -name "*.cc"
    ! -name "*test*"
    -print0
    |
    xargs
      -0
      oclint
        -enable-global-analysis
        -p ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Running oclint"
  VERBATIM
)
add_dependencies(oclint motis motis-test motis-itest)
if (TARGET motis-eval)
  add_dependencies(oclint motis-eval)
endif()


################################
# clang-format check
################################
find_program(CLANG_FORMAT_COMMAND NAMES clang-format clang-format-16)
add_custom_target(format-check
  COMMAND find
      ${CMAKE_CURRENT_SOURCE_DIR}/base
      ${CMAKE_CURRENT_SOURCE_DIR}/modules
      ${CMAKE_CURRENT_SOURCE_DIR}/test
    -type f
    (
    -name "*.cc"
    -o
    -name "*.h"
    )
    -print0
  | xargs -0 ${CLANG_FORMAT_COMMAND} -i
  COMMAND git status --porcelain
  COMMAND git status --porcelain | xargs -I {} -0 test -z \"{}\"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Checking source code formatting"
  VERBATIM
)


################################
# coverage report
################################
if (MOTIS_COV)
  set(LCOV_OPTS "")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(LCOV_OPTS "--gcov-tool=${CMAKE_CURRENT_SOURCE_DIR}/tools/llvm-gcov.sh")
  endif()

  add_custom_target(coverage
    # clean previous files
    rm -rf ./combined_coverage ./motis-test_coverage ./motis-itest_coverage
    COMMAND rm -f combined-test.info motis-test.info motis-itest.info
    COMMAND find . -name "*.gcda" -delete

    # combined (motis-test + motis-itest)
    COMMAND lcov --base-directory . --directory . --zerocounters -q
    COMMAND ./motis-test
    COMMAND ./motis-itest
    COMMAND lcov ${LCOV_OPTS} --base-directory . --directory . -c -o combined-test.info
    COMMAND lcov --remove combined-test.info "/usr*" -o combined-test.info
    COMMAND lcov --remove combined-test.info "/external_lib*" -o combined-test.info
    COMMAND lcov --remove combined-test.info "/test/*" -o combined-test.info
    COMMAND genhtml -o combined_coverage -t "motis-test + motis-itest" --num-spaces 4 combined-test.info

    # only motis-test
    COMMAND lcov --base-directory . --directory . --zerocounters -q
    COMMAND ./motis-test
    COMMAND lcov ${LCOV_OPTS} --base-directory . --directory . -c -o motis-test.info
    COMMAND lcov --remove motis-test.info "/usr*" -o motis-test.info
    COMMAND lcov --remove motis-test.info "/external_lib*" -o motis-test.info
    COMMAND lcov --remove motis-test.info "/test/*" -o motis-test.info
    COMMAND genhtml -o motis-test_coverage -t "motis-test" --num-spaces 4 motis-test.info

    # only motis-itest
    COMMAND lcov --base-directory . --directory . --zerocounters -q
    COMMAND ./motis-itest
    COMMAND lcov ${LCOV_OPTS} --base-directory . --directory . -c -o motis-itest.info
    COMMAND lcov --remove motis-itest.info "/usr*" -o motis-itest.info
    COMMAND lcov --remove motis-itest.info "/external_lib*" -o motis-itest.info
    COMMAND lcov --remove motis-itest.info "/test/*" -o motis-itest.info
    COMMAND genhtml -o motis-itest_coverage -t "motis-itest" --num-spaces 4 motis-itest.info
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating coverage report"
    VERBATIM
  )
  add_dependencies(coverage motis-test motis-itest)
endif()
