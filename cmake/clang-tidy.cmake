# Taken from the CMake project (https://gitlab.kitware.com/cmake/cmake/blob/v3.12.4/CMakeLists.txt)
# Distributed under the OSI-approved BSD 3-Clause License. See accompanying https://cmake.org/licensing for details.
if(CMake_SOURCE_DIR STREQUAL CMake_BINARY_DIR)
  message(FATAL_ERROR "CMake_RUN_CLANG_TIDY requires an out-of-source build!")
endif()

file(RELATIVE_PATH RELATIVE_SOURCE_DIR ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR})

if(CLANG_TIDY_NAMESPACES)
  find_program(CLANG_TIDY_COMMAND NAMES clang-tidy-namespace)
  if(NOT CLANG_TIDY_COMMAND)
    message(FATAL_ERROR "CMake_RUN_CLANG_TIDY is ON but clang-tidy-namespace is not found!")
  endif()
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND};-config={Checks: '-*,modernize-concat-nested-namespaces',WarningsAsErrors: '*',HeaderFilterRegex: ^${RELATIVE_SOURCE_DIR}(base/)|(modules/)|(test/)};-fix;-format-style=file")
else()
  find_program(CLANG_TIDY_COMMAND NAMES clang-tidy clang-tidy-9)
  if(NOT CLANG_TIDY_COMMAND)
    message(FATAL_ERROR "CMake_RUN_CLANG_TIDY is ON but clang-tidy-9 is not found!")
  endif()
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
endif()

# Create a preprocessor definition that depends on .clang-tidy content so
# the compile command will change when .clang-tidy changes.  This ensures
# that a subsequent build re-runs clang-tidy on all sources even if they
# do not otherwise need to be recompiled.  Nothing actually uses this
# definition.  We add it to targets on which we run clang-tidy just to
# get the build dependency on the .clang-tidy file.
file(SHA1 ${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy.in clang_tidy_sha1)
set(CLANG_TIDY_DEFINITIONS "CLANG_TIDY_SHA1=${clang_tidy_sha1}")
unset(clang_tidy_sha1)


configure_file(.clang-tidy.in ${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy)