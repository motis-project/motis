if(CMake_SOURCE_DIR STREQUAL CMake_BINARY_DIR)
  message(FATAL_ERROR "CMake_RUN_CLANG_TIDY requires an out-of-source build!")
endif()

file(RELATIVE_PATH RELATIVE_SOURCE_DIR ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR})

if(NOT "${RELATIVE_SOURCE_DIR}" MATCHES ".+/$")
  set(RELATIVE_SOURCE_DIR "${RELATIVE_SOURCE_DIR}/")
endif()

if(MOTIS_CLANG_TIDY_COMMAND)
  set(CLANG_TIDY_COMMAND "${MOTIS_CLANG_TIDY_COMMAND}")
else()
  find_program(CLANG_TIDY_COMMAND NAMES clang-tidy clang-tidy-18)
endif()

if(NOT CLANG_TIDY_COMMAND)
  message(FATAL_ERROR "CMake_RUN_CLANG_TIDY is ON but clang-tidy is not found!")
endif()

if(DEFINED ENV{CI})
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
else()
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}" --format-style=file --fix-errors)
endif()

file(SHA1 ${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy.in clang_tidy_sha1)
set(CLANG_TIDY_DEFINITIONS "CLANG_TIDY_SHA1=${clang_tidy_sha1}")
unset(clang_tidy_sha1)

configure_file(.clang-tidy.in ${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy)
