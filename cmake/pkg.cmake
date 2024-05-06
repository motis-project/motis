if (NOT DEFINED PROJECT_IS_TOP_LEVEL OR PROJECT_IS_TOP_LEVEL)
  find_program(pkg-bin pkg HINTS /opt/pkg)
  if (pkg-bin)
    message(STATUS "found pkg ${pkg-bin}")
  else ()
    set(pkg-bin "${CMAKE_BINARY_DIR}/dl/pkg")
    if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
      set(pkg-url "pkg-linux-arm64")
    elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
      set(pkg-url "pkg")
    elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
      set(pkg-url "pkg.exe")
    elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
      set(pkg-url "pkgosx")
    else ()
      message(STATUS "Not downloading pkg tool. Using pkg from PATH.")
      set(pkg-bin "pkg")
    endif ()

    if (pkg-url)
      if (NOT EXISTS ${pkg-bin})
        message(STATUS "Downloading pkg binary from https://github.com/motis-project/pkg/releases/latest/download/${pkg-url}")
        file(DOWNLOAD "https://github.com/motis-project/pkg/releases/latest/download/${pkg-url}" ${pkg-bin})
        if (UNIX)
          execute_process(COMMAND chmod +x ${pkg-bin})
        endif ()
      else ()
        message(STATUS "Pkg binary located in project.")
      endif ()
    endif ()
  endif ()

  message(STATUS ${pkg-bin})

  if (DEFINED ENV{GITHUB_ACTIONS})
    execute_process(
      COMMAND ${pkg-bin} -l -h
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
  else ()
    execute_process(
      COMMAND ${pkg-bin} -l
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
  endif ()
endif ()

if (IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/deps")
  add_subdirectory(deps)
endif ()
