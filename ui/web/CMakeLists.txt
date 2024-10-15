find_program(elm_make_program elm-make HINTS /opt/elm-make)
if (elm_make_program)
  message(STATUS "found elm-make ${elm_make_program}")
  add_custom_target(motis-web-ui
    COMMAND "${elm_make_program}" --yes src/Main.elm --output elm.js
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VERBATIM
  )
else()
  if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(elm-platform "linux-64bit")
    set(elm-make "${CMAKE_BINARY_DIR}/dl/elm-make")
  elseif (${CMAKE_SYSTEM_NAME} STREQUAL  "Windows")
    set(elm-platform "windows")
    set(elm-make "${CMAKE_BINARY_DIR}/dl/elm-make.exe")
  elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(elm-platform "macos")
    set(elm-make "${CMAKE_BINARY_DIR}/dl/elm-make")
  else()
    message(WARNING "No Elm binary available for this platform.")
  endif()

  if (elm-platform)
    if (NOT EXISTS ${elm-make})
      set(elm-url "https://github.com/elm-lang/elm-platform/releases/download/0.18.0-exp/elm-platform-${elm-platform}.tar.gz")
      message(STATUS "Downloading Elm from ${elm-url}")
      file(DOWNLOAD "${elm-url}" "${CMAKE_BINARY_DIR}/dl/elm.tar.gz")
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_BINARY_DIR}/dl/elm.tar.gz"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/dl"
      )
    endif()

    add_custom_target(motis-web-ui
      COMMAND "${elm-make}" --yes src/Main.elm --output elm.js
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      VERBATIM
    )
  endif()
endif()
