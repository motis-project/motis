# Injected via -DCMAKE_PROJECT_INCLUDE into the ROCm build (Dockerfile.rocm).
#
# nigiri-gpu links hip::device privately, but a static library's private link
# interface still reaches its consumers, so the executables' link lines carry
# hip::device's `--hip-link`, which only clang understands. Host code keeps
# compiling with g++; only the final link of the two executables is handed to
# the HIP toolchain. Deferred because the targets do not exist yet when
# project() runs.
# link rules are directory-scoped: nigiri enables HIP only under deps/nigiri,
# so at the top level (where motis is defined) the HIP link rule would be
# empty and ninja would "link" motis with a no-op
enable_language(HIP)

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
  CALL set_target_properties motis nigiri-benchmark PROPERTIES LINKER_LANGUAGE HIP)
