# syntax=docker/dockerfile:1.7
#
# GPU routing benchmark image: motis + nigiri-benchmark built for one GPU
# target. No data inside -- fetch-data.sh downloads the imported timetable
# (Transitous' nightly one by default) when the container starts.
#
# Build context = the motis repo root with deps/ resolved; docker/ is kept out
# of it (.dockerignore) so editing these files does not invalidate the build
# stage's source copy -- the runtime stage gets its scripts from the named
# context `bench` that build.sh passes (--build-context bench=docker/bench).
# CUDA (default) and, once nigiri has NIGIRI_HIP, ROCm are the same file with
# different BASE_*/CMAKE_GPU_FLAGS build args -- see README.md.
# CUDA 13.3 + g++-15 is what production runs; Ubuntu 26.04 ships gcc-15.
# The runtime stage is plain Ubuntu: the binaries link nothing from the
# toolkit but libcudart (copied over), the driver comes from the host.
ARG BASE_DEVEL=nvidia/cuda:13.3.1-devel-ubuntu26.04
ARG BASE_RUNTIME=ubuntu:26.04

FROM ${BASE_DEVEL} AS build
ARG GCC=15
ARG CMAKE_GPU_FLAGS="-DNIGIRI_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120"
ARG JOBS=
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++-${GCC} cmake ninja-build git python3 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# deps/ arrives already resolved; cmake/pkg.cmake prefers /opt/pkg/pkg, so a
# no-op there keeps it from trying to re-resolve (the .pkg URLs are ssh)
RUN mkdir -p /opt/pkg && printf '#!/bin/sh\nexit 0\n' > /opt/pkg/pkg \
 && chmod +x /opt/pkg/pkg
WORKDIR /src
COPY . .
RUN cmake -G Ninja -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc-${GCC} \
      -DCMAKE_CXX_COMPILER=g++-${GCC} \
      -DCMAKE_CUDA_HOST_COMPILER=g++-${GCC} \
      ${CMAKE_GPU_FLAGS} \
 && cmake --build build -j ${JOBS:-$(nproc)} --target motis nigiri-benchmark

FROM ${BASE_RUNTIME}
ENV DEBIAN_FRONTEND=noninteractive
# vast.ai's ssh launch mode starts sshd inside the container but does not
# promise to install it; python3-yaml is for strip-config.py
RUN apt-get update && apt-get install -y --no-install-recommends \
      openssh-server curl ca-certificates python3 python3-yaml \
 && mkdir -p /run/sshd && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/cuda/lib64/libcudart.so.13* /usr/lib/x86_64-linux-gnu/
COPY --from=build /src/build/motis /src/build/deps/nigiri/nigiri-benchmark /opt/motis/
COPY --from=bench fetch-data.sh entrypoint.sh strip-config.py bake-data.sh part-tar.py /opt/motis/
# what the nvidia/cuda base images set so the host's container toolkit
# injects the GPU; cudart 13.x runs on any driver reporting CUDA >= 13.0
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0" \
    DATA_URL=https://api.transitous.org/gtfs/data \
    DATA_DIR=/data \
    PATH=/opt/motis:$PATH
WORKDIR /data
ENTRYPOINT ["/opt/motis/entrypoint.sh"]
CMD ["bash"]
