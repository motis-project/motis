> [!NOTE]  
> Due to developer capacity constraints we cannot support newer or older compilers.
> We also cannot support other versions of dependencies.
> In case of doubt you can check the full release build setup used in our CI [here](https://github.com/motis-project/docker-cpp-build/blob/master/Dockerfile).
> To exactly reproduce the CI build, you need to use the according preset from our [CMakePresets.json](../CMakePresets.json).

Requirements:

- A recent C++ compiler: Either [Clang](https://llvm.org/) 21 or GCC 13
- CMake 3.17 (or newer): [cmake.org](https://cmake.org/download/) ([Ubuntu APT Repository](https://apt.kitware.com/))
- Ninja: [ninja-build.org](https://ninja-build.org/)
- Git

> [!CAUTION]
> Unix Makefiles are not working. Please use Ninja to build.

> [!CAUTION]
> Motis' dependency management `pkg` requires that the project is cloned via SSH using an SSH key without a passphrase.
> See:
> - [GitHub Docs: Generate new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
> - [GitHub Docs: Add a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

> [!NOTE]
> Using a different compiler version is not officially supported but might work nevertheless when passing
> `--compile-no-warning-as-error` to CMake.

## Build with GCC

```sh
git clone git@github.com:motis-project/motis.git
cd motis
mkdir build && cd build
CXX=g++-13 CC=gcc-13 cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja ..
ninja
```


## Build with Clang

```sh
git clone git@github.com:motis-project/motis.git
cd motis
mkdir build && cd build
CXX=clang++-21 CC=clang-21 CXXFLAGS=-stdlib=libc++ cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja ..
ninja
```
