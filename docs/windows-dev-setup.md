In the following, we list requirements and a download link. There may be other sources (like package managers) to install these.

- CMake 3.17 (or newer): [cmake.org](https://cmake.org/download/)
- Git: [git-scm.com](https://git-scm.com/download/win)
- Visual Studio 2022 or at least "Build Tools for Visual Studio 2022": [visualstudio.microsoft.com](https://visualstudio.microsoft.com/de/downloads/)
- Ninja: [ninja-build.org](https://ninja-build.org/)

> [!CAUTION]
> Motis' dependency management `pkg` requires that the project is cloned via SSH using an SSH key without a passphrase.
> See:
> - [GitHub Docs: Generate new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
> - [GitHub Docs: Add a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

## Build MOTIS using the command line

Start menu -> `Visual Studio 2022` -> `x64 Native Tools Command Prompt for VS 2022`, then enter:

```bat
git clone "git@github.com:motis-project/motis.git"
cd motis
mkdir build
cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja
```

## Build MOTIS using CLion

- Make sure that the architecture is set to `amd64` (Settings -> `Build, Execution, Deployment` -> `Toolchains`).
- You might have to allow users to create symbolic links. Open `Local Security Policy` and go to `User Rights Assignment`.
- You might have to enable `Developer Mode` under `Advanced System Settings`.
