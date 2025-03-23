Requirements:

- macOS 10.15 or newer
- Command Line Tools for Xcode or Xcode: `xcode-select --install` or [manual download](https://developer.apple.com/downloads)
- [CMake](https://cmake.org/download/) 3.17 (or newer)
- [Ninja](https://ninja-build.org/)
- Git

(Git, Ninja, and CMake can be installed via HomeBrew)

> [!CAUTION]
> Unix Makefiles are not working. Please use Ninja to build.

> [!CAUTION]
> Motis' dependency management `pkg` requires that the project is cloned via SSH using an SSH key without a passphrase.
> See:
> - [GitHub Docs: Generate new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
> - [GitHub Docs: Add a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

To build `motis`:

```sh
git clone git@github.com:motis-project/motis.git
cd motis
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja ..
ninja
```