<p align="center"><img src="logo.svg" width="196" height="196"></p>

![Linux Build](https://github.com/motis-project/motis/workflows/Linux%20Build/badge.svg)
![Windows Build](https://github.com/motis-project/motis/workflows/Windows%20Build/badge.svg)

MOTIS stands for **M**ulti **O**bjective **T**ravel **I**nformation **S**ystem.

The core features are:

  - **Intermodal Routing**: computing optimal journeys mixing public transit, sharing mobility, walking, etc. in sensible ways. [Read more.](https://motis-project.de/docs/features/routing.html)
  - **Real Time Support**: considering delays, train cancellations, additional services, reroutings, track changes, etc. [Read more.](https://motis-project.de/docs/features/realtime.html#real-time-support)
  - **Visualization**: view vehicle movements in real-time. [Try it out!](https://demo.motis-project.de/public/)
  - **JSON API**: the backend provides a JSON API via HTTP. [Documentation](https://motis-project.de/docs/api/)
  
More detailed information can be found at [motis-project.de](https://motis-project.de).

To demonstrate the functionalities, an [Android App](https://play.google.com/store/apps/details?id=de.motis_project.app2) and a [web-based information system](https://demo.motis-project.de/) is available. The source code for both front-ends is available as Open Source Software [as well](https://github.com/motis-project/motis/tree/master/scripts).

The system can consume schedule timetables in the [GTFS](https://developers.google.com/transit/gtfs/) or [HAFAS](https://www.fahrplanfelder.ch/fileadmin/fap_daten_test/hrdf.pdf) format as well as real time information in the [GTFS-RT](https://developers.google.com/transit/gtfs-realtime/reference) (and RISML, a propriatary format at Deutsche Bahn) as input data. For pedestrian routing (handled by [Per Pedes Routing](https://github.com/motis-project/ppr)) and car routing (handled by [OSRM](https://github.com/Project-OSRM/osrm-backend)) OpenStreetMap data is used.

# Installation and Setup

Tested on: Linux (Ubuntu 18.04) and Windows 10

To run your own MOTIS instance, you need an OpenStreetMap dataset and a timetable in either the GTFS or the HAFAS Rohdaten format. Note that currently, MOTIS supports only certain HAFAS Rohdaten versions (notably a version in use at Deutsche Bahn as well the one provided at [opentransportdata.swiss](https://opentransportdata.swiss)) and not all GTFS features.

  - Download the latest OpenStreetMap dataset for Swizerland in the ".osm.pbf" format from [geofabrik.de](https://download.geofabrik.de/europe/switzerland.html) and put it into your `data` folder.
  - Download the latest dataset HAFAS Rohdaten dataset from [opentransportdata.swiss](https://opentransportdata.swiss/en/dataset) and extract it into your `data/hrd` folder.

### Linux

Tested on Ubuntu 18.04.

  - **Step 1**: Download and unzip the latest release: [motis](https://github.com/motis-project/motis/releases/latest/download/motis-linux.zip)
  - **Step 2**: Start MOTIS with `./motis --dataset.path data/hrd`


### Windows

Start a PowerShell or cmd.exe prompt:

  - **Step 1**: Download and unzip the latest release: [motis](https://github.com/motis-project/motis/releases/latest/download/motis-windows.zip)
  - **Step 2**: Start MOTIS with `motis.exe --dataset.path data/hrd`


# API Documentation

The API documentation can be found [here](https://motis-project.de/api/).


# Developer Setup

This section describes the steps to compile MOTIS from source.

The build steps as well as the list of build targets (binaries) required for a full MOTIS distribution are listed in the Continuous Integration (CI) configurations:

  - [Linux](https://github.com/felixguendling/motis/blob/master/.github/workflows/linux.yml)
  - [Windows](https://github.com/felixguendling/motis/blob/master/.github/workflows/windows.yml)

## Windows Developer Setup

In the following, we list requirements and a download link. There may be other sources (like package managers) to install these and other software (for example other archive tools than 7zip, other Git distributions, etc.).

  - 7zip: [7-zip.org](https://www.7-zip.org/)
  - Boost 1.72.0: [boost.org](https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.7z)
  - CMake 3.16: [cmake.org](https://cmake.org/download/)
  - Git: [git-scm.com](https://git-scm.com/download/win)
  - Visual Studio 2019 or at least "Build Tools for Visual Studio 2019": [visualstudio.microsoft.com](https://visualstudio.microsoft.com/de/downloads/)
  - zlib: [zlib.net](https://www.zlib.net/)

After you have installed 7zip, CMake, Git and Visual Studio listed above, follow these steps:

**Build zlib**:

  - Extract the zlib package to `C:\zlib-1.2.11`
  - Start the 64bit Developer Command Prompt
  - `cd C:\zlib-1.2.11`
  - `cmake -DCMAKE_BUILD_TYPE=Release .`
  - `cmake --build .`

**Build Boost**:

  - Extract Boost to `C:\boost_1_72_0` (other paths will work too - adjust in further instructions if you change it)
  - Start the 64bit Developer Command Prompt
  - `cd C:\boost_1_72_0`
  - `bootstrap`
  - `b2 --with-system --with-filesystem --with-chrono --with-thread --with-date_time --with-regex --with-filesystem --with-iostreams --with-program_options  threading=multi link=static runtime-link=static address-model=64 -s ZLIB_SOURCE=C:\zlib-1.2.11`

**Build MOTIS**:

  - `git clone git@github.com:motis-project/motis.git`
  - `cd motis`
  - `cmake -G "Visual Studio 16 2019" -A x64 -S . -B build`
  - `cmake --build build --config Release --target motis motis-test motis-itest path-prepare deps/ppr/ppr-preprocess deps/osrm-backend/osrm-extract deps/osrm-backend/osrm-contract deps/address-typeahead/at-example`


## Linux Developer Setup

On Linux, there are different options. Either you install every dependency into your system or you use the Docker container that is also used for CI builds.

**Docker**:

Tested with Ubuntu 18.04.

  - Install Docker from [docker.com](https://docs.docker.com/engine/install/).
  - Install Git: `sudo apt install git`
  - `git clone git@github.com:motis-project/motis.git`
  - `cd motis`

Build:

    docker run \              
      -v "$PWD:/repo" \
      -e CCACHE_DIR=/repo/ccache \
      --rm motisproject/cpp-build:latest \
      bash -c "ccache -z && cmake-ccache-clang-9 -DCMAKE_BUILD_TYPE=Release /repo && ninja motis && cp motis /repo && ccache -s"

You should have the MOTIS binary now in the "motis" folder.


**Directly**:

Execute the steps from the [Dockerfile](https://github.com/motis-project/docker/blob/master/Dockerfile) manually. This installs all dependencies. You need to download the [binaries](https://github.com/motis-project/docker/tree/master/blob) manually.

  - `git clone git@github.com:motis-project/motis.git`
  - `cd motis`
  - `mkdir build && cd build`
  - `cmake -DCMAKE_BUILD_TYPE=Release -GNinja ..`
  - `ninja`

This builds the MOTIS binary.


# Contribution

Feel free to contribute in any area you like (new features, improvments, documentation, testing, etc.)!
By making a pull-request you agree to license your contribution under the MIT and Apache 2.0 license as described below.


# Alternatives

  - [OpenTripPlanner](https://www.opentripplanner.org/)
  - [navitia.io](https://github.com/CanalTP/navitia)


# License

MOTIS can be licensed under the terms of the MIT license or under the terms of the Apache License, Version 2.0.
