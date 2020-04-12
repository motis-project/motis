<p align="center"><img src="logo.svg" width="196" height="196"></p>

![Linux Build](https://github.com/motis-project/motis/workflows/Linux%20Build/badge.svg)
![Windows Build](https://github.com/motis-project/motis/workflows/Windows%20Build/badge.svg)

MOTIS stands for **M**ulti **O**bjective **T**ravel **I**nformation **S**ystem.

The core features are:

  - **Intermodal Routing**: computing optimal journeys mixing public transit, sharing mobility, walking, etc. in sensible ways
  - **Real Time Support**: considering delays, train cancellations, additional services, reroutings, track changes, etc.
  - **Visualization**: view vehicle movements in real-time
  - **JSON API**: the backend provides a JSON API via HTTP
  
More detailed information can be found at [motis-project.de](https://motis-project.de).

To demonstrate the functionalities, an [Android App](https://play.google.com/store/apps/details?id=de.motis_project.app2) and a [web-based information system](https://demo.motis-project.de/) is available. The source code for both front-ends is available as Open Source Software [as well](https://github.com/motis-project/motis/tree/master/scripts).

The system can consume schedule timetables in the [GTFS](https://developers.google.com/transit/gtfs/) or [HAFAS](https://www.fahrplanfelder.ch/fileadmin/fap_daten_test/hrdf.pdf) format as well as real time information in the [GTFS-RT](https://developers.google.com/transit/gtfs-realtime/reference) (and RISML, a propriatary format at Deutsche Bahn) as input data. For pedestrian routing (handled by [Per Pedes Routing](https://github.com/motis-project/ppr)) and car routing (handled by [OSRM](https://github.com/Project-OSRM/osrm-backend)) OpenStreetMap data is used.

# Installation and Setup

Tested on: Linux (Ubuntu 18.04) and Windows 10

  - **Step 1**: Download the latest release for your operating system:
      - Windows: [motis.exe](https://github.com/motis-project/motis/releases/latest/download/motis.exe)
      - Linux: [motis](https://github.com/motis-project/motis/releases/latest/download/motis) (run `chmod +x motis` to make the binary executable)
  - **Step 2**: Download the latest dataset from [opentransportdata.swiss](https://opentransportdata.swiss/en/dataset) in the GTFS format and extract the archive next to the MOTIS binary.
  - **Step 3**: Download the latest OpenStreetMap dataset for Swizerland in the ".osm.pbf" format from [geofabrik.de](https://download.geofabrik.de/europe/switzerland.html) and place it in the same folder as the MOTIS binary and the extracted GTFS timetable.
  - **Step 4**: Start MOTIS:
    - Windows: `motis.exe`
    - Linux: `./motis --dataset.path .`
    
The system should now be up and running. You can access the web interface at [http://localhost:8080](http://localhost:8080).


# API Documentation

The API documentation can be found [here](https://motis-project.de/api/).


# Developer Setup

This section describes the steps to compile MOTIS from source.

## Windows

In the following, we list requirements and a download link. There may be other sources (like package managers) to install these and other software (for example other archive tools than 7zip, other Git distributions, etc.).

  - 7zip: [Download from 7zip](https://www.7-zip.org/)
  - Boost 1.72.0: [Download from boost.org](https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.7z)
  - CMake 3.16: [Download from cmake.org](https://cmake.org/download/)
  - Git: [git-scm.com](https://git-scm.com/download/win)
  - Visual Studio 2019 or at least "Build Tools for Visual Studio 2019": [visualstudio.microsoft.com](https://visualstudio.microsoft.com/de/downloads/)
  - zlib: [zlib.net](https://www.zlib.net/)

After you have installed 7zip, CMake, Git and Visual Studio listed above, follow these steps:

Build zlib:

  - Extract the zlib package to `C:\zlib-1.2.11`
  - Start the 64bit Developer Command Prompt
  - `cd C:\zlib-1.2.11`
  - `cmake -DCMAKE_BUILD_TYPE=Release .`
  - `cmake --build .`

Build Boost:

  - Extract Boost to `C:\boost_1_72_0` (other paths will work too - adjust in further instructions if you change it)
  - Start the 64bit Developer Command Prompt
  - `cd C:\boost_1_72_0`
  - `bootstrap`
  - `b2 --with-system --with-filesystem --with-chrono --with-thread --with-date_time --with-regex --with-filesystem --with-iostreams --with-program_options  threading=multi link=static runtime-link=static address-model=64 -s ZLIB_SOURCE=C:\zlib-1.2.11`

Build MOTIS:

  - `git clone git@github.com:motis-project/motis.git`
  - `cd motis`
  - `cmake -G "Visual Studio 16 2019" -A x64 -S . -B build`
  - `cmake --build build --config Release --target motis motis-test motis-itest path-prepare deps/ppr/ppr-preprocess deps/osrm-backend/osrm-extract deps/osrm-backend/osrm-contract deps/address-typeahead/at-example`


## Linux

On Linux, there are different options:

**Docker**:



**Directly**:




# License

MOTIS can be licensed under the terms of the MIT license or under the terms of the Apache License, Version 2.0.
