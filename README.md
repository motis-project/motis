<p align="center"><img src="logo.svg" width="196" height="196"></p>

> [!TIP]
> :sparkles: Join the international MOTIS community at [**motis:matrix.org**](https://matrix.to/#/#motis:matrix.org)

MOTIS stands for **M**odular **O**pen **T**ransportation **I**nformation **S**ystem.
It is an open-source software platform designed to facilitate
efficient planning and routing in multi-modal transportation systems.
Developed to handle *large-scale* transportation data,
MOTIS integrates various modes of transport -
such as walking, cycling, sharing mobility (e-scooters, bike sharing, car
sharing), and public transport -
to provide optimized routing solutions.

MOTIS currently supports the following input formats:

- (One) **OpenStreetMap `osm.pbf`** file for the street network, addresses, indoor-routing, etc. 
- (Multiple) **GTFS** feeds for static timetables
- (Multiple) **GTFS-RT** feeds for real-time updates (delays, cancellations, track changes)
- (Multiple) **GBFS** feeds for sharing mobility

*Planned*: GTFS-Flex, NeTEx and SIRI

MOTIS provides an easy-to-use **REST API** (JSON via HTTP) with
an [**OpenAPI specification**](https://redocly.github.io/redoc/?url=https://raw.githubusercontent.com/motis-project/motis/refs/heads/master/openapi.yaml) ([source](openapi.yaml))
that allows you to generate clients for your favorite programming language.


# Features

> [!NOTE]  
> :rocket: MOTIS is optimized for **high performance** with **low memory usage**.
> 
> This enables _planet-sized_ deployments on affordable hardware.

MOTIS is a swiss army knife for mobility and comes with all features you need for a next generation mobility platform:

- **routing**: one mode walking, bike, car, sharing mobility / combined modes
- **geocoding**: multi-language address and stop name completion with fuzzy string matching and resolution to geo coordinates
- **reverse geocoding**: resolving geo coordinates to the closest address
- **tile server**: background map tiles

MOTIS uses efficient traffic day bitsets that allows efficient loading of **full year timetables**!
Loading one year of timetable doesn't take much more RAM than loading one month.

Features can be turned on and off as needed.

# Quick Start

- Create a folder with the following files.
- Download MOTIS from
  the [latest release](https://github.com/motis-project/motis/releases) and
  extract the archive.
- Download a OpenStreetMap dataset as `osm.pbf` (e.g.
  from [Geofabrik](https://download.geofabrik.de/)) and place it in the folder
- Download one or more GTFS datasets and place them in the folder 

```bash
./motis config my.osm.pbf gtfs.zip  # generates a minimal config.yml
./motis import                      # preprocesses data
./motis server                      # starts a HTTP server on port 8080 
```

This will preprocess the input files and create a `data` folder.
After that, it will start a server.

> [!IMPORTANT]
> Ensure a valid timetable is used. If the timetable is outdated, it will not contain any trips to consider for upcoming dates.

This script will execute the steps described above for a small dataset for the city of Aachen, Germany:

**Linux / macOS**

```bash
# set TARGET to linux-arm64, macos-arm64, ... to fit your setup
# see release list for supported platforms
TARGET="linux-amd64"
wget https://github.com/motis-project/motis/releases/latest/download/motis-${TARGET}.tar.bz2
tar xf motis-${TARGET}.tar.bz2
wget https://github.com/motis-project/test-data/raw/aachen/aachen.osm.pbf
wget https://opendata.avv.de/current_GTFS/AVV_GTFS_Masten_mit_SPNV.zip
./motis config aachen.osm.pbf AVV_GTFS_Masten_mit_SPNV.zip
./motis import
./motis server
```

**Windows**

```pwsh
Invoke-WebRequest https://github.com/motis-project/motis/releases/latest/download/motis-windows.zip -OutFile motis-windows.zip
Expand-Archive motis-windows.zip
Invoke-WebRequest https://github.com/motis-project/test-data/archive/refs/heads/aachen.zip -OutFile aachen.zip
Expand-Archive aachen.zip
./motis config aachen.osm.pbf AVV_GTFS_Masten_mit_SPNV.zip
./motis import
./motis server
```

# Documentation

## Developer Setup

Build MOTIS from source:
- [for Linux](docs/linux-dev-setup.md)
- [for Windows](docs/windows-dev-setup.md)
- [for macOS](docs/macos-dev-setup.md)

Set up a server using your build:
- [for Linux](docs/dev-setup-server.md)

MOTIS uses [pkg](https://github.com/motis-project/pkg) for dependency management.
See its [README](https://github.com/motis-project/pkg/blob/master/README.md) for how to work with it.

## Configuration

- [Advanced Setups](docs/setup.md)
