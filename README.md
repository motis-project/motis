<p align="center"><img src="logo.svg" width="196" height="196"></p>

![Linux Build](https://github.com/motis-project/motis/workflows/Linux%20Build/badge.svg)
![Windows Build](https://github.com/motis-project/motis/workflows/Windows%20Build/badge.svg)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/motis-project/motis)

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


# License

MOTIS can be licensed under the terms of the MIT license or under the terms of the Apache License, Version 2.0.
