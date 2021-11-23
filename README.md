<p align="center"><img src="logo.svg" width="196" height="196"></p>

![Linux Build](https://github.com/motis-project/motis/workflows/Linux%20Build/badge.svg)
![Windows Build](https://github.com/motis-project/motis/workflows/Windows%20Build/badge.svg)

MOTIS stands for **M**ulti **O**bjective **T**ravel **I**nformation **S**ystem.

The core features are:

  - **Intermodal Routing**: computing optimal journeys mixing public transit, sharing mobility, walking, etc. in sensible ways. [Read more.](https://motis-project.de/docs/features/routing.html)
  - **Real Time Support**: considering delays, train cancellations, additional services, reroutings, track changes, etc. [Read more.](https://motis-project.de/docs/features/realtime.html#real-time-support)
  - **Visualization**: view vehicle movements in real-time. [Try it out!](https://europe.motis-project.de/)
  - **JSON API**: the backend provides a JSON API via HTTP. [Documentation](https://motis-project.de/docs/api/)
  
More detailed information can be found at [motis-project.de](https://motis-project.de).

To demonstrate the functionalities, an [Android App](https://play.google.com/store/apps/details?id=de.motis_project.demo) and a [web-based information system](https://demo.motis-project.de/) is available. The source code for both front-ends is available as Open Source Software [as well](https://github.com/motis-project/motis/tree/master/scripts).

The system can consume schedule timetables in the [GTFS](https://developers.google.com/transit/gtfs/) or [HAFAS](https://www.fahrplanfelder.ch/fileadmin/fap_daten_test/hrdf.pdf) format as well as real time information in the [GTFS-RT](https://developers.google.com/transit/gtfs-realtime/reference) (and RISML, a propriatary format at Deutsche Bahn) as input data. For pedestrian routing (handled by [Per Pedes Routing](https://github.com/motis-project/ppr)) and car routing (handled by [OSRM](https://github.com/Project-OSRM/osrm-backend)) OpenStreetMap data is used.

# Documentation

  - [Installation and Setup](https://motis-project.de/docs/install)
  - [API Documentation](https://motis-project.de/docs/api/)
  - Developer Setup:
    - [Windows Developer Setup](https://github.com/motis-project/motis/wiki/Windows-Developer-Setup)
    - [Linux Developer Setup](https://github.com/motis-project/motis/wiki/Linux-Developer-Setup)

# Contribution

Feel free to contribute in any area you like (new features, small improvments, bug fixes, documentation, testing, etc.)!
By making a pull-request you agree to license your contribution under the MIT and Apache 2.0 license as described below.

# Alternatives

  - [OpenTripPlanner](https://www.opentripplanner.org/)
  - [navitia.io](https://github.com/CanalTP/navitia)

# License

MOTIS can be licensed under the terms of the MIT license or under the terms of the Apache License, Version 2.0.
