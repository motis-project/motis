# Quickstart

1. Download the [latest release](https://github.com/motis-project/motis/releases)

1. Download a OpenStreetMap dataset, stored in `.osm.pbf` format

1. Download one or multiple timetable datasets

1. Create the configuration:
```sh
motis config <osm-data> <timetable>...
```

1. Import the data:
```sh
motis import
```

1. Start the server:
```sh
motis server
```

1. Access the server at http://localhost:8080

For more details and options see also:
* [dev-setup-server](dev-setup-server.md)
* [elevation-setup](elevation-setup.md)

