# Schnellstart

1. Den [aktuellen Release](https://github.com/motis-project/motis/releases) herunterladen

1. OpenStreetMap-Daten im `.osm.pbf`-Format herunterladen

1. Eine oder mehrere _timetable_s herunterladen

1. Die Konfiguration erstellen:
```sh
motis config <osm-data> <timetable>...
```

1. Die Daten importieren:
```sh
motis import
```

1. Den Server starten:
```sh
motis server
```

1. Der Server ist verfügbar unter http://localhost:8080

Weitere Informationen und Optionen sind dokumentiert unter:
* [dev-setup-server](dev-setup-server.md)
* [elevation-setup](elevation-setup.md)


