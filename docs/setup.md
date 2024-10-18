# Advanced Configuration

This is an example of how to use multiple GTFS-static datasets with multiple real-time feeds. You can also see how to set additional headers like `Authorization` to enable the usage of API keys.

```yaml
server:
  port: 8080
  web_folder: ui
osm: netherlands-latest.osm.pbf
timetable:
  datasets:
    nl:
      path: nl_ovapi.gtfs.zip
      rt:
        - url: https://gtfs.ovapi.nl/nl/trainUpdates.pb
        - url: https://gtfs.ovapi.nl/nl/tripUpdates.pb
    ch:
      path: ch_opentransportdataswiss.gtfs.zip
      rt:
        - url: https://api.opentransportdata.swiss/gtfsrt2020
          headers:
            Authorization: MY_API_KEY
tiles:
  profile: tiles-profiles/full.lua
street_routing: true
geocoding: true
osr_footpath: true
```

This expands to the following configuration:

```yaml
server:
  host: 0.0.0.0                     # host (default = 0.0.0.0)
  port: 8080                        # port (default = 8080)
  web_folder: ui                    # folder with static files to serve
  n_threads: 24                     # default (if not set): number of hardware threads 
osm: netherlands-latest.osm.pbf     # required by tiles, street routing, geocoding and reverse-geocoding
tiles:                              # tiles won't be available if this key is missing
  profile: tiles-profiles/full.lua  # currently `background.lua` (less details) and `full.lua` (more details) are available
  db_size: 1099511627776            # default size for the tiles database (influences VIRT memory usage)
  flush_threshold: 10000000         # usually don't change this (less = reduced memory usage during tiles import)
timetable:                          # if not set, no timetable will be loaded
  first_day: TODAY                  # first day of timetable to load, format: "YYYY-MM-DD" (special value "TODAY")
  num_days: 365                     # number of days to load, default is 365 days
  railviz: true                     # enable viewing vehicles in real-time on the map, requires some extra lookup data structures
  with_shapes: true                 # extract and serve shapes (if disabled, direct lines are used)
  ignore_errors: false              # ignore errors when a dataset could not be loaded
  adjust_footpaths: true            # if footpaths are too fast, they are adjusted if set to true
  merge_dupes_intra_src: false      # duplicates within the same datasets will be merged
  merge_dupes_inter_src: false      # duplicates withing different datasets will be merged
  link_stop_distance: 100           # stops will be linked by footpaths if they're less than X meters (default=100m) apart
  update_interval: 60               # real-time updates are polled every `update_interval` seconds
  http_timeout: 10                  # timeout for the HTTP server to respond with a package
  incremental_rt_update: false      # false = real-time updates are applied to a clean slate, true = no data will be dropped
  max_footpath_length: 15           # maximum footpath length when transitively connecting stops or for routing footpaths if `osr_footpath` is set to true
  datasets:                         # map of tag -> dataset
    ch:                             # the tag will be used as prefix for stop IDs and trip IDs with `_` as divider, so `_` cannot be part of the dataset tag
      path: ch_opentransportdataswiss.gtfs.zip
      default_bikes_allowed: false
      rt:
        - url: https://api.opentransportdata.swiss/gtfsrt2020
          headers:
            Authorization: MY_API_KEY
    nl:
      path: nl_ovapi.gtfs.zip
      default_bikes_allowed: false
      rt:
        - url: https://gtfs.ovapi.nl/nl/trainUpdates.pb
        - url: https://gtfs.ovapi.nl/nl/tripUpdates.pb
street_routing: true              # enable street routing (default = false)
osr_footpath: true                # enable routing footpaths instead of using transfers from timetable datasets
elevators: false                  # currently not supported
geocoding: true                   # enable geocoding for place/stop name autocompletion
reverse_geocoding: false          # enable reverse geocoding for mapping a geo coordinate to nearby places/addresses
```