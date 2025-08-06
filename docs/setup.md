# Advanced Configuration

This is an example of how to use multiple GTFS-static datasets with multiple real-time feeds, as well as GBFS feeds. You can also see how to set additional headers like `Authorization` to enable the usage of API keys.

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
gbfs:
  feeds:
    montreal:
      url: https://gbfs.velobixi.com/gbfs/gbfs.json
    # Example feed for header usage
    example-feed:
      url: https://example.org/gbfs
      headers:
        authorization: MY_OTHER_API_KEY
        other-header: other-value
tiles:
  profile: tiles-profiles/full.lua
street_routing:
  elevation_data_dir: srtm/
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
  data_attribution_link: https://creativecommons.org/licenses/by/4.0/ # link to data sources or license exposed in HTTP headers and UI
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
  http_timeout: 30                  # maximum time in seconds the real-time feed download may take
  incremental_rt_update: false      # false = real-time updates are applied to a clean slate, true = no data will be dropped
  max_footpath_length: 15           # maximum footpath length when transitively connecting stops or for routing footpaths if `osr_footpath` is set to true
  max_matching_distance: 25.0       # maximum distance from geolocation to next OSM ways that will be found
  preprocess_max_matching_distance: 0.0 # max. distance for preprocessing matches from nigiri locations (stops) to OSM ways to speed up querying (set to 0 (default) to disable)
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
      extend_calendar: true         # expand the weekly service pattern beyond the end of `feed_info.txt::feed_end_date` if `feed_end_date` matches `calendar.txt::end_date`
gbfs:
  feeds:
    montreal:
      url: https://gbfs.velobixi.com/gbfs/gbfs.json
    example-feed:
      url: https://example.org/gbfs
      headers:
        authorization: MY_OTHER_API_KEY
        other-header: other-value
street_routing:                   # enable street routing (default = false; Using boolean values true/false is supported for backward compatibility)
  elevation_data_dir: srtm/       # folder which contains elevation data, e.g. SRTMGL1 data tiles in HGT format
limits:
  stoptimes_max_results: 256      # maximum number of stoptimes results that can be requested
  plan_max_results: 256           # maximum number of plan results that can be requested via numItineraries parameter
  plan_max_search_window_minutes: 5760 # maximum (minutes) for searchWindow parameter (seconds), highest possible value: 21600 (15 days)
  onetoall_max_results: 65535     # maximum number of one-to-all results that can be requested
  onetoall_max_travel_minutes: 90 # maximum travel duration for one-to-all query that can be requested
  routing_max_timeout_seconds: 90 # maximum duration a routing query may take
  gtfsrt_expose_max_trip_updates: 100 # how many trip updates are allowed to be exposed via the gtfsrt endpoint
osr_footpath: true                # enable routing footpaths instead of using transfers from timetable datasets
geocoding: true                   # enable geocoding for place/stop name autocompletion
reverse_geocoding: false          # enable reverse geocoding for mapping a geo coordinate to nearby places/addresses
```

# Scenario with Elevators

This is an example configuration for Germany which enables the real-time update of elevators from Deutsche Bahn's FaSta (Facility Status) JSON API. You need to register and obtain an API key.

```yml
server:
  web_folder: ui
tiles:
  profile: tiles-profiles/full.lua
geocoding: true
street_routing: true    # Alternative notion the enable street routing
osr_footpath: true
elevators:
  #  init: fasta.json   # Can be used for debugging, remove `url` key in this case
  url: https://apis.deutschebahn.com/db-api-marketplace/apis/fasta/v2/facilities
  headers:
    DB-Client-ID: b5d28136ffedb73474cc7c97536554df!
    DB-Api-Key: ef27b9ad8149cddb6b5e8ebb559ce245!
osm: germany-latest.osm.pbf
timetable:
  extend_missing_footpaths: true
  use_osm_stop_coordinates: true
  datasets:
    de:
      path: 20250331_fahrplaene_gesamtdeutschland_gtfs.zip
      rt:
        - url: https://stc.traines.eu/mirror/german-delfi-gtfs-rt/latest.gtfs-rt.pbf
```

# GBFS Configuration and Default Restrictions

This examples shows how to configure multiple GBFS feeds.  
A GBFS feed might describe a single system or area, `callabike` in this example, or a set of feeds, that are combined to a manifest, like `mobidata-bw` here. For readability, optional headers are not included.

A GBFS feed can define geofencing zones and rules, that apply to areas within these zones.
For restrictions on areas not included in these geofencing zones, a feed may contain global rules.
If these are missing, it's possible to define `default_restrictions`, that apply to either a single feed or a manifest.
The following example shows possible configurations:

```
gbfs:
  feeds:
    # GBFS feed:
    #callabike:
    #  url: https://api.mobidata-bw.de/sharing/gbfs/callabike/gbfs
    # GBFS manifest / Lamassu feed:
    mobidata-bw:
      url: https://api.mobidata-bw.de/sharing/gbfs/v3/manifest.json
  default_restrictions:
    mobidata-bw:callabike: # "callabike" feed contained in the "mobidata-bw" manifest
      # these restrictions apply outside of the defined geofencing zones if the feed doesn't contain global rules
      ride_start_allowed: true
      ride_end_allowed: true
      ride_through_allowed: true
      #station_parking: false
      #return_constraint: roundtrip_station
    #mobidata-bw: # default restrictions for all feeds contained in the "mobidata-bw" manifest
    #callabike: # default restrictions for standalone GBFS feed "callabike" (when not using the mobidata-bw example)
  update_interval: 300
  http_timeout: 10
```
