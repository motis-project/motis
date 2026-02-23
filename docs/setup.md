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
          protocol: gtfsrt
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
  adjust_footpaths: true            # if footpaths are too fast, they are adjusted if set to true
  merge_dupes_intra_src: false      # duplicates within the same datasets will be merged
  merge_dupes_inter_src: false      # duplicates withing different datasets will be merged
  link_stop_distance: 100           # stops will be linked by footpaths if they're less than X meters (default=100m) apart
  update_interval: 60               # real-time updates are polled every `update_interval` seconds
  http_timeout: 30                  # maximum time in seconds the real-time feed download may take
  incremental_rt_update: false      # false = real-time updates are applied to a clean slate, true = no data will be dropped
  max_footpath_length: 15           # maximum footpath length when transitively connecting stops or for routing footpaths if `osr_footpath` is set to true
  max_matching_distance: 25.0       # maximum distance from geolocation to next OSM ways that will be found
  preprocess_max_matching_distance: 250.0 # max. distance for preprocessing matches from nigiri locations (stops) to OSM ways to speed up querying (set to 0 (default) to disable)
  datasets:                         # map of tag -> dataset
    ch:                             # the tag will be used as prefix for stop IDs and trip IDs with `_` as divider, so `_` cannot be part of the dataset tag
      path: ch_opentransportdataswiss.gtfs.zip
      default_bikes_allowed: false
      rt:
        - url: https://api.opentransportdata.swiss/gtfsrt2020
          headers:
            Authorization: MY_API_KEY
          protocol: gtfsrt          # specify the real time protocol (default: gtfsrt)
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
  street_routing_max_prepost_transit_seconds: 3600 # limit for maxPre/PostTransitTime API params, see below
  street_routing_max_direct_seconds: 21600 # limit for maxDirectTime API param, high values can lead to long-running, RAM-hungry queries 
logging:
  log_level: debug                # log-level (default = debug; Supported log-levels: error, info, debug)
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

# GBFS Configuration

This examples shows how to configure multiple GBFS feeds.  
A GBFS feed might describe a single system or area, `callabike` in this example, or a set of feeds, that are combined to a manifest, like `mobidata-bw` here. For readability, optional headers are not included.

```yaml
gbfs:
  feeds:
    # GBFS feed:
    callabike:
      url: https://api.mobidata-bw.de/sharing/gbfs/callabike/gbfs
    # GBFS manifest / Lamassu feed:
    mobidata-bw:
      url: https://api.mobidata-bw.de/sharing/gbfs/v3/manifest.json
  update_interval: 300
  http_timeout: 10
```

## Provider Groups + Colors

GBFS providers (feeds) can be grouped into "provider groups". For example, a provider may operate in multiple locations and provide a feed per location.
To groups these different feeds into a single provider group, specify the same group name for each feed in the configuration.

Feeds that don't have an explicit group setting in the configuration, their group name is derived from the system name. Group names
may not contain commas. The API supports both provider groups and individual providers.

Provider colors are loaded from the feed (`brand_assets.color`) if available, but can also be set in the configuration
to override the values contained in the feed or to set colors for feeds that don't include color information.
Colors can be set for groups (applies to all providers belonging to the group) or individual providers
(overrides group color for that feed).

```yaml
gbfs:
  feeds:
    de-CallaBike:
      url: https://api.mobidata-bw.de/sharing/gbfs/v2/callabike/gbfs
      color: "#db0016"
    de-VRNnextbike:
      url: https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_vn/gbfs.json
      group: nextbike # uses the group color defined below
    de-NextbikeFrankfurt:
      url: https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_ff/gbfs.json
      group: nextbike
    de-KVV.nextbike:
      url: https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_fg/gbfs.json
      group: nextbike
      color: "#c30937" # override color for this particular feed
  groups:
    nextbike:
      # name: nextbike # Optional: Override the name (otherwise the group id, here "nextbike", is used)
      color: "#0046d6"
```

For aggregated feeds (manifest.json or Lamassu), groups and colors can either be assigned to all providers listed in the aggregated feed
or individually by using the system_id:

```yaml
gbfs:
  feeds:
    aggregated-single-group:
      url: https://example.com/one-provider-group/manifest.json
      group: Example
      color: "#db0016" # or assign a color to the group
    aggregated-multiple-groups:
      url: https://example.com/multiple-provider-groups/manifest.json
      group:
        source-nextbike-westbike: nextbike # "source-nextbike-westbike" is the system_id
        source-voi-muenster: VOI
        source-voi-duisburg-oberhausen: VOI
      # colors can be specified for individual feeds using the same syntax,
      # but in this example they are defined for the groups below
      #color:
      #  "source-nextbike-westbike": "#0046d6"
      #  "source-voi-muenster": "#f26961"
  groups:
    nextbike:
      color: "#0046d6"
    VOI:
      color: "#f26961"
```

## HTTP Headers + OAuth

If a feed requires specific HTTP headers, they can be defined like this:

```yaml
gbfs:
  feeds:
    example:
      url: https://example.com/gbfs
      headers:
        authorization: MY_OTHER_API_KEY
        other-header: other-value
```

OAuth with client credentials and bearer token types is also supported:

```yaml
gbfs:
  feeds:
    example:
      url: https://example.com/gbfs
      oauth:
        token_url: https://example.com/openid-connect/token
        client_id: gbfs
        client_secret: example
```

## Default Restrictions

A GBFS feed can define geofencing zones and rules, that apply to areas within these zones.
For restrictions on areas not included in these geofencing zones, a feed may contain global rules.
If these are missing, it's possible to define `default_restrictions`, that apply to either a single feed or a manifest.
The following example shows possible configurations:

```yaml
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

# Real time protocols

MOTIS supports multiple protocols for real time feeds. This section shows a list of the protocols, including some pitfalls:

| Protocol | `protocol` | Note |
| ---- | ---- | ---- |
| GTFS-RT | `gtfsrt` | This is the default, if `protocol` is ommitted. |
| SIRI Lite (XML) | `siri` | Currently limited to SIRI Lite ET and SX. Still work in progress. Use with care. |
| SIRI Lite (JSON) | `siri_json` | Same as `siri`, but expects JSON server responses. See below for expected JSON structure. |
| VDV AUS / VDV454 | `auser` | Requires [`auser`](https://github.com/motis-project/auser) for subscription handling |

## Supported SIRI Lite services

SIRI feeds are divided into multiple feeds called services (check for instance
[this](https://en.wikipedia.org/wiki/Service_Interface_for_Real_Time_Information#CEN_SIRI_Functional_Services)
for a list of all services). Right now MOTIS only supports parsing the
"Estimated Timetable" (or ET) and the "Situation Exchange" (or SX) SIRI
services. You can see examples of such feeds
[here](https://github.com/SIRI-CEN/SIRI/blob/2.2/examples/siri_exm_ET/ext_estimatedTimetable_response.xml)
and
[here](https://github.com/SIRI-CEN/SIRI/blob/2.2/examples/siri_exm_SX/exx_situationExchange_response.xml).

If you are using the `siri_json` protocol, note that MOTIS expects the
following JSON structure:

- **Valid** SIRI Lite JSON response:

  ```json
  {
    "ResponseTimestamp": "2004-12-17T09:30:46-05:00",
    "ProducerRef": "KUBRICK",
    "Status": true,
    "MoreData": false,
    "EstimatedTimetableDelivery": [
      ...
    ]
  }
  ```

- **Invalid** SIRI Lite JSON response:

  ```json
  {
    "Siri": {
      "ServiceDelivery": {
        "ResponseTimestamp": "2004-12-17T09:30:46-05:00",
        "ProducerRef": "KUBRICK",
        "Status": true,
        "MoreData": false,
        "EstimatedTimetableDelivery": [
          ...
        ]
      }
    }
  }
  ```

If, as above, the two top keys `"Siri"` and `"ServiceDelivery"` are included in
the JSON response, MOTIS will fail to parse the SIRI Lite feed, throwing
`[VERIFY FAIL] unable to parse time ""` errors.

## Shapes

To enable shapes support (polylines for trips), `timetable.with_shapes` must
be set to `true`. This will load shapes that are present in the datasets
(e.g. GTFS shapes.txt).

It is also possible to compute shapes based on OpenStreetMap data. This
requires:

- `timetable.with_shapes` set to `true`
- `osm` data
- `street_routing` set to `true`
- `timetable.route_shapes` config:

```yaml
timetable:
  # with_shapes must be set to true to enable shapes support, otherwise no shapes will be loaded or computed
  with_shapes: true
  route_shapes: # all these options are optional
    # enable this to compute shapes for routes that don't have shapes in the dataset (default = false)
    missing_shapes: true
    # if replace_shapes is enabled, all shapes will be recomputed based on OSM data, even if shapes are already present in the dataset (default = false)
    replace_shapes: true
    # routing for specific clasz types can be disabled (default = all enabled)
    # currently long distance street routing is slow, so in this example
    # we disable routing shapes for COACH
    clasz:
      COACH: false
    # disable shape computation for routes with more than X stops (default = no limit)
    max_stops: 100
    # limit the number of threads used for shape computation (default = number of hardware threads)
    n_threads: 6
    # cache and reuse computed shapes for later imports (dataset updates)
    cache: true

    # for debugging purposes, debug information can be written to files
    # which can be loaded into the debug ui (see osr project)
    debug:
      path: /path/to/debug/directory
      all: false                  # debug all routes
      all_with_beelines: false    # or only those that include beelines
      slow: 10000                 # or only those that take >10.000ms to compute
      # or specific trips/routes:
      trips:
        - "trip_id_1"
      route_ids:
        - "route_id_1"
      route_indices: # these are internal indices (e.g. from debug UI)
        - 123
```

### Cache

Routed shapes can be cached to speed up later imports when a timetable dataset
is updated. If enabled, this will generate an additional cache file. This cache
file and the routed shapes data are then reused during import.

Note that old routes are never removed from the routed shapes data files, i.e.,
these files grow with every import (unless there are no new routes, in which
case the size will stay the same).
It is therefore recommended to monitor the size of the "routed_shapes_*" files
in the data directory.
They can safely be deleted before an import, which will cause all shapes that
are needed for the current datasets to be routed again.

The cache only applies to routed shapes, not shapes contained in the timetables.
