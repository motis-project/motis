# User Scripts

MOTIS can post-process GTFS static timetable data using [Lua](https://www.lua.org/) scripts. The main purpose is to fix data in case the MOTIS user is not the owner of the data nd the data owner cannot or does not want to fix the data. In some cases, the scripts can be used to homogenize data across different datasets. Currently, post-processing is available for the following entities:

If no script is defined or a processing function is not given for a type, the default behaviour will be applied.


## Configuration

Scripts are an optional key for each dataset in the timetable. An empty string or not setting the property indicates no processing. Any non-empty string will be interpreted as file path to a `.lua` file. The file has to exist.

Example configuration with script property set:

```
timetable:
  datasets:
    nl:
      path: nl_ovapi.gtfs.zip
      rt:
        - url: https://gtfs.ovapi.nl/nl/trainUpdates.pb
        - url: https://gtfs.ovapi.nl/nl/tripUpdates.pb
      script: my-script.lua
```

## Types

### Location (stops, platforms, tracks)

processing via `function process_location()`

  - `get_id`
  - `get_name`
  - `set_name`
  - `get_platform_code`
  - `set_platform_code`
  - `get_description`
  - `set_description`
  - `get_pos`
  - `set_pos`
  - `get_timezone`
  - `set_timezone`
  - `get_transfer_time`
  - `set_transfer_time` 

### Agency (as defined in GTFS `agencies.txt`)

processing via `function process_agency(agency)`

  - `get_id`
  - `get_name`
  - `set_name`
  - `get_url`
  - `set_url`
  - `get_timezone`
  - `set_timezone`

### Routes (as defined in GTFS `routes.txt`)

processing via `function process_location(location)`

  - `get_id`
  - `get_short_name`
  - `set_short_name`
  - `get_long_name`
  - `set_long_name`
  - `get_route_type`
  - `set_route_type`
  - `get_color`
  - `set_color`
  - `get_clasz`
  - `set_clasz`
  - `get_text_color`
  - `set_text_color`
  - `get_agency`

The `clasz` attribute is an internal grouping of transport modes in MOTIS that can be overwritten if the automatically derived `clasz` is not correct. The following values exist:

  - Air = 0
  - HighSpeed = 1
  - LongDistance = 2
  - Coach = 3
  - Night = 4
  - RegionalFast = 5
  - Regional = 6
  - Metro = 7
  - Subway = 8
  - Tram = 9
  - Bus = 10
  - Ship = 11
  - CableCar = 12
  - Funicular = 13
  - AreaLift = 14
  - Other = 15

The color is currently set as unsigned 32bit integer. In future versions, we might change this to a hex string like `#FF0000`.

### Trips (as defined in GTFS `trips.txt`)
  
processing via `function process_trip(trip)`

  - `get_id`
  - `get_headsign`
  - `set_headsign`
  - `get_short_name`
  - `set_short_name`
  - `get_display_name`
  - `set_display_name`
  - `get_route`

### Geo Location

This type is used for stop coordinates in `process_location()` for `location:get_pos()` and `location:set_pos`.

  - `get_lat`
  - `get_lng`
  - `set_lat`
  - `set_lng`


## Filtering

Each processing function can return a boolean which will be interpreted as

  - `true`: keep this entity
  - `false`: don't keep this entity

If nothing is returned from a process function (e.g. no return statement at all), no filtering will be applied (i.e. the default is `keep=true`).

Filtering has the following effects:

  - In case an agency is removed, all its routes and trips will be removed as well
  - In case a route is removed, all its trips will be removed as well
  - If locations are filtered, the locations will not be removed from trips and transfers referencing those stops


## Out of Scope

Scripting is currently aiming at cosmetic changes to existing entities to improve the user experience, not the creation of new entities. The creation of new entities currently has to be done outside of MOTIS in a separate preprocessing step. Currently, it is also not supported to mess with primary/foreign keys (IDs such as `trip_id`, `stop_id`, `route_ìd`). 


## Example

This example illustrates the usage of scripting capabilities in MOTIS. Beware that it does not make sense at all and its sole purpose is to demonstrate syntax and usage of available functionality.

```lua
function process_location(stop)
  local name = stop:get_name()
  if string.sub(name, -7) == ' Berlin' then
    stop:set_name(string.sub(name, 1, -8))
  end

  local pos = stop:get_pos()
  pos:set_lat(stop:get_pos():get_lat() + 2.0)
  pos:set_lng(stop:get_pos():get_lng() - 2.0)
  stop:set_pos(pos)

  stop:set_description(stop:get_description() .. ' ' .. stop:get_id() .. ' YEAH')
  stop:set_timezone('Europe/Berlin')
  stop:set_transfer_time(stop:get_transfer_time() + 98)
  stop:set_platform_code(stop:get_platform_code() .. 'A')

  return true
end

function process_route(route)
  if route:get_id() == 'R_RE4' then
    return false
  end

  if route:get_route_type() == 3 then
    route:set_clasz(7)
    route:set_route_type(101)
  elseif route:get_route_type() == 1 then
    route:set_clasz(8)
    route:set_route_type(400)
  end

  if route:get_agency():get_name() == 'Deutsche Bahn' and route:get_route_type() == 101 then
    route:set_short_name('RE ' .. route:get_short_name())
  end

  return true
end

function process_agency(agency)
  if agency:get_id() == 'TT' then
    return false
  end

  if agency:get_name() == 'Deutsche Bahn' and agency:get_id() == 'DB' then
    agency:set_url(agency:get_timezone())
    agency:set_timezone('Europe/Berlin')
    agency:set_name('SNCF')
    return true
  end
  return false
end

function process_trip(trip)
  if trip:get_route():get_route_type() == 101 then
    -- Prepend category and eliminate leading zeros (e.g. '00123' -> 'ICE 123')
    trip:set_short_name('ICE ' .. string.format("%d", tonumber(trip:get_short_name())))
    trip:set_display_name(trip:get_short_name())
  end
  return trip:get_id() == 'T_RE1'
end
```


## Future Work

There are more attributes that could be made readable/writable such as `bikes_allowed`, `cars_allowed`. Also trip stop times and their attributes such as stop sequence numbers could be made available to scripting.

Another topic not addressed yet is API versioning for the lua functions. At the moment, this feature is considered experimental which means that breaking changes might occur without prior notice.