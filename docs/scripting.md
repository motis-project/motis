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

### Translation List

Some string fields are translated. Their default getter (e.g. `get_name`) now
returns the default string, while the accompanying `get_*_translations`
functions expose the full translation list. Lists can be accessed with
[sol2 container operations](https://sol2.readthedocs.io/en/latest/containers.html).
Each entry in that list is of type
`translation` and provides:

  - `get_language`
  - `set_language`
  - `get_text`
  - `set_text`

Example snippet of how to read and write translations:

```lua
function process_route(route)
  route:set_short_name({
    translation.new('en', 'EN_SHORT_NAME'),
    translation.new('de', 'DE_SHORT_NAME'),
    translation.new('fr', 'FR_SHORT_NAME')
  })
  route:get_short_name_translations():add(translation.new('hu', 'HU_SHORT_NAME'))
  print(route:get_short_name_translations():get(1):get_text())
  print(route:get_short_name_translations():get(1):get_language())
end
```

### Location (stops, platforms, tracks)

processing via `function process_location()`

  - `get_id`
  - `get_name`
  - `get_name_translations`
  - `set_name`
  - `get_platform_code`
  - `get_platform_code_translations`
  - `set_platform_code`
  - `get_description`
  - `get_description_translations`
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
  - `get_name_translations`
  - `set_name`
  - `get_url`
  - `get_url_translations`
  - `set_url`
  - `get_timezone`
  - `set_timezone`

### Routes (as defined in GTFS `routes.txt`)

processing via `function process_route(location)`

  - `get_id`
  - `get_short_name`
  - `get_short_name_translations`
  - `set_short_name`
  - `get_long_name`
  - `get_long_name_translations`
  - `set_long_name`
  - `get_route_type`
  - `set_route_type`
  - `get_color`
  - `set_color`
  - `get_clasz`  (deprecated, use `get_route_type`)
  - `set_clasz`  (deprecated, use `set_route_type`)
  - `get_text_color`
  - `set_text_color`
  - `get_agency`

The following constants can be used for `set_route_type`:

- `GTFS_TRAM`
- `GTFS_SUBWAY`
- `GTFS_RAIL`
- `GTFS_BUS`
- `GTFS_FERRY`
- `GTFS_CABLE_TRAM`
- `GTFS_AERIAL_LIFT`
- `GTFS_FUNICULAR`
- `GTFS_TROLLEYBUS`
- `GTFS_MONORAIL`
- `RAILWAY_SERVICE`
- `HIGH_SPEED_RAIL_SERVICE`
- `LONG_DISTANCE_TRAINS_SERVICE`
- `INTER_REGIONAL_RAIL_SERVICE`
- `CAR_TRANSPORT_RAIL_SERVICE`
- `SLEEPER_RAIL_SERVICE`
- `REGIONAL_RAIL_SERVICE`
- `TOURIST_RAILWAY_SERVICE`
- `RAIL_SHUTTLE_WITHIN_COMPLEX_SERVICE`
- `SUBURBAN_RAILWAY_SERVICE`
- `REPLACEMENT_RAIL_SERVICE`
- `SPECIAL_RAIL_SERVICE`
- `LORRY_TRANSPORT_RAIL_SERVICE`
- `ALL_RAILS_SERVICE`
- `CROSS_COUNTRY_RAIL_SERVICE`
- `VEHICLE_TRANSPORT_RAIL_SERVICE`
- `RACK_AND_PINION_RAILWAY_SERVICE`
- `ADDITIONAL_RAIL_SERVICE`
- `COACH_SERVICE`
- `INTERNATIONAL_COACH_SERVICE`
- `NATIONAL_COACH_SERVICE`
- `SHUTTLE_COACH_SERVICE`
- `REGIONAL_COACH_SERVICE`
- `SPECIAL_COACH_SERVICE`
- `SIGHTSEEING_COACH_SERVICE`
- `TOURIST_COACH_SERVICE`
- `COMMUTER_COACH_SERVICE`
- `ALL_COACHS_SERVICE`
- `URBAN_RAILWAY_SERVICE`
- `METRO_SERVICE`
- `UNDERGROUND_SERVICE`
- `URBAN_RAILWAY_2_SERVICE`
- `ALL_URBAN_RAILWAYS_SERVICE`
- `MONORAIL_SERVICE`
- `BUS_SERVICE`
- `REGIONAL_BUS_SERVICE`
- `EXPRESS_BUS_SERVICE`
- `STOPPING_BUS_SERVICE`
- `LOCAL_BUS_SERVICE`
- `NIGHT_BUS_SERVICE`
- `POST_BUS_SERVICE`
- `SPECIAL_NEEDS_BUS_SERVICE`
- `MOBILITY_BUS_SERVICE`
- `MOBILITY_BUS_FOR_REGISTERED_DISABLED_SERVICE`
- `SIGHTSEEING_BUS_SERVICE`
- `SHUTTLE_BUS_SERVICE`
- `SCHOOL_BUS_SERVICE`
- `SCHOOL_AND_PUBLIC_BUS_SERVICE`
- `RAIL_REPLACEMENT_BUS_SERVICE`
- `DEMAND_AND_RESPONSE_BUS_SERVICE`
- `ALL_BUSS_SERVICE`
- `TROLLEYBUS_SERVICE`
- `TRAM_SERVICE`
- `CITY_TRAM_SERVICE`
- `LOCAL_TRAM_SERVICE`
- `REGIONAL_TRAM_SERVICE`
- `SIGHTSEEING_TRAM_SERVICE`
- `SHUTTLE_TRAM_SERVICE`
- `ALL_TRAMS_SERVICE`
- `WATER_TRANSPORT_SERVICE`
- `AIR_SERVICE`
- `FERRY_SERVICE`
- `AERIAL_LIFT_SERVICE`
- `TELECABIN_SERVICE`
- `CABLE_CAR_SERVICE`
- `ELEVATOR_SERVICE`
- `CHAIR_LIFT_SERVICE`
- `DRAG_LIFT_SERVICE`
- `SMALL_TELECABIN_SERVICE`
- `ALL_TELECABINS_SERVICE`
- `FUNICULAR_SERVICE`
- `TAXI_SERVICE`
- `COMMUNAL_TAXI_SERVICE`
- `WATER_TAXI_SERVICE`
- `RAIL_TAXI_SERVICE`
- `BIKE_TAXI_SERVICE`
- `LICENSED_TAXI_SERVICE`
- `PRIVATE_HIRE_VEHICLE_SERVICE`
- `ALL_TAXIS_SERVICE`
- `MISCELLANEOUS_SERVICE`
- `HORSE_DRAWN_CARRIAGE_SERVICE`

The color is currently set as unsigned 32bit integer. In future versions, we might change this to a hex string like `#FF0000`.

### Trips (as defined in GTFS `trips.txt`)
  
processing via `function process_trip(trip)`

  - `get_id`
  - `get_headsign`
  - `get_headsign_translations`
  - `set_headsign`
  - `get_short_name`
  - `get_short_name_translations`
  - `set_short_name`
  - `get_display_name`
  - `get_display_name_translations`
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
