module Localization.En exposing (enLocalization, enTranslations)

import Localization.Base exposing (..)
import Util.DateFormat exposing (..)


enLocalization : Localization
enLocalization =
    { t = enTranslations
    , dateConfig = enDateConfig
    }


enTranslations : Translations
enTranslations =
    { search =
        { search = "Search"
        , start = "Start"
        , destination = "Destination"
        , date = "Date"
        , time = "Time"
        , startTransports = "Transports at the start"
        , destinationTransports = "Transports at the destination"
        , departure = "Departure"
        , arrival = "Arrival"
        , trainNr = "Train Number"
        , maxDuration = "Max. duration (minutes)"
        , searchProfile =
            { profile = "Profil"
            , walkingSpeed = "Walking speed (m/s)"
            , stairsUp = "Stairs (up)"
            , stairsDown = "Stairs (down)"
            , stairsWithHandrailUp = "Stairs with handrail (up)"
            , stairsWithHandrailDown = "Stairs with handrail (down)"
            , timeCost = "Time cost"
            , accessibilityCost = "Accessibility cost"
            , streetCrossings = "Street crossings"
            , signals = "Traffic signals"
            , marked = "Marked (zebra crossings)"
            , island = "Traffic islands"
            , unmarked = "Unmarked"
            , primary = "Primary"
            , secondary = "Secondary"
            , tertiary = "Tertiary"
            , residential = "Residential"
            , elevationUp = "Elevation difference (up)"
            , elevationDown = "Elevation difference (down)"
            , roundAccessibility = "Round accessibility"
            , elevators = "Elevators"
            , escalators = "Escalators"
            , movingWalkways = "Moving walkways"
            }
        , useParking = "Use parkings"
        }
    , connections =
        { timeHeader = "Time"
        , durationHeader = "Duration"
        , transportsHeader = "Transports"
        , scheduleRange =
            \begin end ->
                "Possible dates: "
                    ++ formatDate enDateConfig begin
                    ++ " – "
                    ++ formatDate enDateConfig end
        , loading = "Searching..."
        , noResults = "No connections found"
        , extendBefore = "Earlier"
        , extendAfter = "Later"
        , interchanges =
            \count ->
                case count of
                    0 ->
                        "No interchanges"

                    1 ->
                        "1 interchange"

                    _ ->
                        toString count ++ " interchanges"
        , walkDuration = \duration -> duration ++ " walk"
        , interchangeDuration = \duration -> duration ++ " interchange"
        , arrivalTrack = \track -> "Arrival track " ++ track
        , track = "Track"
        , tripIntermediateStops =
            \count ->
                case count of
                    0 ->
                        "No intermediate stops"

                    1 ->
                        "1 intermediate stop"

                    _ ->
                        toString count ++ " intermediate stops"
        , tripWalk = \duration -> "Walk (" ++ duration ++ ")"
        , tripBike = \duration -> "Bike (" ++ duration ++ ")"
        , tripCar = \duration -> "Car (" ++ duration ++ ")"
        , provider = "Provider"
        , walk = "Walk"
        , bike = "Bike"
        , car = "Car"
        , trainNr = "Train number"
        , lineId = "Line"
        , parking = "Parking"
        }
    , station =
        { direction = "Direction"
        , noDepartures = "No departures"
        , noArrivals = "No arrivals"
        , loading = "Loading..."
        , trackAbbr = "Tr."
        }
    , railViz =
        { noTrains = "No trains"
        , delayColors = "By delay"
        , classColors = "By category"
        , simActive = "Simulation mode active"
        }
    , mapContextMenu =
        { routeFromHere = "Directions from here"
        , routeToHere = "Directions to here"
        }
    , errors =
        { journeyDateNotInSchedule = "Date not in schedule"
        , internalError = \msg -> "Internal error (" ++ msg ++ ")"
        , timeout = "Timeout"
        , network = "Network error"
        , http = \code -> "HTTP error " ++ toString code
        , decode = \msg -> "Invalid response (" ++ msg ++ ")"
        , moduleNotFound = "Module not found"
        , osrmProfileNotAvailable = "OSRM: Profile not available"
        , osrmNoRoutingResponse = "OSRM: No routing response"
        , pprProfileNotAvailable = "PPR: Profile not available"
        }
    , trips =
        { noResults = "No matching trains found"
        }
    , misc =
        { permalink = "Permalink" }
    , simTime =
        { simMode = "Simulation mode"
        }
    , searchProfiles =
        { default = "Default"
        , accessibility1 = "Include accessible routes"
        , wheelchair = "Wheelchair"
        , elevation = "Avoid elevation changes"
        , custom = "Custom"
        }
    }
