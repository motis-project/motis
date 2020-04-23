port module Port exposing
    ( clearRailVizError
    , handleRailVizError
    , localStorageSet
    , setPPRSearchOptions
    , setRoutingResponses
    , setSimulationTime
    , setTimeOffset
    , showStationDetails
    , showTripDetails
    )

import Data.Connection.Types exposing (TripId)
import Json.Encode



-- see also: Widgets.Map.Port


port setRoutingResponses : (List ( String, String ) -> msg) -> Sub msg


port showStationDetails : (String -> msg) -> Sub msg


port showTripDetails : (TripId -> msg) -> Sub msg


port setTimeOffset : Float -> Cmd msg


port setSimulationTime : (Float -> msg) -> Sub msg


port handleRailVizError : (Json.Encode.Value -> msg) -> Sub msg


port clearRailVizError : (() -> msg) -> Sub msg


port localStorageSet : ( String, String ) -> Cmd msg


port setPPRSearchOptions : Json.Encode.Value -> Cmd msg
