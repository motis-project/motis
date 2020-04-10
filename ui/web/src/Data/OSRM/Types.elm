module Data.OSRM.Types exposing (OSRMViaRouteRequest, OSRMViaRouteResponse)

import Data.Connection.Types exposing (Position)
import Data.RailViz.Types exposing (Polyline)


type alias OSRMViaRouteRequest =
    { profile : String
    , waypoints : List Position
    }


type alias OSRMViaRouteResponse =
    { time : Int
    , distance : Float
    , polyline : Polyline
    }
