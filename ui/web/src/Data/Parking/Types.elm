module Data.Parking.Types exposing
    ( Parking
    , ParkingEdgeDirection(..)
    , ParkingEdgeRequest
    , ParkingEdgeResponse
    )

import Data.Connection.Types exposing (Connection, Position)
import Data.OSRM.Types exposing (OSRMViaRouteResponse)
import Data.PPR.Types exposing (Route, SearchProfile)


type ParkingEdgeDirection
    = Outward
    | Return


type alias Parking =
    { id : Int
    , pos : Position
    , fee : Bool
    }


type alias ParkingEdgeRequest =
    { id : Int
    , start : Position
    , destination : Position
    , direction : ParkingEdgeDirection
    , ppr_search_profile : SearchProfile
    , duration : Int
    , accessibility : Int
    , include_steps : Bool
    , include_edges : Bool
    , include_path : Bool
    }


type alias ParkingEdgeResponse =
    { parking : Parking
    , car : OSRMViaRouteResponse
    , walk : Route
    , uses_car : Bool
    }
