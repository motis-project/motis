module Data.Isochrone.Types exposing
    ( IsochroneRequest
    , IsochroneResponse
    )

import Date exposing (Date)
import Data.Connection.Types exposing (Station, Position)


type alias IsochroneRequest =
    { from : Position
    , intervalStart : Int
    , duration : Int
    , foot_time : Int
    }


type alias IsochroneResponse =
    { station : List Station
    , arrival_times: List Int
    }





