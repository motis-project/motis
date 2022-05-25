module Data.Connection.Request exposing (encodePosition, encodeStation)

import Data.Connection.Types exposing (..)
import Json.Encode as Encode



-- for local storage


encodeStation : Station -> Encode.Value
encodeStation station =
    Encode.object
        [ ("id" , Encode.string station.id)
        , ("name" , Encode.string station.name)
        , ("pos" , encodePosition station.pos)
        ]


encodePosition : Position -> Encode.Value
encodePosition pos =
    Encode.object
        [ ("lat" , Encode.float pos.lat)
        , ("lng" , Encode.float pos.lng)
        ]
