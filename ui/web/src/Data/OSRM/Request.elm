module Data.OSRM.Request exposing (encodeOSRMViaRouteRequest)

import Data.Connection.Request exposing (encodePosition)
import Data.OSRM.Types exposing (..)
import Json.Encode as Encode
import Util.Core exposing ((=>))


encodeOSRMViaRouteRequest : OSRMViaRouteRequest -> Encode.Value
encodeOSRMViaRouteRequest req =
    Encode.object
        [ "destination"
            => Encode.object
                [ "type" => Encode.string "Module"
                , "target" => Encode.string "/osrm/via"
                ]
        , "content_type" => Encode.string "OSRMViaRouteRequest"
        , "content"
            => Encode.object
                [ "profile" => Encode.string req.profile
                , "waypoints" => Encode.list (List.map encodePosition req.waypoints)
                ]
        ]
