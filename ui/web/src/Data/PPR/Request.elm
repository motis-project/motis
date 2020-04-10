module Data.PPR.Request exposing
    ( encodeFootRoutingRequest
    , encodeSearchOptions
    )

import Data.Connection.Request exposing (encodePosition)
import Data.PPR.Types exposing (..)
import Json.Encode as Encode
import Util.Core exposing ((=>))


encodeFootRoutingRequest : FootRoutingRequest -> Encode.Value
encodeFootRoutingRequest req =
    Encode.object
        [ "destination"
            => Encode.object
                [ "type" => Encode.string "Module"
                , "target" => Encode.string "/ppr/route"
                ]
        , "content_type" => Encode.string "FootRoutingRequest"
        , "content"
            => Encode.object
                [ "start" => encodePosition req.start
                , "destinations" => Encode.list (List.map encodePosition req.destinations)
                , "search_options" => encodeSearchOptions req.search_options
                , "include_steps" => Encode.bool req.include_steps
                , "include_edges" => Encode.bool req.include_edges
                , "include_path" => Encode.bool req.include_path
                ]
        ]


encodeSearchOptions : SearchOptions -> Encode.Value
encodeSearchOptions opt =
    Encode.object
        [ "profile" => Encode.string opt.profile
        , "duration_limit" => Encode.float opt.duration_limit
        ]
