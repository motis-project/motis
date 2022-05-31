module Data.GBFSInfo.Request exposing (request)

import Json.Encode as Encode
import Util.Core exposing ((=>))


request : Encode.Value
request =
    Encode.object
        [ "destination"
            => Encode.object
                [ "type" => Encode.string "Module"
                , "target" => Encode.string "/gbfs/info"
                ]
        , "content_type" => Encode.string "MotisNoMessage"
        , "content"
            => Encode.object
                []
        ]
