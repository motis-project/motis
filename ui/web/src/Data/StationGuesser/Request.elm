module Data.StationGuesser.Request exposing (encodeRequest)

import Json.Encode as Encode
import Util.Core exposing ((=>))


encodeRequest : Int -> String -> Encode.Value
encodeRequest guessCount input =
    Encode.object
        [ "destination"
            => Encode.object
                [ "type" => Encode.string "Module"
                , "target" => Encode.string "/guesser"
                ]
        , "content_type" => Encode.string "StationGuesserRequest"
        , "content"
            => Encode.object
                [ "input" => Encode.string input
                , "guess_count" => Encode.int guessCount
                ]
        ]
