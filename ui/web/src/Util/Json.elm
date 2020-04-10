module Util.Json exposing (decodeDate)

import Date exposing (Date)
import Json.Decode as Decode


decodeDate : Decode.Decoder Date
decodeDate =
    Decode.int |> Decode.andThen (Decode.succeed << Date.fromTime << toFloat << (\i -> i * 1000))
