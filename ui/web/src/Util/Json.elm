module Util.Json exposing (decodeDate)

import Time exposing (millisToPosix, utc)
import Json.Decode as Decode


decodeDate : Decode.Decoder Time.Posix
decodeDate =
    Decode.int |> Decode.andThen (Decode.succeed << millisToPosix << (\i -> i * 1000))
