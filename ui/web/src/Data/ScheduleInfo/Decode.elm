module Data.ScheduleInfo.Decode exposing (decodeScheduleInfoResponse)

import Data.ScheduleInfo.Types exposing (..)
import Json.Decode as Decode exposing (string)
import Json.Decode.Pipeline exposing (decode, required)
import Util.Json exposing (decodeDate)


decodeScheduleInfoResponse : Decode.Decoder ScheduleInfo
decodeScheduleInfoResponse =
    Decode.at [ "content" ] decodeScheduleInfo


decodeScheduleInfo : Decode.Decoder ScheduleInfo
decodeScheduleInfo =
    Decode.succeed ScheduleInfo
        |> required "name" string
        |> required "begin" decodeDate
        |> required "end" decodeDate
