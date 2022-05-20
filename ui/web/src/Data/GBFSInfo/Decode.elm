module Data.GBFSInfo.Decode exposing (decodeGBFSInfoResponse)

import Data.GBFSInfo.Types exposing (..)
import Json.Decode as Decode exposing (list, string)
import Json.Decode.Pipeline exposing (decode, required)
import Util.Json exposing (decodeDate)


decodeGBFSInfoResponse : Decode.Decoder GBFSInfo
decodeGBFSInfoResponse =
    Decode.at [ "content" ] decodeGBFSInfo


decodeGBFSInfo : Decode.Decoder GBFSInfo
decodeGBFSInfo =
    decode GBFSInfo
        |> required "providers" (list decodeGBFSProvider)


decodeGBFSProvider : Decode.Decoder GBFSProvider
decodeGBFSProvider =
    decode GBFSProvider
        |> required "name" string
        |> required "vehicle_type" string
        |> required "tag" string
