module Data.OSRM.Decode exposing (decodeOSRMViaRouteResponse, decodeOSRMViaRouteResponseContent)

import Data.OSRM.Types exposing (..)
import Data.RailViz.Decode exposing (decodePolyline)
import Json.Decode as Decode exposing (float, int)
import Json.Decode.Pipeline exposing (decode, optional, required)


decodeOSRMViaRouteResponse : Decode.Decoder OSRMViaRouteResponse
decodeOSRMViaRouteResponse =
    Decode.at [ "content" ] decodeOSRMViaRouteResponseContent


decodeOSRMViaRouteResponseContent : Decode.Decoder OSRMViaRouteResponse
decodeOSRMViaRouteResponseContent =
    decode OSRMViaRouteResponse
        |> optional "time" int 0
        |> optional "distance" float 0.0
        |> required "polyline" decodePolyline
