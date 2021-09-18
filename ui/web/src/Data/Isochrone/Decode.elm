module Data.Isochrone.Decode exposing (decodeIsochroneResponse)

import Data.Connection.Decode exposing (decodeStation)
import Data.Connection.Types exposing (Station)
import Json.Decode as Decode
import Json.Decode.Pipeline exposing (decode, required)
import Data.Isochrone.Types exposing (IsochroneResponse)


decodeIsochroneResponse : Decode.Decoder IsochroneResponse
decodeIsochroneResponse =
    Decode.at [ "content" ] decodeIsochroneResponseContent

decodeIsochroneResponseContent : Decode.Decoder IsochroneResponse
decodeIsochroneResponseContent =
    decode IsochroneResponse
        |> required "stations" (Decode.list decodeStation)
        |> required "arrival_times" (Decode.list Decode.int)

