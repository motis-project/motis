module Data.StationGuesser.Decode exposing (decodeStationGuesserResponse)

import Data.Connection.Decode exposing (decodeStation)
import Data.Connection.Types exposing (Station)
import Json.Decode as Decode


decodeStationGuesserResponse : Decode.Decoder (List Station)
decodeStationGuesserResponse =
    Decode.at [ "content", "guesses" ] (Decode.list decodeStation)
