module Data.Lookup.Decode exposing
    ( decodeLookupStationEventsResponse
    , decodeTripToConnectionResponse
    )

import Data.Connection.Decode exposing (decodeConnection, decodeTripId)
import Data.Connection.Types exposing (Connection)
import Data.Lookup.Types exposing (..)
import Json.Decode as Decode
    exposing
        ( Decoder
        , fail
        , int
        , list
        , string
        , succeed
        )
import Json.Decode.Pipeline exposing (decode, optional, required, requiredAt)
import Util.Json exposing (decodeDate)


decodeTripToConnectionResponse : Decoder Connection
decodeTripToConnectionResponse =
    Decode.at [ "content" ] decodeConnection


decodeLookupStationEventsResponse : Decoder LookupStationEventsResponse
decodeLookupStationEventsResponse =
    decode LookupStationEventsResponse
        |> requiredAt [ "content", "events" ] (list decodeStationEvent)


decodeStationEvent : Decoder StationEvent
decodeStationEvent =
    decode StationEvent
        |> required "trip_id" (list decodeTripId)
        |> optional "type" decodeEventType DEP
        |> optional "train_nr" int 0
        |> required "line_id" string
        |> required "time" decodeDate
        |> required "schedule_time" decodeDate
        |> required "direction" string
        |> required "service_name" string
        |> required "track" string


decodeEventType : Decoder EventType
decodeEventType =
    let
        decodeToType string =
            case string of
                "DEP" ->
                    succeed DEP

                "ARR" ->
                    succeed ARR

                _ ->
                    fail ("Not valid pattern for decoder to EventType. Pattern: " ++ toString string)
    in
    string |> Decode.andThen decodeToType
