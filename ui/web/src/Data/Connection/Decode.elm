module Data.Connection.Decode exposing
    ( decodeConnection
    , decodeEventInfo
    , decodePosition
    , decodeStation
    , decodeTransportInfo
    , decodeTripId
    )

import Data.Connection.Types exposing (..)
import Json.Decode as Decode
    exposing
        ( bool
        , fail
        , field
        , float
        , int
        , list
        , nullable
        , string
        , succeed
        )
import Json.Decode.Pipeline exposing (decode, optional, required)
import Util.Json exposing (decodeDate)


decodeConnection : Decode.Decoder Connection
decodeConnection =
    decode Connection
        |> required "stops" (list decodeStop)
        |> required "transports" (list decodeMove)
        |> required "trips" (list decodeTrip)
        |> optional "problems" (list decodeProblem) []


decodeStop : Decode.Decoder Stop
decodeStop =
    decode Stop
        |> required "station" decodeStation
        |> required "arrival" decodeEventInfo
        |> required "departure" decodeEventInfo
        |> optional "exit" bool False
        |> optional "enter" bool False


decodeStation : Decode.Decoder Station
decodeStation =
    decode Station
        |> required "id" string
        |> required "name" string
        |> required "pos" decodePosition


decodePosition : Decode.Decoder Position
decodePosition =
    decode Position
        |> required "lat" float
        |> required "lng" float


decodeEventInfo : Decode.Decoder EventInfo
decodeEventInfo =
    decode EventInfo
        |> optional "time" (nullable decodeDate) Nothing
        |> optional "schedule_time" (nullable decodeDate) Nothing
        |> required "track" string
        |> optional "reason" decodeTimestampReason Schedule


decodeMove : Decode.Decoder Move
decodeMove =
    let
        move : String -> Decode.Decoder Move
        move move_type =
            case move_type of
                "Transport" ->
                    decode Transport
                        |> required "move" decodeTransportInfo

                "Walk" ->
                    decode Walk
                        |> required "move" decodeWalkInfo

                _ ->
                    Decode.fail ("move type " ++ move_type ++ " not supported")
    in
    field "move_type" string |> Decode.andThen move


decodeTransportInfo : Decode.Decoder TransportInfo
decodeTransportInfo =
    decode TransportInfo
        |> required "range" decodeRange
        |> optional "clasz" int 0
        |> required "line_id" string
        |> required "name" string
        |> required "provider" string
        |> required "direction" string


decodeWalkInfo : Decode.Decoder WalkInfo
decodeWalkInfo =
    decode WalkInfo
        |> required "range" decodeRange
        |> optional "mumo_id" int 0
        |> optional "price" int 0
        |> optional "accessibility" int 0
        |> required "mumo_type" string


decodeAttribute : Decode.Decoder Attribute
decodeAttribute =
    decode Attribute
        |> required "range" decodeRange
        |> required "code" string
        |> required "text" string


decodeRange : Decode.Decoder Range
decodeRange =
    decode Range
        |> required "from" int
        |> required "to" int


decodeTimestampReason : Decode.Decoder TimestampReason
decodeTimestampReason =
    let
        decodeToType string =
            case string of
                "SCHEDULE" ->
                    succeed Schedule

                "IS" ->
                    succeed Is

                "REPAIR" ->
                    succeed Is

                "PROPAGATION" ->
                    succeed Propagation

                "FORECAST" ->
                    succeed Forecast

                _ ->
                    fail ("Not valid pattern for decoder to TimestampReason. Pattern: " ++ toString string)
    in
    Decode.string |> Decode.andThen decodeToType


decodeTrip : Decode.Decoder Trip
decodeTrip =
    decode Trip
        |> required "range" decodeRange
        |> required "id" decodeTripId


decodeTripId : Decode.Decoder TripId
decodeTripId =
    decode TripId
        |> required "id" string
        |> required "station_id" string
        |> optional "train_nr" int 0
        |> required "time" int
        |> required "target_station_id" string
        |> required "target_time" int
        |> required "line_id" string


decodeProblemType : Decode.Decoder ProblemType
decodeProblemType =
    let
        decodeToType string =
            case string of
                "NO_PROBLEM" ->
                    succeed NoProblem

                "INTERCHANGE_TIME_VIOLATED" ->
                    succeed InterchangeTimeViolated

                "CANCELED_TRAIN" ->
                    succeed CanceledTrain

                _ ->
                    fail ("Unsupported problem type: " ++ string)
    in
    Decode.string |> Decode.andThen decodeToType


decodeProblem : Decode.Decoder Problem
decodeProblem =
    decode Problem
        |> required "range" decodeRange
        |> optional "type" decodeProblemType NoProblem
