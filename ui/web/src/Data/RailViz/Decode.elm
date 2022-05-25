module Data.RailViz.Decode exposing
    ( decodePolyline
    , decodeRailVizStationResponse
    , decodeRailVizTrainsResponse
    , decodeRailVizTripGuessResponse
    )

import Data.Connection.Decode
    exposing
        ( decodeEventInfo
        , decodeStation
        , decodeTransportInfo
        , decodeTripId
        )
import Data.RailViz.Types exposing (..)
import Json.Decode as JD
    exposing
        ( at
        , fail
        , float
        , int
        , list
        , string
        , succeed
        )
import Json.Decode.Pipeline exposing (decode, optional, required)
import Util.Json exposing (decodeDate)


decodeRailVizTrainsResponse : JD.Decoder RailVizTrainsResponse
decodeRailVizTrainsResponse =
    at [ "content" ] decodeRailVizTrainsResponseContent


decodeRailVizTrainsResponseContent : JD.Decoder RailVizTrainsResponse
decodeRailVizTrainsResponseContent =
    Decode.succeed RailVizTrainsResponse
        |> required "trains" (list decodeRailVizTrain)
        |> required "routes" (list decodeRailVizRoute)
        |> required "stations" (list decodeStation)


decodeRailVizTrain : JD.Decoder RailVizTrain
decodeRailVizTrain =
    Decode.succeed RailVizTrain
        |> required "names" (list string)
        |> required "d_time" decodeDate
        |> required "a_time" decodeDate
        |> required "sched_d_time" decodeDate
        |> required "sched_a_time" decodeDate
        |> optional "route_index" int 0
        |> optional "segment_index" int 0
        |> required "trip" (list decodeTripId)


decodeRailVizRoute : JD.Decoder RailVizRoute
decodeRailVizRoute =
    Decode.succeed RailVizRoute
        |> required "segments" (list decodeRailVizSegment)


decodeRailVizSegment : JD.Decoder RailVizSegment
decodeRailVizSegment =
    Decode.succeed RailVizSegment
        |> required "from_station_id" string
        |> required "to_station_id" string
        |> required "coordinates" decodePolyline


decodePolyline : JD.Decoder Polyline
decodePolyline =
    Decode.succeed Polyline
        |> required "coordinates" (list float)


decodeRailVizStationResponse : JD.Decoder RailVizStationResponse
decodeRailVizStationResponse =
    at [ "content" ] decodeRailVizStationResponseContent


decodeRailVizStationResponseContent : JD.Decoder RailVizStationResponse
decodeRailVizStationResponseContent =
    Decode.succeed RailVizStationResponse
        |> required "station" decodeStation
        |> required "events" (list decodeRailVizEvent)


decodeRailVizEvent : JD.Decoder RailVizEvent
decodeRailVizEvent =
    Decode.succeed RailVizEvent
        |> required "trips" (list decodeTripInfo)
        |> optional "type" decodeEventType DEP
        |> required "event" decodeEventInfo


decodeTripInfo : JD.Decoder TripInfo
decodeTripInfo =
    Decode.succeed TripInfo
        |> required "id" decodeTripId
        |> required "transport" decodeTransportInfo


decodeTrip : JD.Decoder Trip
decodeTrip =
    Decode.succeed Trip
        |> required "first_station" decodeStation
        |> required "trip_info" decodeTripInfo


decodeEventType : JD.Decoder EventType
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
    string |> JD.andThen decodeToType


decodeRailVizTripGuessResponse : JD.Decoder RailVizTripGuessResponse
decodeRailVizTripGuessResponse =
    at [ "content" ] decodeRailVizTripGuessResponseContent


decodeRailVizTripGuessResponseContent : JD.Decoder RailVizTripGuessResponse
decodeRailVizTripGuessResponseContent =
    Decode.succeed RailVizTripGuessResponse
        |> required "trips" (list decodeTrip)
