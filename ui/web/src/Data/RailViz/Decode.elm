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
    decode RailVizTrainsResponse
        |> required "trains" (list decodeRailVizTrain)
        |> required "routes" (list decodeRailVizRoute)
        |> required "stations" (list decodeStation)


decodeRailVizTrain : JD.Decoder RailVizTrain
decodeRailVizTrain =
    decode RailVizTrain
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
    decode RailVizRoute
        |> required "segments" (list decodeRailVizSegment)


decodeRailVizSegment : JD.Decoder RailVizSegment
decodeRailVizSegment =
    decode RailVizSegment
        |> required "from_station_id" string
        |> required "to_station_id" string
        |> required "coordinates" decodePolyline


decodePolyline : JD.Decoder Polyline
decodePolyline =
    decode Polyline
        |> required "coordinates" (list float)


decodeRailVizStationResponse : JD.Decoder RailVizStationResponse
decodeRailVizStationResponse =
    at [ "content" ] decodeRailVizStationResponseContent


decodeRailVizStationResponseContent : JD.Decoder RailVizStationResponse
decodeRailVizStationResponseContent =
    decode RailVizStationResponse
        |> required "station" decodeStation
        |> required "events" (list decodeRailVizEvent)


decodeRailVizEvent : JD.Decoder RailVizEvent
decodeRailVizEvent =
    decode RailVizEvent
        |> required "trips" (list decodeTripInfo)
        |> optional "type" decodeEventType DEP
        |> required "event" decodeEventInfo


decodeTripInfo : JD.Decoder TripInfo
decodeTripInfo =
    decode TripInfo
        |> required "id" decodeTripId
        |> required "transport" decodeTransportInfo


decodeTrip : JD.Decoder Trip
decodeTrip =
    decode Trip
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
    decode RailVizTripGuessResponse
        |> required "trips" (list decodeTrip)
