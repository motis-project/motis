module Data.RailViz.Request exposing
    ( encodeDate
    , encodeDirection
    , encodePosition
    , encodeStationRequest
    , encodeTrainsRequest
    , encodeTripGuessRequest
    , encodeTripsRequest
    , initialStationRequest
    )

import Data.Connection.Types exposing (Position)
import Data.Lookup.Request exposing (encodeTripId)
import Data.RailViz.Types
    exposing
        ( RailVizStationDirection(..)
        , RailVizStationRequest
        , RailVizTrainsRequest
        , RailVizTripGuessRequest
        , RailVizTripsRequest
        )
import Date exposing (Date)
import Json.Encode as Encode
import Util.DateUtil exposing (unixTime)


encodeTrainsRequest : RailVizTrainsRequest -> Encode.Value
encodeTrainsRequest request =
    Encode.object
        [ "destination"
            , Encode.object
                [ "type" , Encode.string "Module"
                , "target" , Encode.string "/railviz/get_trains"
                ]
        , "content_type" , Encode.string "RailVizTrainsRequest"
        , "content"
            , Encode.object
                [ "corner1" , encodePosition request.corner1
                , "corner2" , encodePosition request.corner2
                , "start_time" , encodeDate request.startTime
                , "end_time" , encodeDate request.endTime
                , "max_trains" , Encode.int request.maxTrains
                ]
        ]


encodeTripsRequest : RailVizTripsRequest -> Encode.Value
encodeTripsRequest request =
    Encode.object
        [ "destination"
            , Encode.object
                [ "type" , Encode.string "Module"
                , "target" , Encode.string "/railviz/get_trips"
                ]
        , "content_type" , Encode.string "RailVizTripsRequest"
        , "content"
            , Encode.object
                [ "trips" , Encode.list (List.map encodeTripId request.trips) ]
        ]


encodePosition : Position -> Encode.Value
encodePosition pos =
    Encode.object
        [ "lat" , Encode.float pos.lat
        , "lng" , Encode.float pos.lng
        ]


encodeDate : Date -> Encode.Value
encodeDate date =
    Encode.int (unixTime date)


encodeStationRequest : RailVizStationRequest -> Encode.Value
encodeStationRequest req =
    Encode.object
        [ "destination"
            , Encode.object
                [ "type" , Encode.string "Module"
                , "target" , Encode.string "/railviz/get_station"
                ]
        , "content_type" , Encode.string "RailVizStationRequest"
        , "content"
            , Encode.object
                [ "station_id" , Encode.string req.stationId
                , "time" , Encode.int req.time
                , "event_count" , Encode.int req.eventCount
                , "direction" , encodeDirection req.direction
                , "by_schedule_time" , Encode.bool req.byScheduleTime
                ]
        ]


encodeDirection : RailVizStationDirection -> Encode.Value
encodeDirection dir =
    case dir of
        LATER ->
            Encode.string "LATER"

        EARLIER ->
            Encode.string "EARLIER"

        BOTH ->
            Encode.string "BOTH"


initialStationRequest : String -> Date -> RailVizStationRequest
initialStationRequest stationId date =
    let
        selectedTime =
            unixTime date
    in
    { stationId = stationId
    , time = selectedTime
    , eventCount = 20
    , direction = BOTH
    , byScheduleTime = True
    }


encodeTripGuessRequest : RailVizTripGuessRequest -> Encode.Value
encodeTripGuessRequest req =
    Encode.object
        [ "destination"
            , Encode.object
                [ "type" , Encode.string "Module"
                , "target" , Encode.string "/railviz/get_trip_guesses"
                ]
        , "content_type" , Encode.string "RailVizTripGuessRequest"
        , "content"
            , Encode.object
                [ "train_num" , Encode.int req.trainNum
                , "time" , Encode.int req.time
                , "guess_count" , Encode.int req.guessCount
                ]
        ]
