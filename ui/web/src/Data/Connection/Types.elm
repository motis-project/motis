module Data.Connection.Types exposing
    ( Attribute
    , Connection
    , EventInfo
    , Move(..)
    , Position
    , Problem
    , ProblemType(..)
    , Range
    , Station
    , Stop
    , TimestampReason(..)
    , TransportInfo
    , Trip
    , TripId
    , WalkInfo
    , arrivalEvent
    , arrivalTime
    , departureEvent
    , departureTime
    , duration
    , eventIsInThePast
    , getEventTime
    , hasNoProblems
    , interchanges
    , transportCategories
    , transportsForRange
    , tripsForRange
    )

import Date exposing (Date)
import Date.Extra.Compare exposing (Compare2(..))
import Date.Extra.Duration as Duration exposing (DeltaRecord)
import Maybe.Extra
import Set exposing (Set)
import Util.List exposing (..)


type alias Connection =
    { stops : List Stop
    , transports : List Move
    , trips : List Trip
    , problems : List Problem
    }


type alias Stop =
    { station : Station
    , arrival : EventInfo
    , departure : EventInfo
    , exit : Bool
    , enter : Bool
    }


type alias Station =
    { id : String
    , name : String
    , pos : Position
    }


type alias Position =
    { lat : Float
    , lng : Float
    }


type alias EventInfo =
    { time : Maybe Date
    , schedule_time : Maybe Date
    , track : String
    , reason : TimestampReason
    }


type TimestampReason
    = Schedule
    | Is
    | Propagation
    | Forecast


type Move
    = Transport TransportInfo
    | Walk WalkInfo


type alias TransportInfo =
    { range : Range
    , category_name : String
    , class : Int
    , train_nr : Maybe Int
    , line_id : String
    , name : String
    , provider : String
    , direction : String
    }


type alias WalkInfo =
    { range : Range
    , mumo_id : Int
    , price : Int
    , accessibility : Int
    , mumo_type : String
    }


type alias Attribute =
    { range : Range
    , code : String
    , text : String
    }


type alias Range =
    { from : Int
    , to : Int
    }


type alias TripId =
    { station_id : String
    , train_nr : Int
    , time : Int
    , target_station_id : String
    , target_time : Int
    , line_id : String
    }


type alias Trip =
    { range : Range
    , id : TripId
    }


type alias Problem =
    { range : Range
    , typ : ProblemType
    }


type ProblemType
    = NoProblem
    | InterchangeTimeViolated
    | CanceledTrain


departureEvent : Connection -> Maybe EventInfo
departureEvent connection =
    List.head connection.stops |> Maybe.map .departure


arrivalEvent : Connection -> Maybe EventInfo
arrivalEvent connection =
    last connection.stops |> Maybe.map .arrival


departureTime : Connection -> Maybe Date
departureTime connection =
    departureEvent connection |> Maybe.andThen .schedule_time


arrivalTime : Connection -> Maybe Date
arrivalTime connection =
    arrivalEvent connection |> Maybe.andThen .schedule_time


duration : Connection -> Maybe DeltaRecord
duration connection =
    Maybe.map2 Duration.diff (arrivalTime connection) (departureTime connection)


interchanges : Connection -> Int
interchanges connection =
    Basics.max 0 ((List.length <| List.filter .enter connection.stops) - 1)


hasNoProblems : Connection -> Bool
hasNoProblems connection =
    connection.problems
        |> List.filter (\p -> p.typ /= NoProblem)
        |> List.isEmpty


transportsForRange : Connection -> Int -> Int -> List TransportInfo
transportsForRange connection from to =
    let
        checkMove : Move -> Maybe TransportInfo
        checkMove move =
            case move of
                Transport transport ->
                    if transport.range.from < to && transport.range.to > from then
                        Just transport

                    else
                        Nothing

                Walk _ ->
                    Nothing
    in
    List.filterMap checkMove connection.transports
        |> List.sortBy (\t -> -(t.range.to - t.range.from))


tripsForRange : Connection -> Int -> Int -> List TripId
tripsForRange connection from to =
    let
        checkTrip : Trip -> Maybe Trip
        checkTrip trip =
            if trip.range.from < to && trip.range.to > from then
                Just trip

            else
                Nothing
    in
    List.filterMap checkTrip connection.trips
        |> List.sortBy (\t -> (t.range.from, -t.range.to, t.id.train_nr))
        |> List.map .id


transportCategories : Connection -> Set String
transportCategories connection =
    let
        category : Move -> Maybe String
        category move =
            case move of
                Transport t ->
                    Just t.category_name

                Walk _ ->
                    Nothing
    in
    List.filterMap category connection.transports |> Set.fromList


getEventTime : EventInfo -> Maybe Date
getEventTime event =
    Maybe.Extra.or
        event.time
        event.schedule_time


eventIsInThePast : Date -> EventInfo -> Bool
eventIsInThePast currentTime event =
    let
        eventTime =
            getEventTime event
    in
    case eventTime of
        Just t ->
            Date.Extra.Compare.is SameOrBefore t currentTime

        Nothing ->
            False
