module Data.RailViz.Types exposing
    ( EventType(..)
    , Polyline
    , RailVizEvent
    , RailVizRoute
    , RailVizSegment
    , RailVizStationDirection(..)
    , RailVizStationRequest
    , RailVizStationResponse
    , RailVizTrain
    , RailVizTrainsRequest
    , RailVizTrainsResponse
    , RailVizTripGuessRequest
    , RailVizTripGuessResponse
    , RailVizTripsRequest
    , Trip
    , TripInfo
    )

import Data.Connection.Types
    exposing
        ( EventInfo
        , Position
        , Station
        , TransportInfo
        , TripId
        )
import Date exposing (Date)


type alias RailVizTrainsRequest =
    { corner1 : Position
    , corner2 : Position
    , startTime : Date
    , endTime : Date
    , maxTrains : Int
    }


type alias RailVizTripsRequest =
    { trips : List TripId }


type alias RailVizTrainsResponse =
    { trains : List RailVizTrain
    , routes : List RailVizRoute
    , stations : List Station
    }


type alias RailVizTrain =
    { names : List String
    , depTime : Date
    , arrTime : Date
    , scheduledDepTime : Date
    , scheduledArrTime : Date
    , routeIndex : Int
    , segmentIndex : Int
    , trip : List TripId
    }


type alias RailVizSegment =
    { fromStationId : String
    , toStationId : String
    , coordinates : Polyline
    }


type alias RailVizRoute =
    { segments : List RailVizSegment }


type alias Polyline =
    { coordinates : List Float }


type RailVizStationDirection
    = LATER
    | EARLIER
    | BOTH


type EventType
    = DEP
    | ARR


type alias RailVizStationRequest =
    { stationId : String
    , time : Int
    , eventCount : Int
    , direction : RailVizStationDirection
    , byScheduleTime : Bool
    }


type alias RailVizStationResponse =
    { station : Station
    , events : List RailVizEvent
    }


type alias RailVizEvent =
    { trips : List TripInfo
    , eventType : EventType
    , event : EventInfo
    }


type alias TripInfo =
    { id : TripId
    , transport : TransportInfo
    }


type alias Trip =
    { firstStation : Station
    , tripInfo : TripInfo
    }


type alias RailVizTripGuessRequest =
    { trainNum : Int
    , time : Int
    , guessCount : Int
    }


type alias RailVizTripGuessResponse =
    { trips : List Trip }
