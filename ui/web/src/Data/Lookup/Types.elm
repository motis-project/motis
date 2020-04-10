module Data.Lookup.Types exposing
    ( EventType(..)
    , LookupStationEventsRequest
    , LookupStationEventsResponse
    , StationEvent
    , TableType(..)
    )

import Data.Connection.Types exposing (TripId)
import Date exposing (Date)


type TableType
    = ArrivalsAndDepartures
    | OnlyArrivals
    | OnlyDepartures


type alias LookupStationEventsRequest =
    { stationId : String
    , intervalStart : Int
    , intervalEnd : Int
    , tableType : TableType
    }


type alias LookupStationEventsResponse =
    { events : List StationEvent }


type EventType
    = DEP
    | ARR


type alias StationEvent =
    { tripId : List TripId
    , eventType : EventType
    , trainNr : Int
    , lineId : String
    , time : Date
    , scheduleTime : Date
    , direction : String
    , serviceName : String
    , track : String
    }
