module Data.Routing.Types exposing
    ( Interval
    , RoutingRequest
    , RoutingResponse
    , SearchDirection(..)
    , SearchType(..)
    )

import Data.Connection.Types exposing (Connection, Station)
import Date exposing (Date)


type alias RoutingResponse =
    { connections : List Connection
    , intervalStart : Date
    , intervalEnd : Date
    }


type alias RoutingRequest =
    { from : Station
    , to : Station
    , intervalStart : Int
    , intervalEnd : Int
    , minConnectionCount : Int
    , searchDirection : SearchDirection
    , extendIntervalEarlier : Bool
    , extendIntervalLater : Bool
    }


type SearchDirection
    = Forward
    | Backward


type SearchType
    = DefaultSearchType
    | SingleCriterion
    | SingleCriterionNoIntercity
    | LateConnections
    | LateConnectionsTest
    | AccessibilitySearchType


type alias Interval =
    { begin : Int
    , end : Int
    }
