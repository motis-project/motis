module Data.Intermodal.Types exposing
    ( BikeModeInfo
    , CarModeInfo
    , FootModeInfo
    , FootPPRInfo
    , IntermodalDestination(..)
    , IntermodalPretripStartInfo
    , IntermodalRoutingRequest
    , IntermodalStart(..)
    , Mode(..)
    , PretripStartInfo
    )

import Data.Connection.Types exposing (Connection, Position, Station)
import Data.PPR.Types exposing (SearchOptions)
import Data.Routing.Types exposing (Interval, SearchDirection, SearchType)


type alias IntermodalRoutingRequest =
    { start : IntermodalStart
    , startModes : List Mode
    , destination : IntermodalDestination
    , destinationModes : List Mode
    , searchType : SearchType
    , searchDir : SearchDirection
    }


type IntermodalStart
    = IntermodalPretripStart IntermodalPretripStartInfo
    | PretripStart PretripStartInfo


type alias IntermodalPretripStartInfo =
    { position : Position
    , interval : Interval
    , minConnectionCount : Int
    , extendIntervalEarlier : Bool
    , extendIntervalLater : Bool
    }


type alias PretripStartInfo =
    { station : Station
    , interval : Interval
    , minConnectionCount : Int
    , extendIntervalEarlier : Bool
    , extendIntervalLater : Bool
    }


type IntermodalDestination
    = InputStation Station
    | InputPosition Position


type Mode
    = Foot FootModeInfo
    | Bike BikeModeInfo
    | GBFS GBFSModeInfo
    | Car CarModeInfo
    | FootPPR FootPPRInfo
    | CarParking CarParkingModeInfo


type alias FootModeInfo =
    { maxDuration : Int }


type alias BikeModeInfo =
    { maxDuration : Int }


type alias GBFSModeInfo =
    { maxWalkDuration : Int
    , maxVehicleDuration : Int
    , provider : String
    }


type alias CarModeInfo =
    { maxDuration : Int }


type alias FootPPRInfo =
    { searchOptions : SearchOptions }


type alias CarParkingModeInfo =
    { maxCarDuration : Int
    , pprSearchOptions : SearchOptions
    }
