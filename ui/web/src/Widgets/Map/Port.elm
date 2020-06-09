port module Widgets.Map.Port exposing
    ( MapClickInfo
    , MapFitBounds
    , MapFlyLocation
    , MapGeoBounds
    , MapInfo
    , MapLocale
    , MapMarkerSettings
    , MapPixelBounds
    , MapTooltip
    , RVConnection
    , RVConnectionSection
    , RVConnectionSegmentTrip
    , RVConnectionSegmentWalk
    , RVConnectionTrain
    , RVConnectionWalk
    , RVConnections
    , RVDetailFilter
    , RVTrain
    , mapCloseContextMenu
    , mapFitBounds
    , mapFlyTo
    , mapHighlightConnections
    , mapInit
    , mapSetDetailFilter
    , mapSetConnections
    , mapSetLocale
    , mapSetMarkers
    , mapSetTooltip
    , mapShowContextMenu
    , mapShowTrains
    , mapUpdate
    , mapUpdateWalks
    , mapUseTrainClassColors
    )

import Data.Connection.Types exposing (Position, Station, TripId)
import Time exposing (Time)


type alias MapInfo =
    { scale : Float
    , zoom : Float
    , pixelBounds : MapPixelBounds
    , geoBounds : MapGeoBounds
    , railVizBounds : MapGeoBounds
    , center : Position
    , bearing : Float
    , pitch : Float
    }


type alias MapPixelBounds =
    { north : Float
    , west : Float
    , width : Float
    , height : Float
    }


type alias MapGeoBounds =
    { north : Float
    , west : Float
    , south : Float
    , east : Float
    }


type alias MapTooltip =
    { mouseX : Int
    , mouseY : Int
    , hoveredTrain : Maybe RVTrain
    , hoveredStation : Maybe String
    , hoveredTripSegments : Maybe (List RVConnectionSegmentTrip)
    , hoveredWalkSegment : Maybe RVConnectionSegmentWalk
    }


type alias MapClickInfo =
    { mouseX : Int
    , mouseY : Int
    , lat : Float
    , lng : Float
    }


type alias RVTrain =
    { names : List String
    , departureTime : Time
    , arrivalTime : Time
    , scheduledDepartureTime : Time
    , scheduledArrivalTime : Time
    , hasDepartureDelayInfo : Bool
    , hasArrivalDelayInfo : Bool
    , departureStation : String
    , arrivalStation : String
    }


type alias RVConnectionSegmentTrip =
    { connectionIds : List Int
    , trip : List TripId
    , d_station_id : String
    , a_station_id : String
    }

type alias RVConnectionSegmentWalk =
    { connectionIds : List Int
    , walk : RVConnectionWalk
    }


type alias MapFlyLocation =
    { mapId : String
    , lat : Float
    , lng : Float
    , zoom : Maybe Float
    , bearing : Maybe Float
    , pitch : Maybe Float
    , animate : Bool
    }


type alias MapFitBounds =
    { mapId : String
    , coords : List (List Float)
    }


type alias RVDetailFilter =
    { trains : List RVConnectionTrain
    , walks : List RVConnectionWalk
    , interchangeStations : List Station
    }


type alias RVConnectionTrain =
    { sections : List RVConnectionSection
    , trip : Maybe TripId
    }


type alias RVConnectionSection =
    { departureStation : Station
    , arrivalStation : Station
    , scheduledDepartureTime : Time
    , scheduledArrivalTime : Time
    }


type alias RVConnectionWalk =
    { departureStation : Station
    , arrivalStation : Station
    , polyline : Maybe (List Float)
    , mumoType : String
    , duration : Int
    , accessibility : Int
    }


type alias MapMarkerSettings =
    { startPosition : Maybe Position
    , destinationPosition : Maybe Position
    , startName : Maybe String
    , destinationName : Maybe String
    }


type alias MapLocale =
    { start : String
    , destination : String
    }


type alias RVConnections =
    { mapId : String
    , connections : List RVConnection
    , lowestId : Int
    }


type alias RVConnection =
    { id : Int
    , stations : List Station
    , trains : List RVConnectionTrain
    , walks : List RVConnectionWalk
    }


port mapInit : String -> Cmd msg


port mapUpdate : (MapInfo -> msg) -> Sub msg


port mapSetTooltip : (MapTooltip -> msg) -> Sub msg


port mapFlyTo : MapFlyLocation -> Cmd msg


port mapFitBounds : MapFitBounds -> Cmd msg


port mapUseTrainClassColors : Bool -> Cmd msg


port mapShowTrains : Bool -> Cmd msg


port mapSetDetailFilter : Maybe RVDetailFilter -> Cmd msg


port mapUpdateWalks : List RVConnectionWalk -> Cmd msg


port mapShowContextMenu : (MapClickInfo -> msg) -> Sub msg


port mapCloseContextMenu : (() -> msg) -> Sub msg


port mapSetMarkers : MapMarkerSettings -> Cmd msg


port mapSetLocale : MapLocale -> Cmd msg


port mapSetConnections : RVConnections -> Cmd msg


port mapHighlightConnections : List Int -> Cmd msg
