module Data.PPR.Types exposing
    ( CrossingType(..)
    , Edge
    , EdgeType(..)
    , FootRoutingRequest
    , FootRoutingResponse
    , Route
    , RouteStep
    , RouteStepType(..)
    , Routes
    , SearchOptions
    , StreetType(..)
    , TriState(..)
    )

import Data.Connection.Types exposing (Position)
import Data.RailViz.Types exposing (Polyline)


type alias FootRoutingRequest =
    { start : Position
    , destinations : List Position
    , search_options : SearchOptions
    , include_steps : Bool
    , include_edges : Bool
    , include_path : Bool
    }


type alias SearchOptions =
    { profile : String
    , duration_limit : Float
    }


type alias FootRoutingResponse =
    { routes : List Routes
    }


type EdgeType
    = ConnectionEdge
    | StreetEdge
    | FootwayEdge
    | CrossingEdge
    | ElevatorEdge


type CrossingType
    = NoCrossing
    | GeneratedCrossing
    | UnmarkedCrossing
    | MarkedCrossing
    | IslandCrossing
    | SignalsCrossing


type StreetType
    = ST_No
    | ST_Track
    | ST_Footway
    | ST_Path
    | ST_Cycleway
    | ST_Bridleway
    | ST_Stairs
    | ST_Escalator
    | ST_MovingWalkway
    | ST_Service
    | ST_Pedestrian
    | ST_Living
    | ST_Residential
    | ST_Unclassified
    | ST_Tertiary
    | ST_Secondary
    | ST_Primary
    | ST_Rail
    | ST_Tram


type TriState
    = UNKNOWN
    | NO
    | YES


type alias Edge =
    { distance : Float
    , duration : Float
    , accessibility : Float
    , path : Polyline
    , name : String
    , osm_way_id : Int
    , edge_type : EdgeType
    , street_type : StreetType
    , crossing_type : CrossingType
    , elevation_up : Int
    , elevation_down : Int
    , incline_up : Bool
    , handrail : TriState
    }


type RouteStepType
    = InvalidStep
    | StreetStep
    | FootwayStep
    | CrossingStep
    | ElevatorStep


type alias RouteStep =
    { step_type : RouteStepType
    , street_name : String
    , street_type : StreetType
    , crossing_type : CrossingType
    , distance : Float
    , duration : Float
    , accessibility : Float
    , path : Polyline
    , elevation_up : Int
    , elevation_down : Int
    , incline_up : Bool
    , handrail : TriState
    }


type alias Route =
    { distance : Float
    , duration : Int
    , accessibility : Int
    , start : Position
    , destination : Position
    , steps : List RouteStep
    , edges : List Edge
    , path : Polyline
    , elevation_up : Int
    , elevation_down : Int
    }


type alias Routes =
    { routes : List Route }
