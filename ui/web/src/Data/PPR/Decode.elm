module Data.PPR.Decode exposing
    ( decodeFootRoutingResponse
    , decodeRoute
    , decodeSearchOptions
    )

import Data.Connection.Decode exposing (decodePosition)
import Data.PPR.Types exposing (..)
import Data.RailViz.Decode exposing (decodePolyline)
import Json.Decode as Decode exposing (bool, fail, float, int, string, succeed)
import Json.Decode.Pipeline exposing (decode, optional, required)


decodeFootRoutingResponse : Decode.Decoder FootRoutingResponse
decodeFootRoutingResponse =
    Decode.at [ "content" ] decodeFootRoutingResponseContent


decodeFootRoutingResponseContent : Decode.Decoder FootRoutingResponse
decodeFootRoutingResponseContent =
    Decode.succeed FootRoutingResponse
        |> optional "routes" (Decode.list decodeRoutes) []


decodeRoutes : Decode.Decoder Routes
decodeRoutes =
    Decode.succeed Routes
        |> optional "routes" (Decode.list decodeRoute) []


decodeEdgeType : Decode.Decoder EdgeType
decodeEdgeType =
    let
        decodeToType string =
            case string of
                "CONNECTION" ->
                    succeed ConnectionEdge

                "STREET" ->
                    succeed StreetEdge

                "FOOTWAY" ->
                    succeed FootwayEdge

                "CROSSING" ->
                    succeed CrossingEdge

                "ELEVATOR" ->
                    succeed ElevatorEdge

                _ ->
                    fail ("Not valid pattern for decoder to EdgeType. Pattern: " ++ toString string)
    in
    Decode.string |> Decode.andThen decodeToType


decodeCrossingType : Decode.Decoder CrossingType
decodeCrossingType =
    let
        decodeToType string =
            case string of
                "NONE" ->
                    succeed NoCrossing

                "GENERATED" ->
                    succeed GeneratedCrossing

                "UNMARKED" ->
                    succeed UnmarkedCrossing

                "MARKED" ->
                    succeed MarkedCrossing

                "ISLAND" ->
                    succeed IslandCrossing

                "SIGNALS" ->
                    succeed SignalsCrossing

                _ ->
                    fail ("Not valid pattern for decoder to CrossingType. Pattern: " ++ toString string)
    in
    Decode.string |> Decode.andThen decodeToType


decodeStreetType : Decode.Decoder StreetType
decodeStreetType =
    let
        decodeToType string =
            case string of
                "NONE" ->
                    succeed ST_No

                "TRACK" ->
                    succeed ST_Track

                "FOOTWAY" ->
                    succeed ST_Footway

                "PATH" ->
                    succeed ST_Path

                "CYCLEWAY" ->
                    succeed ST_Cycleway

                "BRIDLEWAY" ->
                    succeed ST_Bridleway

                "STAIRS" ->
                    succeed ST_Stairs

                "ESCALATOR" ->
                    succeed ST_Escalator

                "MOVING_WALKWAY" ->
                    succeed ST_MovingWalkway

                "SERVICE" ->
                    succeed ST_Service

                "PEDESTRIAN" ->
                    succeed ST_Pedestrian

                "LIVING" ->
                    succeed ST_Living

                "RESIDENTIAL" ->
                    succeed ST_Residential

                "UNCLASSIFIED" ->
                    succeed ST_Unclassified

                "TERTIARY" ->
                    succeed ST_Tertiary

                "SECONDARY" ->
                    succeed ST_Secondary

                "PRIMARY" ->
                    succeed ST_Primary

                "RAIL" ->
                    succeed ST_Rail

                "TRAM" ->
                    succeed ST_Tram

                _ ->
                    fail ("Not valid pattern for decoder to StreetType. Pattern: " ++ toString string)
    in
    Decode.string |> Decode.andThen decodeToType


decodeRouteStepType : Decode.Decoder RouteStepType
decodeRouteStepType =
    let
        decodeToType string =
            case string of
                "INVALID" ->
                    succeed InvalidStep

                "STREET" ->
                    succeed StreetStep

                "FOOTWAY" ->
                    succeed FootwayStep

                "CROSSING" ->
                    succeed CrossingStep

                "ELEVATOR" ->
                    succeed ElevatorStep

                _ ->
                    fail ("Not valid pattern for decoder to RouteStepType. Pattern: " ++ toString string)
    in
    Decode.string |> Decode.andThen decodeToType


decodeTriState : Decode.Decoder TriState
decodeTriState =
    let
        decodeToTriState string =
            case string of
                "UNKNOWN" ->
                    succeed UNKNOWN

                "NO" ->
                    succeed NO

                "YES" ->
                    succeed YES

                _ ->
                    fail ("Not valid pattern for decoder to TriState. Pattern: " ++ toString string)
    in
    Decode.string |> Decode.andThen decodeToTriState


decodeEdge : Decode.Decoder Edge
decodeEdge =
    Decode.succeed Edge
        |> optional "distance" float 0
        |> optional "duration" float 0
        |> optional "accessibility" float 0
        |> required "path" decodePolyline
        |> required "name" string
        |> optional "osm_way_id" int 0
        |> optional "edge_type" decodeEdgeType ConnectionEdge
        |> optional "street_type" decodeStreetType ST_No
        |> optional "crossing_type" decodeCrossingType NoCrossing
        |> optional "elevation_up" int 0
        |> optional "elevation_down" int 0
        |> optional "incline_up" bool False
        |> optional "handrail" decodeTriState UNKNOWN


decodeRouteStep : Decode.Decoder RouteStep
decodeRouteStep =
    Decode.succeed RouteStep
        |> optional "step_type" decodeRouteStepType InvalidStep
        |> required "street_name" string
        |> optional "street_type" decodeStreetType ST_No
        |> optional "crossing_type" decodeCrossingType NoCrossing
        |> optional "distance" float 0
        |> optional "duration" float 0
        |> optional "accessibility" float 0
        |> required "path" decodePolyline
        |> optional "elevation_up" int 0
        |> optional "elevation_down" int 0
        |> optional "incline_up" bool False
        |> optional "handrail" decodeTriState UNKNOWN


decodeRoute : Decode.Decoder Route
decodeRoute =
    Decode.succeed Route
        |> optional "distance" float 0
        |> optional "duration" int 0
        |> optional "accessibility" int 0
        |> required "start" decodePosition
        |> required "destination" decodePosition
        |> optional "steps" (Decode.list decodeRouteStep) []
        |> optional "edges" (Decode.list decodeEdge) []
        |> optional "path" decodePolyline { coordinates = [] }
        |> optional "elevation_up" int 0
        |> optional "elevation_down" int 0


decodeSearchOptions : Decode.Decoder SearchOptions
decodeSearchOptions =
    Decode.succeed SearchOptions
        |> required "profile" string
        |> required "duration_limit" float
