module Data.Parking.Request exposing (encodeParkingEdgeDirection, encodeParkingEdgeRequest)

import Data.PPR.Request exposing (encodeSearchProfile)
import Data.Parking.Types exposing (..)
import Data.RailViz.Request exposing (encodePosition)
import Json.Encode as Encode
import Util.Core exposing ((=>))


encodeParkingEdgeDirection : ParkingEdgeDirection -> Encode.Value
encodeParkingEdgeDirection dir =
    case dir of
        Outward ->
            Encode.string "Outward"

        Return ->
            Encode.string "Return"


encodeParkingEdgeRequest : ParkingEdgeRequest -> Encode.Value
encodeParkingEdgeRequest request =
    Encode.object
        [ "destination"
            => Encode.object
                [ "type" => Encode.string "Module"
                , "target" => Encode.string "/parking/edge"
                ]
        , "content_type" => Encode.string "ParkingEdgeRequest"
        , "content"
            => Encode.object
                [ "id" => Encode.int request.id
                , "start" => encodePosition request.start
                , "destination" => encodePosition request.destination
                , "direction" => encodeParkingEdgeDirection request.direction
                , "ppr_search_profile" => encodeSearchProfile request.ppr_search_profile
                , "duration" => Encode.int request.duration
                , "accessibility" => Encode.int request.accessibility
                , "include_steps" => Encode.bool request.include_steps
                , "include_edges" => Encode.bool request.include_edges
                , "include_path" => Encode.bool request.include_path
                ]
        ]
