module Data.Parking.Decode exposing
    ( decodeParking
    , decodeParkingEdgeResponse
    , decodeParkingEdgeResponseContent
    )

import Data.Connection.Decode exposing (decodePosition)
import Data.OSRM.Decode exposing (decodeOSRMViaRouteResponseContent)
import Data.PPR.Decode exposing (decodeRoute)
import Data.Parking.Types exposing (..)
import Json.Decode as Decode
import Json.Decode.Pipeline exposing (decode, optional, required)


decodeParkingEdgeResponse : Decode.Decoder ParkingEdgeResponse
decodeParkingEdgeResponse =
    Decode.at [ "content" ] decodeParkingEdgeResponseContent


decodeParkingEdgeResponseContent : Decode.Decoder ParkingEdgeResponse
decodeParkingEdgeResponseContent =
    decode ParkingEdgeResponse
        |> required "parking" decodeParking
        |> required "car" decodeOSRMViaRouteResponseContent
        |> required "walk" decodeRoute
        |> optional "uses_car" Decode.bool False


decodeParking : Decode.Decoder Parking
decodeParking =
    decode Parking
        |> optional "id" Decode.int 0
        |> required "pos" decodePosition
        |> optional "fee" Decode.bool False
