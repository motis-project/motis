module Data.Address.Decode exposing (decodeAddress, decodeAddressResponse)

import Data.Address.Types exposing (..)
import Data.Connection.Decode exposing (decodePosition)
import Json.Decode as Decode
    exposing
        ( Decoder
        , int
        , list
        , string
        )
import Json.Decode.Pipeline exposing (decode, required, requiredAt)


decodeAddressResponse : Decoder AddressResponse
decodeAddressResponse =
    decode AddressResponse
        |> requiredAt [ "content", "guesses" ] (list decodeAddress)


decodeAddress : Decoder Address
decodeAddress =
    decode Address
        |> required "pos" decodePosition
        |> required "name" string
        |> required "type" string
        |> required "regions" (list decodeRegion)


decodeRegion : Decoder Region
decodeRegion =
    decode Region
        |> required "name" string
        |> required "admin_level" int
