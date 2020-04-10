module Data.Address.Request exposing (encodeAddress, encodeAddressRequest, encodeRegion)

import Data.Address.Types exposing (..)
import Data.Connection.Request exposing (encodePosition)
import Json.Encode as Encode
import Util.Core exposing ((=>))


encodeAddressRequest : String -> Encode.Value
encodeAddressRequest input =
    Encode.object
        [ "destination"
            => Encode.object
                [ "type" => Encode.string "Module"
                , "target" => Encode.string "/address"
                ]
        , "content_type" => Encode.string "AddressRequest"
        , "content"
            => Encode.object
                [ "input" => Encode.string input ]
        ]



-- for local storage


encodeAddress : Address -> Encode.Value
encodeAddress address =
    Encode.object
        [ "pos" => encodePosition address.pos
        , "name" => Encode.string address.name
        , "type" => Encode.string address.type_
        , "regions" => Encode.list (List.map encodeRegion address.regions)
        ]


encodeRegion : Region -> Encode.Value
encodeRegion region =
    Encode.object
        [ "name" => Encode.string region.name
        , "admin_level" => Encode.int region.adminLevel
        ]
