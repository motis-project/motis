module Data.Address.Types exposing (Address, AddressResponse, Region)

import Data.Connection.Types exposing (Position)


type alias AddressResponse =
    { guesses : List Address }


type alias Address =
    { pos : Position
    , name : String
    , type_ : String
    , regions : List Region
    }


type alias Region =
    { name : String
    , adminLevel : Int
    }
