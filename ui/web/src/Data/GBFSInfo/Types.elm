module Data.GBFSInfo.Types exposing (GBFSInfo, GBFSProvider)


type alias GBFSProvider =
    { name : String
    , vehicle_type : String
    , tag : String
    }


type alias GBFSInfo =
    { providers : List GBFSProvider
    }
