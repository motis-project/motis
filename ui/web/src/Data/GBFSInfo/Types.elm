module Data.GBFSInfo.Types exposing (GBFSInfo, GBFSProvider)

type alias GBFSProvider =
    { vehicle_type : String
    , name : String
    }

type alias GBFSInfo =
    { providers : List GBFSProvider
    }
