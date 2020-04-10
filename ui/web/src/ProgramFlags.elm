module ProgramFlags exposing (ProgramFlags)

import Time exposing (Time)


type alias ProgramFlags =
    { apiEndpoint : String
    , currentTime : Time
    , simulationTime : Maybe Time
    , language : String
    , motisParam : Maybe String
    , timeParam : Maybe String
    , langParam : Maybe String
    , fromLocation : Maybe String
    , toLocation : Maybe String
    , fromModes : Maybe String
    , toModes : Maybe String
    , intermodalPprMode : Maybe String
    }
