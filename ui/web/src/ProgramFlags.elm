module ProgramFlags exposing (ProgramFlags)

import Time exposing (Posix)


type alias ProgramFlags =
    { apiEndpoint : String
    , currentTime : Posix
    , simulationTime : Maybe Posix
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
