module Localization.Base exposing (Localization, SearchProfileNames, Translations)

import Date exposing (Date)
import Util.DateFormat exposing (DateConfig)


type alias Localization =
    { t : Translations
    , dateConfig : DateConfig
    }


type alias SearchProfileNames =
    { default : String
    , accessibility1 : String
    , wheelchair : String
    , elevation : String
    , custom : String
    }


type alias Translations =
    { search :
        { search : String
        , start : String
        , destination : String
        , date : String
        , time : String
        , startTransports : String
        , destinationTransports : String
        , departure : String
        , arrival : String
        , trainNr : String
        , maxDuration : String
        , searchProfile :
            { profile : String
            , walkingSpeed : String
            , stairsUp : String
            , stairsDown : String
            , stairsWithHandrailUp : String
            , stairsWithHandrailDown : String
            , timeCost : String
            , accessibilityCost : String
            , streetCrossings : String
            , signals : String
            , marked : String
            , island : String
            , unmarked : String
            , primary : String
            , secondary : String
            , tertiary : String
            , residential : String
            , elevationUp : String
            , elevationDown : String
            , roundAccessibility : String
            , elevators : String
            , escalators : String
            , movingWalkways : String
            }
        , useParking : String
        }
    , connections :
        { timeHeader : String
        , durationHeader : String
        , transportsHeader : String
        , scheduleRange : Date -> Date -> String
        , loading : String
        , noResults : String
        , extendBefore : String
        , extendAfter : String
        , interchanges : Int -> String
        , walkDuration : String -> String
        , interchangeDuration : String -> String
        , arrivalTrack : String -> String
        , track : String
        , tripIntermediateStops : Int -> String
        , tripWalk : String -> String
        , tripBike : String -> String
        , tripCar : String -> String
        , provider : String
        , walk : String
        , bike : String
        , car : String
        , trainNr : String
        , lineId : String
        , parking : String
        }
    , station :
        { direction : String
        , noDepartures : String
        , noArrivals : String
        , loading : String
        , trackAbbr : String
        }
    , railViz :
        { noTrains : String
        , delayColors : String
        , classColors : String
        , simActive : String
        }
    , mapContextMenu :
        { routeFromHere : String
        , routeToHere : String
        }
    , errors :
        { journeyDateNotInSchedule : String
        , internalError : String -> String
        , timeout : String
        , network : String
        , http : Int -> String
        , decode : String -> String
        , moduleNotFound : String
        , osrmProfileNotAvailable : String
        , osrmNoRoutingResponse : String
        , pprProfileNotAvailable : String
        }
    , trips :
        { noResults : String
        }
    , misc :
        { permalink : String }
    , simTime :
        { simMode : String
        }
    , searchProfiles :
        SearchProfileNames
    }
