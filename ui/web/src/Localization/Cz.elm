module Localization.Cz exposing (czLocalization, czTranslations)

import Localization.Base exposing (..)
import Util.DateFormat exposing (..)


czLocalization : Localization
czLocalization =
    { t = czTranslations
    , dateConfig = czDateConfig
    }


czTranslations : Translations
czTranslations =
    { search =
        { search = "Hledat"
        , start = "Start"
        , destination = "Cíl"
        , date = "Datum"
        , time = "Čas"
        , startTransports = "Doprava na startu"
        , destinationTransports = "Doprava v cíli"
        , departure = "Odjezd"
        , arrival = "Příjezd"
        , trainNr = "Číslo vlaku"
        , maxDuration = "Max. délka (minuty)"
        , searchProfile =
            { profile = "Profil"
            , walkingSpeed = "Rychlost chůze (m/s)"
            , stairsUp = "Schody (nahoru)"
            , stairsDown = "Schody (dolů)"
            , stairsWithHandrailUp = "Schody se zábradlím (nahoru)"
            , stairsWithHandrailDown = "Schody se zábradlím (dolů)"
            , timeCost = "Časová náročnost"
            , accessibilityCost = "Přístupnost"
            , streetCrossings = "Přechody pro chodce"
            , signals = "Semafory"
            , marked = "Přechody pro chodce"
            , island = "Osrůvky"
            , unmarked = "Neoznačené"
            , primary = "Primární"
            , secondary = "Sekundární"
            , tertiary = "Terciální"
            , residential = "Rezidenční"
            , elevationUp = "Výškový rozdíl (nahoru)"
            , elevationDown = "Výškový rozdíl (dolů)"
            , roundAccessibility = "Round accessibility"
            , elevators = "Výtahy"
            , escalators = "Eskalátory"
            , movingWalkways = "Pohyblivé chodníky"
            }
        , useParking = "Použít parkoviště"
        }
    , connections =
        { timeHeader = "Čas"
        , durationHeader = "Jízdní doba"
        , transportsHeader = "Transports"
        , scheduleRange =
            \begin end ->
                "Možné rozmezí dat: "
                    ++ formatDate enDateConfig begin
                    ++ " – "
                    ++ formatDate enDateConfig end
        , loading = "Hledám..."
        , noResults = "Nebyla nalezena žádná spojení"
        , extendBefore = "Dřívější"
        , extendAfter = "Pozdější"
        , interchanges =
            \count ->
                case count of
                    0 ->
                        "Bez přestupů"

                    1  ->
                        "1 přestup"

                    2 ->
                        "2 přestupy"

                    3 ->
                        "3 přestupy"

                    4 ->
                        "4 přestupy"

                    _ ->
                        toString count ++ " přestupů"
        , walkDuration = \duration -> duration ++ " chůže"
        , interchangeDuration = \duration -> duration ++ " přestup"
        , arrivalTrack = \track -> "Arrival track " ++ track
        , track = "Track"
        , tripIntermediateStops =
            \count ->
                case count of
                    0 ->
                        "Bez mezizastávek"

                    1 ->
                        "1 mezizastávka"

                    2 ->
                        "2 mezizastávky"

                    3 ->
                        "3 mezizastávky"

                    4 ->
                        "4 mezizastávky"

                    _ ->
                        toString count ++ " mezizastávek"
        , tripWalk = \duration -> "Chůze (" ++ duration ++ ")"
        , tripBike = \duration -> "Jízdní kolo (" ++ duration ++ ")"
        , tripCar = \duration -> "Car (" ++ duration ++ ")"
        , provider = "Dopravce"
        , walk = "Walk"
        , bike = "Jízdní kolo"
        , car = "Car"
        , trainNr = "Číslo vlaku"
        , lineId = "Linka"
        , parking = "Parkování"
        }
    , station =
        { direction = "Směr"
        , noDepartures = "Žádné odjezdy"
        , noArrivals = "Žádné příjezdy"
        , loading = "Načítám..."
        , trackAbbr = "Nást."
        }
    , railViz =
        { noTrains = "Žádné vlaky"
        , delayColors = "Podle zpoždení"
        , classColors = "Podle kategorie"
        , simActive = "Mód simulace aktivní"
        }
    , mapContextMenu =
        { routeFromHere = "Trasa odtud"
        , routeToHere = "Trasa sem"
        }
    , errors =
        { journeyDateNotInSchedule = "Datum není v jízdních řádech"
        , internalError = \msg -> "Interní chyba (" ++ msg ++ ")"
        , timeout = "Časový limit odpovědi vypršel"
        , network = "Chyba sítě"
        , http = \code -> "HTTP chyba " ++ toString code
        , decode = \msg -> "Neplatná odpověď (" ++ msg ++ ")"
        , moduleNotFound = "Modul nenalezen"
        , osrmProfileNotAvailable = "OSRM: Profil není dostupný"
        , osrmNoRoutingResponse = "OSRM: Žádná odpověď trasování"
        , pprProfileNotAvailable = "PPR: Profil není dostupný"
        }
    , trips =
        { noResults = "Nebyl nalezen žádný odpovídající vlak"
        }
    , misc =
        { permalink = "Trvalý odkaz" }
    , simTime =
        { simMode = "Mód simulace"
        }
    , searchProfiles =
        { default = "Výchozí"
        , accessibility1 = "Zahrnout přístupné trasy"
        , wheelchair = "Invalidní vozík"
        , elevation = "Vyhněte se výškovým změnám"
        , custom = "Vlastní"
        }
    }
