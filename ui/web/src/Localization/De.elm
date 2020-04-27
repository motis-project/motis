module Localization.De exposing (deLocalization, deTranslations)

import Localization.Base exposing (..)
import Util.DateFormat exposing (..)


deLocalization : Localization
deLocalization =
    { t = deTranslations
    , dateConfig = deDateConfig
    }


deTranslations : Translations
deTranslations =
    { search =
        { search = "Suchen"
        , start = "Start"
        , destination = "Ziel"
        , date = "Datum"
        , time = "Uhrzeit"
        , startTransports = "Verkehrsmittel am Start"
        , destinationTransports = "Verkehrsmittel am Ziel"
        , departure = "Abfahrt"
        , arrival = "Ankunft"
        , trainNr = "Zugnummer"
        , maxDuration = "Maximale Dauer (Minuten)"
        , searchProfile =
            { profile = "Profil"
            , walkingSpeed = "Laufgeschwindigkeit (m/s)"
            , stairsUp = "Treppen (aufwärts)"
            , stairsDown = "Treppen (abwärts)"
            , stairsWithHandrailUp = "Treppen mit Geländer (aufwärts)"
            , stairsWithHandrailDown = "Treppen mit Geländer (abwärts)"
            , timeCost = "Zeitaufwand"
            , accessibilityCost = "Beschwerlichkeit"
            , streetCrossings = "Straßenüberquerungen"
            , signals = "Ampeln"
            , marked = "Zebrastreifen"
            , island = "Verkehrsinseln"
            , unmarked = "Unmarkiert"
            , primary = "Primary"
            , secondary = "Secondary"
            , tertiary = "Tertiary"
            , residential = "Residential"
            , elevationUp = "Höhenunterschiede (aufwärts)"
            , elevationDown = "Höhenunterschiede (abwärts)"
            , roundAccessibility = "Rundung Beweglichkeit"
            , elevators = "Aufzüge"
            , escalators = "Rolltreppen"
            , movingWalkways = "Fahrsteige"
            }
        , useParking = "Parkplätze verwenden"
        }
    , connections =
        { timeHeader = "Zeit"
        , durationHeader = "Dauer"
        , transportsHeader = "Verkehrsmittel"
        , scheduleRange =
            \begin end ->
                "Auskunft von "
                    ++ formatDate deDateConfig begin
                    ++ " bis "
                    ++ formatDate deDateConfig end
                    ++ " möglich"
        , loading = "Verbindungen suchen..."
        , noResults = "Keine Verbindungen gefunden"
        , extendBefore = "Früher"
        , extendAfter = "Später"
        , interchanges =
            \count ->
                case count of
                    0 ->
                        "Keine Umstiege"

                    1 ->
                        "1 Umstieg"

                    _ ->
                        toString count ++ " Umstiege"
        , walkDuration = \duration -> duration ++ " Fußweg"
        , interchangeDuration = \duration -> duration ++ " Umstieg"
        , arrivalTrack = \track -> "Ankunft Gleis " ++ track
        , track = "Gleis"
        , tripIntermediateStops =
            \count ->
                case count of
                    0 ->
                        "Fahrt ohne Zwischenhalt"

                    1 ->
                        "Fahrt 1 Station"

                    _ ->
                        "Fahrt " ++ toString count ++ " Stationen"
        , tripWalk = \duration -> "Fußweg (" ++ duration ++ ")"
        , tripBike = \duration -> "Fahrrad (" ++ duration ++ ")"
        , tripCar = \duration -> "Auto (" ++ duration ++ ")"
        , provider = "Betreiber"
        , walk = "Fußweg"
        , bike = "Fahrrad"
        , car = "Auto"
        , trainNr = "Zugnummer"
        , lineId = "Linie"
        , parking = "Parkplatz"
        }
    , station =
        { direction = "Richtung"
        , noDepartures = "Keine Abfahrten im gewählten Zeitraum"
        , noArrivals = "Keine Ankünfte im gewählten Zeitraum"
        , loading = "Laden..."
        , trackAbbr = "Gl."
        }
    , railViz =
        { noTrains = "Keine Züge"
        , delayColors = "Nach Verspätung"
        , classColors = "Nach Kategorie"
        , simActive = "Simulationsmodus aktiv"
        }
    , mapContextMenu =
        { routeFromHere = "Routen von hier"
        , routeToHere = "Routen hierher"
        }
    , errors =
        { journeyDateNotInSchedule = "Zeitraum nicht im Fahrplan"
        , internalError = \msg -> "Interner Fehler (" ++ msg ++ ")"
        , timeout = "Zeitüberschreitung"
        , network = "Netzwerkfehler"
        , http = \code -> "HTTP-Fehler " ++ toString code
        , decode = \msg -> "Ungültige Antwort (" ++ msg ++ ")"
        , moduleNotFound = "Modul nicht geladen"
        , osrmProfileNotAvailable = "OSRM: Profil nicht verfügbar"
        , osrmNoRoutingResponse = "OSRM: Keine Routing-Antwort"
        , pprProfileNotAvailable = "PPR: Profil nicht verfügbar"
        }
    , trips =
        { noResults = "Keine passenden Züge gefunden"
        }
    , misc =
        { permalink = "Permalink" }
    , simTime =
        { simMode = "Simulationsmodus"
        }
    , searchProfiles =
        { default = "Standard"
        , accessibility1 = "Auch nach leichten Wegen suchen"
        , wheelchair = "Rollstuhl"
        , elevation = "Weniger Steigung"
        , custom = "Benutzerdefiniert"
        }
    }
