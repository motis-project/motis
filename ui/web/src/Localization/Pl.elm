module Localization.Pl exposing (plLocalization, plTranslations)

import Localization.Base exposing (..)
import Util.DateFormat exposing (..)


plLocalization : Localization
plLocalization =
    { t = plTranslations
    , dateConfig = deDateConfig
    }


plTranslations : Translations
plTranslations =
    { search =
        { search = "Szukaj"
        , start = "Początek"
        , destination = "Koniec"
        , date = "Data"
        , time = "Godzina"
        , startTransports = "Środki transportu na początku"
        , destinationTransports = "Środki transportu na końcu"
        , departure = "Odjazd"
        , arrival = "Przyjazd"
        , trainNr = "Numer kursu"
        , maxDuration = "Max. czas trwania (w min.)"
        , searchProfile =
            { profile = "Profil"
            , walkingSpeed = "Prędkość marszu (m/s)"
            , stairsUp = "Schody (w górę)"
            , stairsDown = "Schody (w dół)"
            , stairsWithHandrailUp = "Schody z poręczą (w górę)"
            , stairsWithHandrailDown = "Schody z poręczą (w gół)"
            , timeCost = "Nakład czasu"
            , accessibilityCost = "uciążliwość"
            , streetCrossings = "przejście przez ulicę"
            , signals = "lampy"
            , marked = "Przejście dla pieszych"
            , island = "Wysepka"
            , unmarked = "Nieoznaczone"
            , primary = "Primary"
            , secondary = "Secondary"
            , tertiary = "Tertiary"
            , residential = "Residential"
            , elevationUp = "Różnica wysokości (w górę)"
            , elevationDown = "Różnice wysokości (w dół)"
            , roundAccessibility = "Zdolność do zaokrąglania"
            , elevators = "Windy"
            , escalators = "Schody ruchome"
            , movingWalkways = "Chodniki ruchome"
            }
        , useParking = "Korzystaj z parkingów"
        }
    , connections =
        { timeHeader = "Czas"
        , durationHeader = "Czas trwania"
        , transportsHeader = "środki transportu"
        , scheduleRange =
            \begin end ->
                "Informacja od "
                    ++ formatDate deDateConfig begin
                    ++ " do "
                    ++ formatDate deDateConfig end
                    ++ " możliwa"
        , loading = "Szukam połączeń..."
        , noResults = "Nie znaleziono połączeń"
        , extendBefore = "Wcześniej"
        , extendAfter = "Później"
        , interchanges =
            \count ->
                case count of
                    0 ->
                        "Bez przesiadek"

                    1 ->
                        "1 przesiadka"

                    _ ->
                        toString count ++ " przesiadek"
        , walkDuration = \duration -> duration ++ " Spacer"
        , interchangeDuration = \duration -> duration ++ " przesiadka"
        , arrivalTrack = \track -> "peron przyjazdu " ++ track
        , track = "Peron"
        , tripIntermediateStops =
            \count ->
                case count of
                    0 ->
                        "Podróż bez zatrzymywania się"

                    1 ->
                        "Podróż 1 przystanek"

                    _ ->
                        "Podróż " ++ toString count ++ " przystanków"
        , tripWalk = \duration -> "Spacer (" ++ duration ++ ")"
        , tripBike = \duration -> "Rower (" ++ duration ++ ")"
        , tripCar = \duration -> "Samochód (" ++ duration ++ ")"
        , provider = "Operator"
        , walk = "Spacer"
        , bike = "Rower"
        , car = "Samochód"
        , trainNr = "Numer kursu"
        , lineId = "Linia"
        , parking = "Parking"
        }
    , station =
        { direction = "Kierunek"
        , noDepartures = "Brak odjazdów w wybranym okresie"
        , noArrivals = "Brak przyjazdów w wybranym okresie"
        , loading = "Ładuję..."
        , trackAbbr = "Per."
        }
    , railViz =
        { noTrains = "Brak kursów"
        , delayColors = "Według opóźnień"
        , classColors = "Według kategorii"
        , simActive = "Tryb symulacji aktywny"
        }
    , mapContextMenu =
        { routeFromHere = "Początek podróży tutaj"
        , routeToHere = "Koniec podróży tutaj"
        }
    , errors =
        { journeyDateNotInSchedule = "Okres poza harmonogramem"
        , internalError = \msg -> "Błąd wewnętrzny (" ++ msg ++ ")"
        , timeout = "Przekroczono limit czasu"
        , network = "Błąd sieci"
        , http = \code -> "Błąd HTTP " ++ toString code
        , decode = \msg -> "Nieprawidłowa odpowiedź (" ++ msg ++ ")"
        , moduleNotFound = "Moduł nie załadowany"
        , osrmProfileNotAvailable = "OSRM: Profil niedostępny"
        , osrmNoRoutingResponse = "OSRM: brak odpowiedzi nawigacji"
        , pprProfileNotAvailable = "PPR: Profil niedostępny"
        }
    , trips =
        { noResults = "Nie znaleziono pasujących kursów"
        }
    , misc =
        { permalink = "Permalink" }
    , simTime =
        { simMode = "Tryb symulacji"
        }
    , searchProfiles =
        { default = "Domyślny"
        , accessibility1 = "Poszukaj też łatwej trasy"
        , wheelchair = "wózek inwalidzki"
        , elevation = "Mniej wzniesień"
        , custom = "Użytkownika"
        }
    }
