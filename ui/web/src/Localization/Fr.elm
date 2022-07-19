module Localization.Fr exposing (frLocalization, frTranslations)

import Localization.Base exposing (..)
import Util.DateFormat exposing (..)


frLocalization : Localization
frLocalization =
    { t = frTranslations
    , dateConfig = deDateConfig
    }


frTranslations : Translations
frTranslations =
    { search =
        { search = "Rechercher"
        , start = "Départ"
        , destination = "Destination"
        , date = "Date"
        , time = "Heure"
        , startTransports = "Transports au départ"
        , destinationTransports = "Transports à destination"
        , departure = "Départ"
        , arrival = "Arrivée"
        , trainNr = "Numéro de Train"
        , maxDuration = "Durée maximale (minutes)"
        , searchProfile =
            { profile = "Profil"
            , walkingSpeed = "Vitesse de marche (m/s)"
            , stairsUp = "Escaliers (montants)"
            , stairsDown = "Escaliers (descendants)"
            , stairsWithHandrailUp = "Escaliers montants avec rambarde"
            , stairsWithHandrailDown = "Escaliers descendants avec rambarde"
            , timeCost = "Coût en temps"
            , accessibilityCost = "Coût d'accessibilité"
            , streetCrossings = "Traversées de rues"
            , signals = "Feux tricolores"
            , marked = "Marked (zebra crossings)"
            , island = "Ilôts de trafic"
            , unmarked = "Unmarked"
            , primary = "Primaire"
            , secondary = "Secondaire"
            , tertiary = "Tertiaire"
            , residential = "Résidentiel"
            , elevationUp = "Dénivelé positif"
            , elevationDown = "Dénivelé négatif"
            , roundAccessibility = "Round accessibility"
            , elevators = "Ascenseurs"
            , escalators = "Escaliers mécaniques"
            , movingWalkways = "Trottoirs roulants"
            }
        , useParking = "Utiliser les parkings"
        }
    , connections =
        { timeHeader = "Temps"
        , durationHeader = "Durée"
        , transportsHeader = "Transports"
        , scheduleRange =
            \begin end ->
                "Possible dates: "
                    ++ formatDate deDateConfig begin
                    ++ " – "
                    ++ formatDate deDateConfig end
        , loading = "Recherche en cours..."
        , noResults = "Aucun trajet trouvé"
        , extendBefore = "Plus tôt"
        , extendAfter = "Plus tard"
        , interchanges =
            \count ->
                case count of
                    0 ->
                        "Aucune correspondance"

                    1 ->
                        "1 correspondance"

                    _ ->
                        toString count ++ " correspondances"
        , walkDuration = \duration -> duration ++ " de marche"
        , interchangeDuration = \duration -> duration ++ " de correspondance"
        , arrivalTrack = \track -> "Voie d'arrivée " ++ track
        , track = "Voie"
        , tripIntermediateStops =
            \count ->
                case count of
                    0 ->
                        "Aucun arrêt intermédiaire"

                    1 ->
                        "1 arrêt intermédiaire"

                    _ ->
                        toString count ++ " arrêts intermédiaires"
        , tripWalk = \duration -> "Marche (" ++ duration ++ ")"
        , tripBike = \duration -> "Vélo (" ++ duration ++ ")"
        , tripCar = \duration -> "Voiture (" ++ duration ++ ")"
        , provider = "Fournisseur"
        , walk = "Marche"
        , bike = "Vélo"
        , car = "Voiture"
        , trainNr = "Numéro de train"
        , lineId = "Ligne"
        , parking = "Parking"
        }
    , station =
        { direction = "Direction"
        , noDepartures = "Aucun départ"
        , noArrivals = "Aucune arrivée"
        , loading = "Chargement..."
        , trackAbbr = "V."
        }
    , railViz =
        { noTrains = "Aucun train"
        , delayColors = "Par délai"
        , classColors = "Par catégorie"
        , simActive = "Mode simulation activé"
        }
    , mapContextMenu =
        { routeFromHere = "Itinéraire depuis ici"
        , routeToHere = "Itinéraire jusqu'ici"
        }
    , errors =
        { journeyDateNotInSchedule = "La date indiquée n'est pas dans le planning"
        , internalError = \msg -> "Erreur interne (" ++ msg ++ ")"
        , timeout = "Expiration du délai"
        , network = "Erreur réseau"
        , http = \code -> "Erreur HTTP " ++ toString code
        , decode = \msg -> "Réponse invalide (" ++ msg ++ ")"
        , moduleNotFound = "Module introuvable"
        , osrmProfileNotAvailable = "OSRM: Profil non disponible"
        , osrmNoRoutingResponse = "OSRM: Aucun itinéraire trouvé"
        , pprProfileNotAvailable = "PPR: Profil non disponible"
        }
    , trips =
        { noResults = "Aucun train correspondant trouvé"
        }
    , misc =
        { permalink = "Permalink" }
    , simTime =
        { simMode = "Mode Simulation"
        }
    , searchProfiles =
        { default = "Par défaut"
        , accessibility1 = "Inclure les routes accessibles"
        , wheelchair = "Fauteuil roulant"
        , elevation = "Eviter les dénivelés"
        , custom = "Personnalisé"
        }
    }
