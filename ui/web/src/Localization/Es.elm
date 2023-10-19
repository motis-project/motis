module Localization.De exposing (esLocalization, esTranslations)

import Localization.Base exposing (..)
import Util.DateFormat exposing (..)


esLocalization : Localization
esLocalization =
    { t = esTranslations
    , dateConfig = esDateConfig
    }


esTranslations : Translations
esTranslations =
    { search =
        { search = "Buscar"
        , start = "Origen"
        , destination = "Destino"
        , date = "Fecha"
        , time = "Hora"
        , startTransports = "Medios de transporte en el origen"
        , destinationTransports = "Medios de transporte en el destino"
        , departure = "Salida"
        , arrival = "Llegada"
        , trainNr = "Código tren"
        , maxDuration = "Duración máxima (minutos)"
        , searchProfile =
            { profile = "Perfil"
            , walkingSpeed = "Velocidad a pie (m/s)"
            , stairsUp = "Escaleras (hacia arriba)"
            , stairsDown = "Escaleras (hacia abajo)"
            , stairsWithHandrailUp = "Escaleras con barandilla (hacia arriba)"
            , stairsWithHandrailDown = "Escaleras con barandilla (hacia abajo)"
            , timeCost = "Coste de tiempo"
            , accessibilityCost = "Coste de accesibilidad"
            , streetCrossings = "Cruces"
            , signals = "Señales de tráfico"
            , marked = "Paso de cebra"
            , island = "Isleta de tráfico"
            , unmarked = "Sin marcar"
            , primary = "Primario"
            , secondary = "Secundario"
            , tertiary = "Terciario"
            , residential = "Residencial"
            , elevationUp = "Diferencia de elevación (hacia arriba)"
            , elevationDown = "Diferencia de elevación (hacia abajo)"
            , roundAccessibility = "Redondeo de accesibilidad"
            , elevators = "Ascensores"
            , escalators = "Escaleras mecánicas"
            , movingWalkways = "Cinta transportadora"
            }
        , useParking = "Incluir parkings"
        }
    , connections =
        { timeHeader = "Hora"
        , durationHeader = "Duración"
        , transportsHeader = "Medios de transporte"
        , scheduleRange =
            \begin end ->
                "Fechas posibles: "
                    ++ formatDate deDateConfig begin
                    ++ " a "
                    ++ formatDate deDateConfig end
        , loading = "Buscando conexiones..."
        , noResults = "No se han encontrado conexiones"
        , extendBefore = "Más temprano"
        , extendAfter = "Más tarde"
        , interchanges =
            \count ->
                case count of
                    0 ->
                        "Sin trasbordos"

                    1 ->
                        "Un trasbordo"

                    _ ->
                        toString count ++ " trasbordos"
        , walkDuration = \duration -> duration ++ " andando"
        , interchangeDuration = \duration -> "Trasbordo de " ++ duration
        , arrivalTrack = \track -> "Andén de llegada: " ++ track
        , track = "Andén"
        , tripIntermediateStops =
            \count ->
                case count of
                    0 ->
                        "Sin paradas intermedias"

                    1 ->
                        "Una parada intermedia"

                    _ ->
                        toString count ++ " paradas intermedias"
        , tripWalk = \duration -> "A pie (" ++ duration ++ ")"
        , tripBike = \duration -> "Bicicleta (" ++ duration ++ ")"
        , tripCar = \duration -> "Coche (" ++ duration ++ ")"
        , provider = "Operador"
        , walk = "A pie"
        , bike = "Bicicleta"
        , car = "Coche"
        , trainNr = "Código tren"
        , lineId = "Linea"
        , parking = "Parking"
        }
    , station =
        { direction = "Dirección"
        , noDepartures = "No hay salidas en el intervalo seleccionado"
        , noArrivals = "No hay llegadas en el intervalo seleccionado"
        , loading = "Cargando..."
        , trackAbbr = "Andén"
        }
    , railViz =
        { noTrains = "Sin trenes"
        , delayColors = "Por retraso"
        , classColors = "Por categorías"
        , simActive = "Modo simulación activado"
        }
    , mapContextMenu =
        { routeFromHere = "Rutas desde aquí"
        , routeToHere = "Rutas hacia aquí"
        }
    , errors =
        { journeyDateNotInSchedule = "La fecha no está en el horario"
        , internalError = \msg -> "Error interno (" ++ msg ++ ")"
        , timeout = "Tiempo excedido"
        , network = "Error de red"
        , http = \code -> "Error HTTP " ++ toString code
        , decode = \msg -> "Respuesta no válida (" ++ msg ++ ")"
        , moduleNotFound = "Módulo no encontrado"
        , osrmProfileNotAvailable = "OSRM: Perfil no disponible"
        , osrmNoRoutingResponse = "OSRM: Sin respuesta del enrutador"
        , pprProfileNotAvailable = "PPR: Perfil no disponible"
        }
    , trips =
        { noResults = "No se han encontrado trenes"
        }
    , misc =
        { permalink = "Permalink" }
    , simTime =
        { simMode = "Modo simulación"
        }
    , searchProfiles =
        { default = "Por defecto"
        , accessibility1 = "Incluir rutas accesibles"
        , wheelchair = "Silla de ruedas"
        , elevation = "Evitar cambios de elevación"
        , custom = "Personalizado"
        }
    }
