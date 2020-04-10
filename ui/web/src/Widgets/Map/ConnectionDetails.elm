module Widgets.Map.ConnectionDetails exposing
    ( setConnectionFilter
    , updateWalks
    )

import Data.Journey.Types exposing (..)
import List.Extra
import Maybe.Extra exposing (maybeToList)
import Util.List exposing ((!!), dropEnd, last)
import Widgets.Map.Port exposing (..)
import Widgets.Map.RailViz exposing (buildRVTrain, buildRVWalk, mapId)


setConnectionFilter : Journey -> Cmd msg
setConnectionFilter journey =
    let
        ( connectionFilter, bounds ) =
            buildConnectionFilter journey
    in
    Cmd.batch
        [ mapSetConnectionFilter connectionFilter
        , mapFitBounds bounds
        ]


updateWalks : Journey -> Cmd msg
updateWalks journey =
    journey.walks
        |> List.map buildRVWalk
        |> mapUpdateWalks


buildConnectionFilter : Journey -> ( RVConnectionFilter, MapFitBounds )
buildConnectionFilter journey =
    let
        interchangeStops =
            List.concatMap
                (\train ->
                    maybeToList (List.head train.stops) ++ maybeToList (last train.stops)
                )
                journey.trains

        walkStops =
            List.concatMap (\walk -> [ walk.from, walk.to ]) journey.walks

        interchangeStations =
            List.map (.station >> .id) (interchangeStops ++ walkStops)
                |> List.Extra.unique

        intermediateStops =
            List.concatMap (\train -> dropEnd 1 (List.drop 1 train.stops)) journey.trains

        intermediateStations =
            List.map (.station >> .id) intermediateStops
                |> List.Extra.unique
                |> List.filter (\station -> not (List.member station interchangeStations))

        bounds =
            (interchangeStops ++ intermediateStops ++ walkStops)
                |> List.map (.station >> .pos)
                |> List.map (\pos -> [ pos.lat, pos.lng ])
                |> List.Extra.unique
    in
    ( { trains = List.map buildRVTrain journey.trains
      , walks = List.map buildRVWalk journey.walks
      , interchangeStations = interchangeStations
      , intermediateStations = intermediateStations
      }
    , { mapId = mapId
      , coords = bounds
      }
    )
