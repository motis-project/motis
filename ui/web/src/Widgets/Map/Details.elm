module Widgets.Map.Details exposing
    ( setDetailFilter
    , updateWalks
    )

import Data.Journey.Types exposing (..)
import List.Extra
import Maybe.Extra exposing (maybeToList)
import Util.List exposing ((!!), dropEnd, last)
import Widgets.Map.Port exposing (..)
import Widgets.Map.RailViz exposing (buildRVTrain, buildRVWalk, mapId)


setDetailFilter : Maybe Journey -> Cmd msg
setDetailFilter journey =
    case journey of
        Just journey ->
            let
                ( detailFilter, bounds ) =
                    buildDetailFilter journey
            in
            Cmd.batch
                [ mapSetDetailFilter ( Just detailFilter ) 
                , mapFitBounds bounds
                ]

        Nothing ->
            mapSetDetailFilter Nothing


updateWalks : Journey -> Cmd msg
updateWalks journey =
    journey.walks
        |> List.map buildRVWalk
        |> mapUpdateWalks


buildDetailFilter : Journey -> ( RVDetailFilter, MapFitBounds )
buildDetailFilter journey =
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
            List.map .station (interchangeStops ++ walkStops)
                |> List.Extra.uniqueBy .id

        intermediateStops =
            List.concatMap (\train -> dropEnd 1 (List.drop 1 train.stops)) journey.trains

        bounds =
            (interchangeStops ++ intermediateStops ++ walkStops)
                |> List.map (.station >> .pos)
                |> List.map (\pos -> [ pos.lat, pos.lng ])
                |> List.Extra.unique
    in
    ( { trains = List.map buildRVTrain journey.trains
      , walks = List.map buildRVWalk journey.walks
      , interchangeStations = interchangeStations
      }
    , { mapId = mapId
      , coords = bounds
      }
    )
