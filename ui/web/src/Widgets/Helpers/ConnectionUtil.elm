module Widgets.Helpers.ConnectionUtil exposing
    ( TransportViewMode(..)
    , delay
    , delayText
    , isDelayed
    , longTransportName
    , longTransportNameWithoutIcon
    , shortTransportName
    , trainBox
    , trainIcon
    , useLineId
    , walkBox
    , zeroDelay
    )

import Data.Connection.Types as Connection exposing (..)
import Date.Extra.Duration as Duration exposing (DeltaRecord)
import Html exposing (Html, div, span, text)
import Html.Attributes exposing (..)
import Localization.Base exposing (..)
import String
import Svg
import Svg.Attributes exposing (xlinkHref)
import Util.Core exposing ((=>))


useLineId : Int -> Bool
useLineId class =
    class == 5 || class == 6


longTransportName : TransportInfo -> String
longTransportName transport =
    if useLineId transport.class then
        transport.line_id

    else
        transport.name


shortTransportName : TransportInfo -> String
shortTransportName transport =
    let
        train_nr =
            Maybe.withDefault 0 transport.train_nr
    in
    if useLineId transport.class then
        transport.line_id

    else if String.length transport.name < 7 then
        transport.name

    else if String.isEmpty transport.line_id && train_nr == 0 then
        transport.name

    else if String.isEmpty transport.line_id then
        toString train_nr

    else
        transport.line_id


longTransportNameWithoutIcon : TransportInfo -> String
longTransportNameWithoutIcon transport =
    transport.name


trainIcon : Int -> String
trainIcon class =
    case class of
        0 ->
            "plane"

        1 ->
            "train"

        2 ->
            "train"

        3 ->
            "bus"

        4 ->
            "train"

        5 ->
            "train"

        6 ->
            "train"

        7 ->
            "sbahn"

        8 ->
            "ubahn"

        9 ->
            "tram"

        11 ->
            "ship"

        _ ->
            "bus"


type TransportViewMode
    = LongName
    | ShortName
    | IconOnly
    | IconOnlyNoSep


trainBox : TransportViewMode -> Localization -> TransportInfo -> Html msg
trainBox viewMode locale t =
    let
        icon =
            Svg.svg
                [ Svg.Attributes.class "train-icon" ]
                [ Svg.use
                    [ xlinkHref <| "#" ++ trainIcon t.class ]
                    []
                ]

        name =
            case viewMode of
                LongName ->
                    longTransportName t

                ShortName ->
                    shortTransportName t

                IconOnly ->
                    ""

                IconOnlyNoSep ->
                    ""

        providerTooltip =
            if not (String.isEmpty t.provider) then
                Just (locale.t.connections.provider ++ ": " ++ t.provider)

            else
                Nothing

        trainNr =
            Maybe.withDefault 0 t.train_nr

        trainNrTooltip =
            if trainNr /= 0 && not (String.contains (toString trainNr) name) then
                Just (locale.t.connections.trainNr ++ ": " ++ toString trainNr)

            else
                Nothing

        lineIdTooltip =
            if not (String.isEmpty t.line_id) && not (String.contains t.line_id name) then
                Just (locale.t.connections.lineId ++ ": " ++ t.line_id)

            else
                Nothing

        tooltipText =
            List.filterMap identity [ providerTooltip, trainNrTooltip, lineIdTooltip ]
                |> String.join "\n"
    in
    div
        [ classList
            [ "train-box" => True
            , ("train-class-" ++ toString t.class) => True
            , "with-tooltip" => not (String.isEmpty tooltipText)
            ]
        , attribute "data-tooltip" tooltipText
        ]
        [ icon
        , if String.isEmpty name then
            text ""

          else
            span [ class "train-name" ] [ text name ]
        ]


walkBox : String -> Html msg
walkBox mumoType =
    let
        icon =
            case mumoType of
                "walk" ->
                    "#walk"

                "bike" ->
                    "#bike"

                "car" ->
                    "#car"

                _ ->
                    "#walk"
    in
    div [ class <| "train-box train-class-walk" ]
        [ Svg.svg
            [ Svg.Attributes.class "train-icon" ]
            [ Svg.use
                [ xlinkHref <| icon ]
                []
            ]
        ]


isDelayed : DeltaRecord -> Bool
isDelayed dr =
    dr.minute > 0 || dr.hour > 0 || dr.day > 0


zeroDelay : DeltaRecord -> Bool
zeroDelay dr =
    dr.minute == 0 && dr.hour == 0 && dr.day == 0


delayText : DeltaRecord -> String
delayText dr =
    let
        str =
            abs >> toString
    in
    str (dr.minute + (dr.hour * 60) + (dr.day * 24 * 60))


delay : EventInfo -> Html msg
delay event =
    let
        diff =
            Maybe.map2 Duration.diff event.time event.schedule_time
    in
    case event.reason of
        Schedule ->
            text ""

        _ ->
            case diff of
                Just d ->
                    let
                        delayed =
                            isDelayed d
                    in
                    div
                        [ class <|
                            if delayed then
                                "delay pos-delay"

                            else
                                "delay neg-delay"
                        ]
                        [ span []
                            [ text <|
                                (if delayed || zeroDelay d then
                                    "+"

                                 else
                                    "-"
                                )
                                    ++ delayText d
                            ]
                        ]

                Nothing ->
                    text ""
