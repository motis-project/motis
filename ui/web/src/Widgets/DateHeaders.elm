module Widgets.DateHeaders exposing (dateHeader, withDateHeaders)

import Date exposing (Date)
import Html exposing (Html, div, span, text)
import Html.Attributes exposing (..)
import Localization.Base exposing (..)
import Util.Date exposing (isSameDay)
import Util.DateFormat exposing (..)


dateHeader : Localization -> Date -> Html msg
dateHeader { dateConfig } date =
    div [ class "date-header divider" ] [ span [] [ text <| formatDate dateConfig date ] ]



-- b = Html msg
--   | (String, Html msg) for keyed nodes


withDateHeaders :
    (a -> Date)
    -> (a -> b)
    -> (Date -> b)
    -> List a
    -> List b
withDateHeaders getDate renderElement renderDateHeader elements =
    let
        f element ( lastDate, result ) =
            let
                currentDate =
                    getDate element

                base =
                    if not (isSameDay currentDate lastDate) then
                        result ++ [ renderDateHeader currentDate ]

                    else
                        result
            in
            ( currentDate, base ++ [ renderElement element ] )

        ( _, result ) =
            List.foldl f ( Date.fromTime 0, [] ) elements
    in
    result
