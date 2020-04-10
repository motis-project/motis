module Util.View exposing (onStopAll, onStopPropagation)

import Html exposing (..)
import Html.Events exposing (onWithOptions)
import Json.Decode as Json


onStopAll : String -> msg -> Html.Attribute msg
onStopAll event msg =
    onWithOptions event { stopPropagation = True, preventDefault = True } (Json.succeed msg)


onStopPropagation : String -> msg -> Html.Attribute msg
onStopPropagation event msg =
    onWithOptions event { stopPropagation = True, preventDefault = False } (Json.succeed msg)
