module Util.View exposing (onStopAll, onStopPropagation)

import Html exposing (..)
import Html.Events exposing (custom)
import Json.Decode


onStopAll : String -> msg -> Html.Attribute msg
onStopAll event msg =
    custom event (Json.Decode.succeed { message = msg, stopPropagation = True, preventDefault = True })


onStopPropagation : String -> msg -> Html.Attribute msg
onStopPropagation event msg =
    custom event (Json.Decode.succeed { message = msg, stopPropagation = True, preventDefault = False })
