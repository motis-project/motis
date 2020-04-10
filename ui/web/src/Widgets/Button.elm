module Widgets.Button exposing (view)

import Html exposing (Attribute, Html, a)
import Html.Attributes exposing (..)



-- VIEW


view : List (Html.Attribute msg) -> List (Html msg) -> Html msg
view attr content =
    a (class "gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select" :: attr) content
