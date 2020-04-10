module Widgets.SearchProfileEditor exposing (Model, Msg(..), init, update, view)

import Data.PPR.Types exposing (SearchProfile)
import Html exposing (Attribute, Html, a, div, i, input, label, span, text)
import Html.Attributes exposing (..)
import Html.Events exposing (onClick)
import Localization.Base exposing (..)
import Task
import Util.Core exposing ((=>))



-- MODEL


type alias Model =
    { visible : Bool
    , profile : SearchProfile
    }


init : SearchProfile -> ( Model, Cmd Msg )
init initialProfile =
    { visible = False
    , profile = initialProfile
    }
        ! []



-- UPDATE


type Msg
    = Show
    | Hide
    | Toggle


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        Show ->
            { model | visible = True } ! []

        Hide ->
            { model | visible = False } ! []

        Toggle ->
            if model.visible then
                update Hide model

            else
                update Show model



-- VIEW


view : Localization -> Model -> Html Msg
view locale model =
    div
        [ classList
            [ "search-profile-editor" => True
            , "hidden" => not model.visible
            ]
        ]
        [ div [ class "title" ]
            [ text locale.t.simTime.simMode ]
        , div [ class "close", onClick Hide ]
            [ i [ class "icon" ] [ text "close" ] ]
        ]
