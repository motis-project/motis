module Widgets.Input exposing
    ( Model
    , Msg(..)
    , addIcon
    , addInputWidget
    , init
    , update
    , view
    )

import Html exposing (Attribute, Html, div, i, input, text)
import Html.Attributes exposing (..)
import Html.Events exposing (onBlur, onClick, onFocus)



-- MODEL


type alias Model =
    Bool


init : Model
init =
    False



-- UPDATE


type Msg
    = Focus
    | Blur
    | Click


update : Msg -> Model -> Model
update msg model =
    case msg of
        Focus ->
            True

        Blur ->
            False

        Click ->
            model



-- VIEW


addIcon : Maybe String -> List (Html msg) -> List (Html msg)
addIcon icon list =
    case icon of
        Just str ->
            div [ class "gb-input-icon" ]
                [ i [ class "icon" ] [ text str ] ]
                :: list

        Nothing ->
            list


addInputWidget : Maybe (List (Html msg)) -> List (Html msg) -> List (Html msg)
addInputWidget inputWidget list =
    case inputWidget of
        Just widget ->
            List.append list [ div [ class "gb-input-widget" ] widget ]

        Nothing ->
            list


view :
    (Msg -> msg)
    -> List (Html.Attribute msg)
    -> String
    -> Maybe (List (Html msg))
    -> Maybe String
    -> Model
    -> Html msg
view makeMsg attributes label inputWidget icon model =
    div []
        [ div [ class "label" ] [ text label ]
        , div
            [ classList
                [ ( "gb-input-group", True )
                , ( "gb-input-group-selected", model )
                ]
            ]
            ([ input
                (class "gb-input"
                    :: onBlur (makeMsg Blur)
                    :: onFocus (makeMsg Focus)
                    :: onClick (makeMsg Click)
                    :: attributes
                )
                []
             ]
                |> addInputWidget inputWidget
                |> addIcon icon
            )
        ]
