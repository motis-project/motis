module Widgets.DurationInput exposing
    ( Model
    , Msg(InitDur)
    , init
    , update
    , view
    )


import Html exposing (..)
import Html.Attributes exposing (class, tabindex, value)
import Html.Events exposing (onClick, onInput)
import Html.Lazy exposing (..)
import String
import Task
import Util.StringSplit exposing (..)
import Widgets.Button as Button
import Widgets.Input as Input



-- MODEL


type alias Model =
    { duration : Int
    , inputStr : String
    , inputWidget : Input.Model
    }

{-
init : Model
init =
    { duration = 0
    , inputStr = ""
    , inputWidget = Input.init
    }
    -}
init : Int ->  Model
init d_i =
    ( { duration = d_i
      , inputStr = toString d_i
      , inputWidget = Input.init
      })






-- UPDATE


type Msg
    = DurationInput String
    | InitDur Int
    | NoOp String
    | InputUpdate Input.Msg
    | AddTen
    | SubTen


update : Msg -> Model -> Model
update msg model =
    case msg of
        DurationInput s ->
            case String.toInt s of
                Err err ->
                    { model | inputStr = s }

                Ok val ->
                    { model | duration = val, inputStr = s }

        NoOp s ->
            model

        InputUpdate msg_ ->
            { model | inputWidget = Input.update msg_ model.inputWidget }

        InitDur d ->
            model

        AddTen ->
            let
                newDuration =
                    model.duration + 10
            in
            { model | duration = newDuration, inputStr =  toString newDuration }

        SubTen ->
            let
                newDuration =
                    model.duration - 10
            in
            { model | duration = newDuration, inputStr = toString newDuration }

-- View

durButtons : Html Msg
durButtons =
    div [ class "hour-buttons" ]
        [ div [] [ Button.view [ onClick AddTen ] [ i [ class "icon" ] [ text "chevron_left" ] ] ]
        , div [] [ Button.view [ onClick SubTen ] [ i [ class "icon" ] [ text "chevron_right" ] ] ]
        ]


durationInputView : Int -> String -> Model -> Html Msg
durationInputView tabIndex label model =
    Input.view InputUpdate
        [ onInput DurationInput
        , value model.inputStr
        , tabindex tabIndex
        ]
        label
        (Just [ durButtons ])
        (Just "schedule")
        model.inputWidget


view : Int -> String -> Model -> Html Msg
view tabIndex label model =
    lazy3 durationInputView tabIndex label model
