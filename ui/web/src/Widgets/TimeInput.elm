module Widgets.TimeInput exposing
    ( Model
    , Msg(..)
    , init
    , update
    , view
    )

import Date exposing (Date)
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
    { date : Date
    , inputStr : String
    , inputWidget : Input.Model
    , withSeconds : Bool
    }


init : Bool -> ( Model, Cmd Msg )
init withSeconds =
    ( { date = Date.fromTime 0
      , inputStr = formatDate withSeconds (Date.fromTime 0)
      , inputWidget = Input.init
      , withSeconds = withSeconds
      }
    , getCurrentDate
    )


getCurrentDate : Cmd Msg
getCurrentDate =
    Task.perform (InitDate False) Date.now



-- UPDATE


type Msg
    = TimeInput String
    | InitDate Bool Date
    | NoOp String
    | PrevHour
    | NextHour
    | InputUpdate Input.Msg


update : Msg -> Model -> Model
update msg model =
    case msg of
        TimeInput s ->
            case parseInput s of
                Nothing ->
                    { model | inputStr = s }

                Just date ->
                    { model | date = date, inputStr = s }

        InitDate force d ->
            if force || Date.toTime model.date == 0 then
                let
                    date =
                        fieldToDateClamp (Second 0) d
                in
                { model
                    | date = date
                    , inputStr = formatDate model.withSeconds date
                }

            else
                model

        NoOp s ->
            model

        PrevHour ->
            let
                newDate =
                    Duration.add Duration.Hour -1 model.date
            in
            { model | date = newDate, inputStr = formatDate model.withSeconds newDate }

        NextHour ->
            let
                newDate =
                    Duration.add Duration.Hour 1 model.date
            in
            { model | date = newDate, inputStr = formatDate model.withSeconds newDate }

        InputUpdate msg_ ->
            { model | inputWidget = Input.update msg_ model.inputWidget }


parseInput : String -> Maybe Date
parseInput str =
    let
        hourStr =
            intNthToken 0 ":" str

        minuteStr =
            intNthToken 1 ":" str

        secondStr =
            intNthToken 2 ":" str
                |> Maybe.withDefault 0
                |> Just
    in
    Maybe.map3 toDate hourStr minuteStr secondStr


toDate : Int -> Int -> Int -> Date
toDate h m s =
    dateFromFields 0 Date.Jan 0 h m s 0


formatDate : Time.Zone -> Bool -> Time.Posix -> String
formatDate zone withSeconds d =
    let
        h =
            twoDigits (Time.toHour zone d)

        m =
            twoDigits (Time.toMinute zone d)

        s =
            twoDigits (Time.toSecond zone d)
    in
    if withSeconds then
        h ++ ":" ++ m ++ ":" ++ s

    else
        h ++ ":" ++ m


twoDigits : Int -> String
twoDigits =
    String.fromInt >> String.padLeft 2 '0'



-- VIEW


hourButtons : Html Msg
hourButtons =
    div [ class "hour-buttons" ]
        [ div [] [ Button.view [ onClick PrevHour ] [ i [ class "icon" ] [ text "chevron_left" ] ] ]
        , div [] [ Button.view [ onClick NextHour ] [ i [ class "icon" ] [ text "chevron_right" ] ] ]
        ]


timeInputView : Int -> String -> Model -> Html Msg
timeInputView tabIndex label model =
    Input.view InputUpdate
        [ onInput TimeInput
        , value model.inputStr
        , tabindex tabIndex
        ]
        label
        (Just [ hourButtons ])
        (Just "schedule")
        model.inputWidget


view : Int -> String -> Model -> Html Msg
view tabIndex label model =
    lazy3 timeInputView tabIndex label model
