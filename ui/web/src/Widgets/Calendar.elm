module Widgets.Calendar exposing (Model, Msg(..), init, update, view)

import Date exposing (Date, Day, day, dayOfWeek)
import Html exposing (..)
import Html.Attributes exposing (..)
import Html.Events exposing (onClick, onInput)
import Html.Lazy exposing (..)
import Task
import Util.DateUtil exposing (atNoon)
import Util.DateFormat exposing (..)
import Util.View exposing (onStopAll)
import Widgets.Button as Button
import Widgets.Input as Input



-- MODEL


type alias Model =
    { conf : DateConfig
    , today : Date
    , date : Date
    , inputStr : String
    , visible : Bool
    , inputWidget : Input.Model
    , validRange : Maybe ( Date, Date )
    }


init : DateConfig -> ( Model, Cmd Msg )
init dateConfig =
    ( emptyModel dateConfig, getCurrentDate )


emptyModel : DateConfig -> Model
emptyModel dateConfig =
    { conf = dateConfig
    , today = Date.fromTime 0
    , date = Date.fromTime 0
    , visible = False
    , inputStr = ""
    , inputWidget = Input.init
    , validRange = Nothing
    }



-- UPDATE


type Msg
    = NoOp
    | InitDate Bool Date
    | NewDate Date
    | DateInput String
    | PrevDay
    | NextDay
    | PrevMonth
    | NextMonth
    | ToggleVisibility
    | InputUpdate Input.Msg
    | SetValidRange (Maybe ( Date, Date ))
    | SetDateConfig DateConfig


update : Msg -> Model -> Model
update msg model =
    case msg of
        NoOp ->
            model

        InputUpdate msg_ ->
            let
                updated =
                    case msg_ of
                        Input.Focus ->
                            { model | visible = True }

                        Input.Click ->
                            { model | visible = True }

                        Input.Blur ->
                            { model | visible = False }
            in
            { updated | inputWidget = Input.update msg_ model.inputWidget }

        InitDate force d ->
            let
                d_ =
                    atNoon d
            in
            if force || Date.toTime model.date == 0 then
                { model
                    | date = d_
                    , inputStr = formatDate model.conf d_
                    , today = d_
                }

            else
                model

        NewDate d ->
            { model
                | date = d
                , inputStr = formatDate model.conf d
                , visible = False
            }

        DateInput str ->
            case parseDate model.conf str of
                Nothing ->
                    { model | inputStr = str }

                Just date ->
                    { model | date = date, inputStr = str }

        PrevDay ->
            let
                newDate =
                    Duration.add Duration.Day -1 model.date
            in
            { model
                | date = newDate
                , inputStr = formatDate model.conf newDate
            }

        NextDay ->
            let
                newDate =
                    Duration.add Duration.Day 1 model.date
            in
            { model
                | date = newDate
                , inputStr = formatDate model.conf newDate
            }

        PrevMonth ->
            let
                newDate =
                    Duration.add Duration.Month -1 model.date
            in
            { model
                | date = newDate
                , inputStr = formatDate model.conf newDate
            }

        NextMonth ->
            let
                newDate =
                    Duration.add Duration.Month 1 model.date
            in
            { model
                | date = newDate
                , inputStr = formatDate model.conf newDate
            }

        ToggleVisibility ->
            model

        SetValidRange range ->
            { model | validRange = range }

        SetDateConfig newConf ->
            { model
                | conf = newConf
                , inputStr = formatDate newConf model.date
            }



-- VIEW


weekDays : DateConfig -> List (Html Msg)
weekDays conf =
    dayListForMonthView Nothing (Date.fromTime 0) (Date.fromTime 0)
        |> List.take 7
        |> List.map (\d -> li [] [ text (weekDayName conf d.day) ])


calendarDay : CalendarDay -> Html Msg
calendarDay date =
    li
        [ onClick (NewDate date.day)
        , classList
            [ ( "out-of-month", not date.inMonth )
            , ( "in-month", date.inMonth )
            , ( "today", date.today )
            , ( "selected", date.selected )
            , ( "valid-day", Maybe.withDefault False date.valid )
            , ( "invalid-day", not <| Maybe.withDefault True date.valid )
            ]
        ]
        [ text (toString (day date.day)) ]


calendarDays : Maybe ( Date, Date ) -> Date -> Date -> List (Html Msg)
calendarDays validRange today date =
    List.map calendarDay (dayListForMonthView validRange today date)


monthView : DateConfig -> Date -> Html Msg
monthView conf date =
    div [ class "month" ]
        [ i [ class "icon", onClick PrevMonth ] [ text "chevron_left" ]
        , span [ class "month-name" ] [ text (monthAndYearStr conf date) ]
        , i [ class "icon", onClick NextMonth ] [ text "chevron_right" ]
        ]


dayButtons : Html Msg
dayButtons =
    div [ class "day-buttons" ]
        [ div [] [ Button.view [ onStopAll "mousedown" PrevDay ] [ i [ class "icon" ] [ text "chevron_left" ] ] ]
        , div [] [ Button.view [ onStopAll "mousedown" NextDay ] [ i [ class "icon" ] [ text "chevron_right" ] ] ]
        ]


calendarView : Int -> String -> Model -> Html Msg
calendarView tabIndex label model =
    div []
        [ Input.view InputUpdate
            [ onInput DateInput
            , value model.inputStr
            , tabindex tabIndex
            ]
            label
            (Just [ dayButtons ])
            (Just "event")
            model.inputWidget
        , div
            [ classList
                [ ( "paper", True )
                , ( "calendar", True )
                , ( "hide", not model.visible )
                ]
            , onStopAll "mousedown" NoOp
            ]
            [ monthView model.conf model.date
            , ul [ class "weekdays" ] (weekDays model.conf)
            , ul [ class "calendardays" ] (calendarDays model.validRange model.today model.date)
            ]
        ]


view : Int -> String -> Model -> Html Msg
view =
    lazy3 calendarView



-- DATE


type alias CalendarDay =
    { day : Date
    , inMonth : Bool
    , today : Bool
    , selected : Bool
    , valid : Maybe Bool
    }


getCurrentDate : Cmd Msg
getCurrentDate =
    Task.perform (InitDate False) Date.now


daysInSixWeeks : Int
daysInSixWeeks =
    42


dayListForMonthView : Maybe ( Date, Date ) -> Date -> Date -> List CalendarDay
dayListForMonthView validRange today selected =
    let
        firstOfMonth =
            toFirstOfMonth selected

        lastOfMonth =
            lastOfMonthDate selected

        daysToStart =
            isoDayOfWeek (dayOfWeek firstOfMonth) - 1

        first =
            Duration.add Duration.Day -daysToStart firstOfMonth
    in
    dayList daysInSixWeeks first
        |> List.map
            (\date ->
                { day = date
                , inMonth = Compare.is3 Compare.BetweenOpen date firstOfMonth lastOfMonth
                , today = Compare.is Compare.Same today date
                , selected = Compare.is Compare.Same selected date
                , valid = Maybe.map2 isValidDay validRange (Just date)
                }
            )


isValidDay : ( Date, Date ) -> Date -> Bool
isValidDay ( begin, end ) day =
    Compare.is3 Compare.BetweenOpen day begin end
