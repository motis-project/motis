module Widgets.TripSearch exposing
    ( Config(..)
    , Model
    , Msg(..)
    , init
    , update
    , view
    )

import Data.Connection.Types exposing (Station, TripId)
import Data.RailViz.Decode exposing (decodeRailVizTripGuessResponse)
import Data.RailViz.Request as Request exposing (encodeTripGuessRequest)
import Data.RailViz.Types exposing (RailVizTripGuessRequest, RailVizTripGuessResponse, Trip, TripInfo)
import Data.ScheduleInfo.Types exposing (ScheduleInfo)
import Date exposing (Date)
import Debounce
import Html exposing (Html, a, div, i, input, label, li, span, text, ul)
import Html.Attributes exposing (..)
import Html.Events exposing (on, onClick, onInput)
import Html.Lazy exposing (..)
import Localization.Base exposing (..)
import Maybe.Extra exposing (isJust)
import Time exposing (Time)
import Util.Api as Api exposing (ApiError)
import Util.Core exposing ((=>))
import Util.Date exposing (combineDateTime, isSameDay, unixTime)
import Util.DateFormat exposing (..)
import Widgets.Calendar as Calendar
import Widgets.Helpers.ApiErrorUtil exposing (errorText)
import Widgets.Helpers.ConnectionUtil
    exposing
        ( TransportViewMode(..)
        , delay
        , trainBox
        )
import Widgets.Input as Input
import Widgets.LoadingSpinner as LoadingSpinner
import Widgets.TimeInput as TimeInput



-- MODEL


type alias Model =
    { loading : Bool
    , remoteAddress : String
    , trainNrInput : String
    , trainNrWidget : Input.Model
    , date : Calendar.Model
    , time : TimeInput.Model
    , errorMessage : Maybe ApiError
    , currentRequest : Maybe RailVizTripGuessRequest
    , debounce : Debounce.State
    , trips : Maybe (List Trip)
    }


type Config msg
    = Config
        { internalMsg : Msg -> msg
        , selectTripMsg : TripId -> msg
        , selectStationMsg : Station -> Maybe Date -> msg
        }


init : String -> Localization -> ( Model, Cmd Msg )
init remoteAddress locale =
    let
        ( dateModel, dateCmd ) =
            Calendar.init locale.dateConfig

        ( timeModel, timeCmd ) =
            TimeInput.init False
    in
    { loading = False
    , remoteAddress = remoteAddress
    , trainNrInput = ""
    , trainNrWidget = Input.init
    , date = dateModel
    , time = timeModel
    , errorMessage = Nothing
    , currentRequest = Nothing
    , debounce = Debounce.init
    , trips = Nothing
    }
        ! [ Cmd.map DateUpdate dateCmd
          , Cmd.map TimeUpdate timeCmd
          ]



-- UPDATE


type Msg
    = NoOp
    | ReceiveResponse RailVizTripGuessRequest RailVizTripGuessResponse
    | ReceiveError RailVizTripGuessRequest ApiError
    | SetLocale Localization
    | SetTime Date
    | TrainNrInput String
    | TrainNrInputUpdate Input.Msg
    | DateUpdate Calendar.Msg
    | TimeUpdate TimeInput.Msg
    | Deb (Debounce.Msg Msg)
    | SearchTrips
    | UpdateScheduleInfo ScheduleInfo


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            model ! []

        ReceiveResponse request response ->
            { model
                | trips = Just response.trips
                , loading = False
            }
                ! []

        ReceiveError request err ->
            { model
                | trips = Nothing
                , errorMessage = Just err
                , loading = False
            }
                ! []

        SetLocale locale ->
            { model
                | date = Calendar.update (Calendar.SetDateConfig locale.dateConfig) model.date
            }
                ! []

        SetTime newDate ->
            let
                ( model1, cmds1 ) =
                    update (DateUpdate (Calendar.InitDate True newDate)) model

                ( model2, cmds2 ) =
                    update (TimeUpdate (TimeInput.InitDate True newDate)) model1
            in
            model2 ! [ cmds1, cmds2 ]

        TrainNrInput str ->
            checkRequest ({ model | trainNrInput = str } ! [])

        TrainNrInputUpdate msg_ ->
            { model | trainNrWidget = Input.update msg_ model.trainNrWidget } ! []

        DateUpdate msg_ ->
            checkRequest ({ model | date = Calendar.update msg_ model.date } ! [])

        TimeUpdate msg_ ->
            checkRequest ({ model | time = TimeInput.update msg_ model.time } ! [])

        Deb a ->
            Debounce.update debounceCfg a model

        SearchTrips ->
            let
                request =
                    buildRequest model
            in
            { model
                | currentRequest = Just request
                , loading = True
            }
                ! [ sendRequest model.remoteAddress request ]

        UpdateScheduleInfo si ->
            let
                newDate =
                    Calendar.update (Calendar.SetValidRange (Just ( si.begin, si.end ))) model.date
            in
            { model | date = newDate } ! []


checkRequest : ( Model, Cmd Msg ) -> ( Model, Cmd Msg )
checkRequest ( model, cmds ) =
    let
        completeQuery =
            isCompleteQuery model

        newRequest =
            buildRequest model

        requestChanged =
            case model.currentRequest of
                Just currentRequest ->
                    newRequest /= currentRequest

                Nothing ->
                    True
    in
    if completeQuery && requestChanged then
        model ! [ cmds, Debounce.debounceCmd debounceCfg <| SearchTrips ]

    else
        ( model, cmds )


isCompleteQuery : Model -> Bool
isCompleteQuery model =
    isJust (getTrainNr model.trainNrInput)


buildRequest : Model -> RailVizTripGuessRequest
buildRequest model =
    let
        trainNr =
            getTrainNr model.trainNrInput
                |> Maybe.withDefault 0

        time =
            unixTime (combineDateTime model.date.date model.time.date)
    in
    { trainNum = trainNr
    , time = time
    , guessCount = 20
    }


getTrainNr : String -> Maybe Int
getTrainNr input =
    case String.toInt input of
        Ok nr ->
            if nr >= 0 then
                Just nr

            else
                Nothing

        Err _ ->
            Nothing


debounceCfg : Debounce.Config Model Msg
debounceCfg =
    Debounce.config
        .debounce
        (\model s -> { model | debounce = s })
        Deb
        300



-- VIEW


view : Config msg -> Localization -> Model -> Html msg
view config locale model =
    div [ class "trip-search" ]
        [ lazy3 headerView config locale model
        , lazy3 contentView config locale model
        ]


headerView : Config msg -> Localization -> Model -> Html msg
headerView config locale model =
    div [ class "header" ]
        (searchView config locale model)


searchView : Config msg -> Localization -> Model -> List (Html msg)
searchView (Config { internalMsg }) locale model =
    [ div [ id "trip-search-form" ]
        [ div [ class "pure-g gutters" ]
            [ div [ class "pure-u-1 pure-u-sm-1-2 train-nr" ]
                [ Html.map internalMsg <|
                    trainNrInputView 1 locale.t.search.trainNr model
                ]
            ]
        , div [ class "pure-g gutters" ]
            [ div [ class "pure-u-1 pure-u-sm-12-24 to-location" ]
                [ Html.map (internalMsg << DateUpdate) <|
                    Calendar.view 3 locale.t.search.date model.date
                ]
            , div [ class "pure-u-1 pure-u-sm-12-24" ]
                [ Html.map (internalMsg << TimeUpdate) <|
                    TimeInput.view 4 locale.t.search.time model.time
                ]
            ]
        ]
    ]


trainNrInputView : Int -> String -> Model -> Html Msg
trainNrInputView tabIndex label model =
    Input.view TrainNrInputUpdate
        [ onInput TrainNrInput
        , value model.trainNrInput
        , tabindex tabIndex
        , attribute "inputmode" "numeric"
        , attribute "pattern" "[0-9]+"
        , id "trip-search-trainnr-input"
        ]
        label
        Nothing
        (Just "train")
        model.trainNrWidget


contentView : Config msg -> Localization -> Model -> Html msg
contentView config locale model =
    let
        content =
            if model.loading then
                div [ class "loading" ] [ LoadingSpinner.view ]

            else
                case model.errorMessage of
                    Just err ->
                        errorView "main-error" locale model err

                    Nothing ->
                        tripsView config locale model
    in
    div [ class "trips" ] [ content ]


tripsView : Config msg -> Localization -> Model -> Html msg
tripsView config locale model =
    case model.trips of
        Just trips ->
            if List.isEmpty trips then
                div [ class "no-results" ]
                    [ div [] [ text locale.t.trips.noResults ] ]

            else
                div [] (List.map (tripView config locale model.date.date) trips)

        Nothing ->
            text ""


tripView : Config msg -> Localization -> Date -> Trip -> Html msg
tripView (Config { internalMsg, selectTripMsg, selectStationMsg }) locale queryDate trip =
    let
        departureTime =
            Date.fromTime (toFloat trip.tripInfo.id.time * 1000)

        transport =
            trip.tripInfo.transport

        direction =
            if not (String.isEmpty transport.direction) then
                div
                    [ class "direction"
                    , title transport.direction
                    ]
                    [ i [ class "icon" ] [ text "arrow_forward" ]
                    , text transport.direction
                    ]

            else
                text ""

        dateText =
            if isSameDay queryDate departureTime then
                text ""

            else
                div [ class "date" ] [ text (formatShortDate locale.dateConfig departureTime) ]
    in
    div [ class "trip" ]
        [ div [ class "trip-train" ]
            [ span
                [ onClick (selectTripMsg trip.tripInfo.id) ]
                [ trainBox LongName locale transport ]
            ]
        , div [ class "trip-time" ]
            [ div [ class "time" ] [ text (formatTime departureTime) ]
            , dateText
            ]
        , div [ class "trip-first-station" ]
            [ div
                [ class "station"
                , onClick (selectStationMsg trip.firstStation (Just departureTime))
                , title trip.firstStation.name
                ]
                [ text trip.firstStation.name ]
            , direction
            ]
        ]


errorView : String -> Localization -> Model -> ApiError -> Html msg
errorView divClass locale model err =
    let
        errorMsg =
            errorText locale err
    in
    div [ class divClass ]
        [ div [] [ text errorMsg ] ]



-- API


sendRequest : String -> RailVizTripGuessRequest -> Cmd Msg
sendRequest remoteAddress request =
    Api.sendRequest
        (remoteAddress ++ "?elm=TripSearch")
        decodeRailVizTripGuessResponse
        (ReceiveError request)
        (ReceiveResponse request)
        (encodeTripGuessRequest request)
