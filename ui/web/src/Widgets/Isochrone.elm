module Widgets.Isochrone exposing
    ( Config(..)
    , Model
    , Msg(..)
    , init
    , update
    , view
    )

import Data.Connection.Types exposing (Station, Position)
import Data.ScheduleInfo.Types exposing (ScheduleInfo)
import Date exposing (Date)
import Data.Isochrone.Types exposing( IsochroneRequest, IsochroneResponse)
import Debounce
import Data.Isochrone.Decode exposing (decodeIsochroneResponse)
import Data.Isochrone.Request as IsochroneRequest exposing (encodeIsochroneRequest, IntermodalLocation(..))
import Html exposing (Html, a, div, i, input, label, li, span, text, ul)
import Html.Attributes exposing (..)
import Html.Events exposing (on, onClick, onInput)
import Html.Lazy exposing (..)
import Localization.Base exposing (..)
import Maybe.Extra exposing (isJust)
import ProgramFlags exposing (..)
import Routes exposing (..)
import Time exposing (Time)
import Util.Api as Api exposing (ApiError)
import Util.Core exposing ((=>))
import Util.Date exposing (combineDateTime, isSameDay, unixTime)
import Util.DateFormat exposing (..)
import Widgets.Calendar as Calendar
import Widgets.Helpers.ApiErrorUtil exposing (errorText)
import Navigation exposing (Location)
import Widgets.Helpers.ConnectionUtil
    exposing
        ( TransportViewMode(..)
        , delay
        , trainBox
        )
import Widgets.DurationInput as DurationInput
import Widgets.Input as Input
import Widgets.LoadingSpinner as LoadingSpinner
import Widgets.TimeInput as TimeInput
import Widgets.Typeahead as Typeahead
import Widgets.Map.Port as Port exposing (..)



-- MODEL


type alias Model =
    { loading : Bool
    , locationInput : Typeahead.Model
    , stationWidget : Input.Model
    , apiEndpoint : String
    , date : Calendar.Model
    , time : TimeInput.Model
    , duration : String
    , errorMessage : Maybe ApiError
    , currentRequest : Maybe IsochroneRequest
    , debounce : Debounce.State
    , durationWidget : Input.Model
    }


type Config msg
    = Config
        { internalMsg : Msg -> msg
        , selectStationMsg : Station -> Maybe Date -> msg
        }


init : ProgramFlags -> Localization -> ( Model, Cmd Msg )
init flags locale =
    let
        remoteAddress =
                    flags.apiEndpoint

        ( dateModel, dateCmd ) =
            Calendar.init locale.dateConfig

        ( timeModel, timeCmd ) =
            TimeInput.init False

        fromLocation =
                    flags.fromLocation
                        |> Maybe.withDefault ""

        ( fromLocationModel, fromLocationCmd ) =
                    Typeahead.init remoteAddress fromLocation

        initialModel =
                    { loading = False
                    , locationInput = fromLocationModel
                    , apiEndpoint = remoteAddress
                    , stationWidget = Input.init
                    , date = dateModel
                    , time = timeModel
                    , duration = ""
                    , errorMessage = Nothing
                    , currentRequest = Nothing
                    , debounce = Debounce.init
                    , durationWidget = Input.init
                    }

    in initialModel
        ! [ Cmd.map DateUpdate dateCmd
          , Cmd.map TimeUpdate timeCmd
          ]



-- UPDATE


type Msg
    = NoOp
    | ReceiveResponse IsochroneRequest IsochroneResponse
    | ReceiveError IsochroneRequest ApiError
    | SetLocale Localization
    | SetTime Date
    | DateUpdate Calendar.Msg
    | TimeUpdate TimeInput.Msg
    | DurationUpdate Input.Msg
    | UpdateScheduleInfo ScheduleInfo
    | DurationInput String
    | LocationInputUpdate Typeahead.Msg
    | Isochrones
    | Deb (Debounce.Msg Msg)


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            model ! []

        ReceiveResponse request response ->
            let
                cmd = generateIsochrones response.station response.arrival_times
            in
                { model
                    | loading = False
                }
                    ! [cmd]

        ReceiveError request err ->
            { model
                | errorMessage = Just err
                , loading = False
            }
                ! []

        Deb a ->
            Debounce.update debounceCfg a model

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

        LocationInputUpdate msg_ ->
           let
               ( m, c ) =
                   Typeahead.update msg_ model.locationInput
           in
           { model | locationInput = m }
               ! [ Cmd.map LocationInputUpdate c ]
               |> checkTypeaheadUpdate msg_

        DurationInput dur ->
            checkRequest ({ model | duration= dur } ! [])

        Isochrones ->
            let
                request =
                    buildRequest model
            in
            { model
                | currentRequest = Just request
                , loading = True
            }
                ! [ sendRequest model.apiEndpoint request ]
        DateUpdate msg_ ->
            checkRequest ({ model | date = Calendar.update msg_ model.date } ! [])

        TimeUpdate msg_ ->
            checkRequest ({ model | time = TimeInput.update msg_ model.time } ! [])

        DurationUpdate msg_ ->
            { model | durationWidget = Input.update msg_ model.durationWidget } ! []

        UpdateScheduleInfo si ->
            let
                newDate =
                    Calendar.update (Calendar.SetValidRange (Just ( si.begin, si.end ))) model.date
            in
            { model | date = newDate } ! []

checkTypeaheadUpdate : Typeahead.Msg -> ( Model, Cmd Msg ) -> ( Model, Cmd Msg )
checkTypeaheadUpdate msg ( model, cmds ) =
    let
        model_ = model
    in
    checkRequest ( model_, cmds )

checkRequest : ( Model, Cmd Msg ) -> ( Model, Cmd Msg )
checkRequest ( model, cmds ) =
    let


        newRequest =
            buildRequest model


    in
        model ! [ cmds, Debounce.debounceCmd debounceCfg <| Isochrones ]



buildRequest : Model -> IsochroneRequest
buildRequest model =
    let
        default =
                    IntermodalPosition (Position 0 0)



        toPosition typeahead =
                    case Typeahead.getSelectedSuggestion typeahead of
                        Just (Typeahead.StationSuggestion s) ->
                            IntermodalStation s
                        Just (Typeahead.AddressSuggestion a) ->
                            IntermodalPosition a.pos

                        Just (Typeahead.PositionSuggestion p) ->
                            IntermodalPosition p
                        Nothing ->
                            default

        position = toPosition model.locationInput


        time =
            unixTime (combineDateTime model.date.date model.time.date)

        duration =
            case String.toInt model.duration of
                Err msg -> 0
                Ok val -> val * 60

        foot_time = 900

    in
    IsochroneRequest.initialRequest
        position
        time
        duration
        foot_time

isCompleteQuery : Model -> Bool
isCompleteQuery model =
    model.duration /= "" && model.duration /= "0"



debounceCfg : Debounce.Config Model Msg
debounceCfg =
    Debounce.config
        .debounce
        (\model s -> { model | debounce = s })
        Deb
        300

generateIsochrones : List Station -> List Int -> Cmd msg
generateIsochrones stations times =
    Port.mapGenerateIsochrones
        {
        stations = stations
        ,times = times
        }

-- VIEW


view : Config msg -> Localization -> Model -> Html msg
view config locale model =
    div [ class "isochrone" ]
        [ lazy3 headerView config locale model
        , lazy3 contentView config locale model
        ]


headerView : Config msg -> Localization -> Model -> Html msg
headerView config locale model =
    div [ class "header" ]
        (searchView config locale model)


searchView : Config msg -> Localization -> Model -> List (Html msg)
searchView (Config { internalMsg }) locale model =
    [ div [ id "isochrone-form" ]
        [ div [ class "pure-g gutters" ]
            [ div [ class "pure-u-1 pure-u-sm-12-24 from-location" ]
                [ Html.map (internalMsg << LocationInputUpdate) <|
                    Typeahead.view 1 locale.t.search.start (Just "place") model.locationInput
                ]
            , div [ class "pure-u-1 pure-u-sm-12-24" ]
                [ Html.map (internalMsg << DateUpdate) <|
                    Calendar.view 2 locale.t.search.date model.date
                ]
            ]
        , div [ class "pure-g gutters" ]
            [ div [ class "pure-u-1 pure-u-sm-12-24 to-location" ]
                [ Html.map (internalMsg) <|
                    durationInputView 3 locale.t.search.maxDuration model
                ]
            , div [ class "pure-u-1 pure-u-sm-12-24" ]
                [ Html.map (internalMsg << TimeUpdate) <|
                    TimeInput.view 4 locale.t.search.time model.time
                ]
            ]
        ]
    ]


durationInputView : Int -> String -> Model -> Html Msg
durationInputView tabIndex label model =
    Input.view DurationUpdate
        [ onInput DurationInput
        , value model.duration
        , tabindex tabIndex
        , attribute "inputmode" "numeric"
        , attribute "pattern" "[0-9]+"
        , id "isochrone-duration-input"
        ]
        label
        Nothing
        (Just "adjust")
        model.durationWidget


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
                        text ""


    in
    div [ class "isochrone" ] [ content ]





errorView : String -> Localization -> Model -> ApiError -> Html msg
errorView divClass locale model err =
    let
        errorMsg =
            errorText locale err
    in
    div [ class divClass ]
        [ div [] [ text errorMsg ] ]



-- API


sendRequest : String -> IsochroneRequest -> Cmd Msg
sendRequest remoteAddress request =
    Api.sendRequest
        (remoteAddress ++ "?elm=isochrone")
        decodeIsochroneResponse
        (ReceiveError request)
        (ReceiveResponse request)
        (encodeIsochroneRequest request)
