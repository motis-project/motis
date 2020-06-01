module Widgets.Routing exposing
    ( Model
    , Msg(..)
    , getDestinationSearchProfile
    , getStartSearchProfile
    , init
    , subscriptions
    , update
    , view
    )

import Data.Connection.Types exposing (Connection, Position, Station, TripId)
import Data.Intermodal.Request as IntermodalRoutingRequest exposing (IntermodalLocation(..))
import Data.Intermodal.Types as Intermodal exposing (IntermodalRoutingRequest)
import Data.PPR.Request exposing (encodeSearchOptions)
import Data.PPR.Types exposing (SearchOptions)
import Data.Routing.Decode exposing (decodeRoutingResponse)
import Data.Routing.Types exposing (SearchDirection(..))
import Data.ScheduleInfo.Types exposing (ScheduleInfo)
import Date exposing (Date)
import Debounce
import Dom.Scroll as Scroll
import Html exposing (..)
import Html.Attributes exposing (..)
import Html.Events exposing (..)
import Html.Lazy exposing (..)
import Json.Decode as Decode
import Localization.Base exposing (..)
import Maybe.Extra exposing (isJust)
import Navigation exposing (Location)
import Port
import ProgramFlags exposing (..)
import Routes exposing (..)
import Task
import Util.Api as Api exposing (ApiError(..))
import Util.Core exposing ((=>))
import Util.Date exposing (combineDateTime)
import Widgets.Calendar as Calendar
import Widgets.Connections as Connections
import Widgets.Map.RailViz as RailViz
import Widgets.ModePicker as ModePicker
import Widgets.TimeInput as TimeInput
import Widgets.Typeahead as Typeahead



-- MODEL


type alias Model =
    { fromLocation : Typeahead.Model
    , toLocation : Typeahead.Model
    , fromModes : ModePicker.Model
    , toModes : ModePicker.Model
    , date : Calendar.Model
    , time : TimeInput.Model
    , searchDirection : SearchDirection
    , connections : Connections.Model
    , apiEndpoint : String
    , currentRoutingRequest : Maybe IntermodalRoutingRequest
    , debounce : Debounce.State
    , connectionListScrollPos : Float
    , optionsVisible : Bool
    , intermodalPprMode : ModePicker.PprProfileMode
    }


init : ProgramFlags -> Localization -> ( Model, Cmd Msg )
init flags locale =
    let
        remoteAddress =
            flags.apiEndpoint

        fromLocation =
            flags.fromLocation
                |> Maybe.withDefault ""

        toLocation =
            flags.toLocation
                |> Maybe.withDefault ""

        ( dateModel, dateCmd ) =
            Calendar.init locale.dateConfig

        ( timeModel, timeCmd ) =
            TimeInput.init False

        ( fromLocationModel, fromLocationCmd ) =
            Typeahead.init remoteAddress fromLocation

        ( toLocationModel, toLocationCmd ) =
            Typeahead.init remoteAddress toLocation

        intermodalPprMode =
            flags.intermodalPprMode
                |> Maybe.andThen loadPprMode
                |> Maybe.withDefault ModePicker.PremadePprProfiles

        connections =
            Connections.init remoteAddress

        initialModel =
            { fromLocation = fromLocationModel
            , toLocation = toLocationModel
            , fromModes = ModePicker.init flags.fromModes intermodalPprMode
            , toModes = ModePicker.init flags.toModes intermodalPprMode
            , date = dateModel
            , time = timeModel
            , searchDirection = Forward
            , connections = Connections.init remoteAddress
            , apiEndpoint = remoteAddress
            , currentRoutingRequest = Nothing
            , debounce = Debounce.init
            , connectionListScrollPos = 0
            , optionsVisible = True
            , intermodalPprMode = intermodalPprMode
            }
    in
    initialModel
        ! [ Cmd.map DateUpdate dateCmd
          , Cmd.map TimeUpdate timeCmd
          , Cmd.map FromLocationUpdate fromLocationCmd
          , Cmd.map ToLocationUpdate toLocationCmd
          , setMapMarkers initialModel
          ]


getStartSearchProfile : Model -> SearchOptions
getStartSearchProfile model =
    ModePicker.getSearchProfile model.fromModes


getDestinationSearchProfile : Model -> SearchOptions
getDestinationSearchProfile model =
    ModePicker.getSearchProfile model.toModes



-- UPDATE


type Msg
    = NoOp
    | FromLocationUpdate Typeahead.Msg
    | ToLocationUpdate Typeahead.Msg
    | FromModesUpdate ModePicker.Msg
    | ToModesUpdate ModePicker.Msg
    | DateUpdate Calendar.Msg
    | TimeUpdate TimeInput.Msg
    | SearchDirectionUpdate SearchDirection
    | SwitchInputs
    | ConnectionsUpdate Connections.Msg
    | SearchConnections
    | PrepareSelectConnection Int
    | StoreConnectionListScrollPos Msg Float
    | Deb (Debounce.Msg Msg)
    | SetRoutingResponses (List ( String, String ))
    | NavigateTo Route
    | ScheduleInfoError ApiError
    | ScheduleInfoResponse ScheduleInfo
    | SetSearchTime Date
    | ResetNew
    | SetLocale Localization
    | ShowOptions Bool


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            ( model, Cmd.none )

        FromLocationUpdate msg_ ->
            let
                ( m, c ) =
                    Typeahead.update msg_ model.fromLocation
            in
            { model | fromLocation = m }
                ! [ Cmd.map FromLocationUpdate c ]
                |> checkTypeaheadUpdate msg_

        ToLocationUpdate msg_ ->
            let
                ( m, c ) =
                    Typeahead.update msg_ model.toLocation
            in
            { model | toLocation = m }
                ! [ Cmd.map ToLocationUpdate c ]
                |> checkTypeaheadUpdate msg_

        FromModesUpdate msg_ ->
            { model | fromModes = ModePicker.update msg_ model.fromModes }
                ! []
                |> checkRoutingRequest

        ToModesUpdate msg_ ->
            { model | toModes = ModePicker.update msg_ model.toModes }
                ! []
                |> checkRoutingRequest

        DateUpdate msg_ ->
            { model | date = Calendar.update msg_ model.date }
                ! []
                |> checkRoutingRequest

        TimeUpdate msg_ ->
            { model | time = TimeInput.update msg_ model.time }
                ! []
                |> checkRoutingRequest

        SearchDirectionUpdate dir ->
            { model | searchDirection = dir }
                ! []
                |> checkRoutingRequest

        SwitchInputs ->
            { model
                | fromLocation = model.toLocation
                , toLocation = model.fromLocation
            }
                ! []
                |> checkRoutingRequest

        ConnectionsUpdate msg_ ->
            let
                ( m, c ) =
                    Connections.update msg_ model.connections
            in
            ( { model | connections = m }, Cmd.map ConnectionsUpdate c )

        SearchConnections ->
            let
                routingRequest =
                    buildRoutingRequest model

                fromName =
                    Typeahead.getSelectedSuggestion model.fromLocation
                        |> Maybe.map Typeahead.getShortSuggestionName

                toName =
                    Typeahead.getSelectedSuggestion model.toLocation
                        |> Maybe.map Typeahead.getShortSuggestionName

                searchReq =
                    Connections.Search
                        Connections.ReplaceResults
                        routingRequest
                        fromName
                        toName

                ( m, c ) =
                    Connections.update searchReq model.connections
            in
            { model
                | connections = m
                , currentRoutingRequest = Just routingRequest
                , connectionListScrollPos = 0
            }
                ! [ Cmd.map ConnectionsUpdate c ]

        PrepareSelectConnection idx ->
            let
                selectMsg =
                    NavigateTo (ConnectionDetails idx)
            in
            model
                ! [ Task.attempt
                        (\r ->
                            case r of
                                Ok pos ->
                                    StoreConnectionListScrollPos selectMsg pos

                                Err _ ->
                                    selectMsg
                        )
                        (Scroll.y "connections")
                  ]

        StoreConnectionListScrollPos msg_ pos ->
            let
                newModel =
                    { model | connectionListScrollPos = pos }
            in
            update msg_ newModel

        Deb a ->
            Debounce.update debounceCfg a model

        SetRoutingResponses files ->
            let
                parsed =
                    List.map
                        (\( name, json ) -> ( name, Decode.decodeString decodeRoutingResponse json ))
                        files

                valid =
                    List.filterMap
                        (\( name, result ) ->
                            case result of
                                Ok routingResponse ->
                                    Just ( name, routingResponse )

                                Err _ ->
                                    Nothing
                        )
                        parsed

                errors =
                    List.filterMap
                        (\( name, result ) ->
                            case result of
                                Err msg_ ->
                                    Just ( name, msg_ )

                                Ok _ ->
                                    Nothing
                        )
                        parsed
            in
            update
                (ConnectionsUpdate (Connections.SetRoutingResponses valid))
                model

        NavigateTo route ->
            model ! [ Navigation.newUrl (toUrl route) ]

        ScheduleInfoError err ->
            let
                ( connections_, _ ) =
                    Connections.update (Connections.SetError err) model.connections

                ( newConnections, c ) =
                    Connections.update (Connections.UpdateScheduleInfo Nothing) connections_

                newDate =
                    Calendar.update (Calendar.SetValidRange Nothing) model.date
            in
            { model
                | connections = newConnections
                , date = newDate
            }
                ! [ Cmd.map ConnectionsUpdate c ]

        ScheduleInfoResponse si ->
            let
                ( connections_, connCmd ) =
                    Connections.update (Connections.UpdateScheduleInfo (Just si)) model.connections

                newDate =
                    Calendar.update (Calendar.SetValidRange (Just ( si.begin, si.end ))) model.date
            in
            { model
                | connections = connections_
                , date = newDate
            }
                ! [ Cmd.map ConnectionsUpdate connCmd ]

        SetSearchTime newDate ->
            let
                ( model1, cmds1 ) =
                    update (DateUpdate (Calendar.InitDate True newDate)) model

                ( model2, cmds2 ) =
                    update (TimeUpdate (TimeInput.InitDate True newDate)) model1
            in
            model2 ! [ cmds1, cmds2 ]

        ResetNew ->
            let
                ( connectionsModel, connectionsCmd ) =
                    Connections.update Connections.ResetNew model.connections
            in
            { model | connections = connectionsModel } ! [ Cmd.map ConnectionsUpdate connectionsCmd ]

        SetLocale newLocale ->
            let
                date_ =
                    Calendar.update (Calendar.SetDateConfig newLocale.dateConfig) model.date
            in
            { model | date = date_ } ! []

        ShowOptions visible ->
            { model | optionsVisible = visible } ! []


buildRoutingRequest : Model -> IntermodalRoutingRequest
buildRoutingRequest model =
    let
        default =
            IntermodalStation (Station "" "" (Position 0 0))

        toIntermodalLocation typeahead =
            case Typeahead.getSelectedSuggestion typeahead of
                Just (Typeahead.StationSuggestion s) ->
                    IntermodalStation s

                Just (Typeahead.AddressSuggestion a) ->
                    IntermodalPosition a.pos

                Just (Typeahead.PositionSuggestion p) ->
                    IntermodalPosition p

                Nothing ->
                    default

        fromLocation =
            toIntermodalLocation model.fromLocation

        toLocation =
            toIntermodalLocation model.toLocation

        fromModes =
            ModePicker.getModes model.fromModes

        toModes =
            ModePicker.getModes model.toModes

        minConnectionCount =
            5
    in
    IntermodalRoutingRequest.initialRequest
        minConnectionCount
        fromLocation
        toLocation
        fromModes
        toModes
        (combineDateTime model.date.date model.time.date)
        model.searchDirection


isCompleteQuery : Model -> Bool
isCompleteQuery model =
    let
        fromLocation =
            Typeahead.getSelectedSuggestion model.fromLocation

        toLocation =
            Typeahead.getSelectedSuggestion model.toLocation
    in
    isJust fromLocation && isJust toLocation


checkRoutingRequest : ( Model, Cmd Msg ) -> ( Model, Cmd Msg )
checkRoutingRequest ( model, cmds ) =
    let
        completeQuery =
            isCompleteQuery model

        newRoutingRequest =
            buildRoutingRequest model

        requestChanged =
            case model.currentRoutingRequest of
                Just currentRequest ->
                    newRoutingRequest /= currentRequest

                Nothing ->
                    True

        modePickerVisible =
            model.fromModes.editorVisible || model.toModes.editorVisible

        fromLocation =
            Typeahead.saveSelection model.fromLocation

        toLocation =
            Typeahead.saveSelection model.toLocation

        fromModes =
            ModePicker.saveSelections model.fromModes

        toModes =
            ModePicker.saveSelections model.toModes
    in
    if completeQuery && requestChanged && not modePickerVisible then
        model
            ! [ cmds
              , Debounce.debounceCmd debounceCfg <| SearchConnections
              , Port.localStorageSet ("motis.routing.from_location" => fromLocation)
              , Port.localStorageSet ("motis.routing.to_location" => toLocation)
              , Port.localStorageSet ("motis.routing.from_modes" => fromModes)
              , Port.localStorageSet ("motis.routing.to_modes" => toModes)
              , getCombinedSearchProfile model
                    |> encodeSearchOptions
                    |> Port.setPPRSearchOptions
              ]

    else
        ( model, cmds )


getCombinedSearchProfile : Model -> SearchOptions
getCombinedSearchProfile model =
    let
        fromProfile =
            ModePicker.getSearchProfile model.fromModes

        toProfile =
            ModePicker.getSearchProfile model.toModes
    in
    case ( model.fromModes.walkEnabled, model.toModes.walkEnabled ) of
        ( True, False ) ->
            fromProfile

        ( False, True ) ->
            toProfile

        _ ->
            { fromProfile | duration_limit = fromProfile.duration_limit + toProfile.duration_limit }


checkTypeaheadUpdate : Typeahead.Msg -> ( Model, Cmd Msg ) -> ( Model, Cmd Msg )
checkTypeaheadUpdate msg ( model, cmds ) =
    let
        model_ =
            case msg of
                Typeahead.StationSuggestionsError err ->
                    let
                        ( m, _ ) =
                            Connections.update (Connections.SetError err) model.connections
                    in
                    { model | connections = m }

                _ ->
                    model
    in
    checkRoutingRequest ( model_, Cmd.batch [ cmds, setMapMarkers model_ ] )


setMapMarkers : Model -> Cmd msg
setMapMarkers model =
    let
        startPosition =
            Typeahead.getSelectedSuggestion model.fromLocation
                |> Maybe.map Typeahead.getSuggestionPosition

        destinationPosition =
            Typeahead.getSelectedSuggestion model.toLocation
                |> Maybe.map Typeahead.getSuggestionPosition

        startName =
            Typeahead.getSelectedSuggestion model.fromLocation
                |> Maybe.map Typeahead.getSuggestionName

        destinationName =
            Typeahead.getSelectedSuggestion model.toLocation
                |> Maybe.map Typeahead.getSuggestionName

    in
    RailViz.setMapMarkers startPosition destinationPosition startName destinationName


debounceCfg : Debounce.Config Model Msg
debounceCfg =
    Debounce.config
        .debounce
        (\model s -> { model | debounce = s })
        Deb
        700


noop : a -> Msg
noop =
    \_ -> NoOp


loadPprMode : String -> Maybe ModePicker.PprProfileMode
loadPprMode str =
    case str of
        "default" ->
            Just ModePicker.OnlyDefaultPprProfile

        "premade" ->
            Just ModePicker.PremadePprProfiles

        _ ->
            Nothing



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions model =
    Sub.batch
        [ Sub.map ConnectionsUpdate (Connections.subscriptions model.connections) ]



-- VIEW


view : Localization -> Model -> List (Html Msg)
view locale model =
    searchView locale model


searchView : Localization -> Model -> List (Html Msg)
searchView locale model =
    [ div [ id "search" ]
        [ div [ class "pure-g gutters" ]
            [ div [ class "pure-u-1 pure-u-sm-12-24 from-location" ]
                [ Html.map FromLocationUpdate <|
                    Typeahead.view 1 locale.t.search.start (Just "place") model.fromLocation
                , Html.map FromModesUpdate <|
                    ModePicker.view locale locale.t.search.startTransports model.fromModes
                , swapLocationsView model
                ]
            , div [ class "pure-u-1 pure-u-sm-12-24" ]
                [ Html.map DateUpdate <|
                    Calendar.view 3 locale.t.search.date model.date
                ]
            ]
        , div [ class "pure-g gutters" ]
            [ div [ class "pure-u-1 pure-u-sm-12-24 to-location" ]
                [ Html.map ToLocationUpdate <|
                    Typeahead.view 2 locale.t.search.destination (Just "place") model.toLocation
                , Html.map ToModesUpdate <|
                    ModePicker.view locale locale.t.search.destinationTransports model.toModes
                ]
            , div [ class "pure-u-1 pure-u-sm-9-24" ]
                [ Html.map TimeUpdate <|
                    TimeInput.view 4 locale.t.search.time model.time
                ]
            , div
                [ class "pure-u-1 pure-u-sm-3-24 time-option" ]
                (searchDirectionView locale model)
            ]
        ]
    , div [ id "connections" ]
        [ lazy3 Connections.view connectionConfig locale model.connections ]
    ]


searchDirectionView : Localization -> Model -> List (Html Msg)
searchDirectionView locale model =
    [ div []
        [ input
            [ type_ "radio"
            , id "search-forward"
            , name "time-option"
            , checked (model.searchDirection == Forward)
            , onClick (SearchDirectionUpdate Forward)
            ]
            []
        , label [ for "search-forward" ] [ text locale.t.search.departure ]
        ]
    , div []
        [ input
            [ type_ "radio"
            , id "search-backward"
            , name "time-option"
            , checked (model.searchDirection == Backward)
            , onClick (SearchDirectionUpdate Backward)
            ]
            []
        , label [ for "search-backward" ] [ text locale.t.search.arrival ]
        ]
    ]


swapLocationsView : Model -> Html Msg
swapLocationsView model =
    div
        [ class "swap-locations-btn" ]
        [ label
            [ class "gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select" ]
            [ input
                [ type_ "checkbox"
                , onClick SwitchInputs
                ]
                []
            , i [ class "icon" ] [ text "swap_vert" ]
            ]
        ]


connectionConfig : Connections.Config Msg
connectionConfig =
    Connections.Config
        { internalMsg = ConnectionsUpdate
        , selectMsg = PrepareSelectConnection
        }
