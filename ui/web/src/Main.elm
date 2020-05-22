module Main exposing
    ( Model
    , Msg(..)
    , SubView(..)
    , closeSelectedConnection
    , closeSubOverlay
    , connectionDetailsConfig
    , connectionDetailsView
    , getBaseUrl
    , getCurrentDate
    , getCurrentTime
    , getLocale
    , getPPRRoute
    , getPermalink
    , init
    , journeyTrips
    , loadTripById
    , locationToMsg
    , main
    , noop
    , overlayView
    , requestScheduleInfo
    , requestWalkRoutes
    , routeToMsg
    , selectConnection
    , selectConnectionTrip
    , sendFootRoutingRequest
    , sendOSRMViaRouteRequest
    , sendTripRequest
    , setFullTripConnection
    , setFullTripError
    , setPPRWalkRoute
    , setWalkRoute
    , simTimePickerView
    , stationConfig
    , stationSearchView
    , stationView
    , subscriptions
    , tripDetailsConfig
    , tripDetailsView
    , tripSearchConfig
    , tripSearchView
    , update
    , view
    )

import Data.Connection.Types exposing (Connection, Position, Station, TripId)
import Data.Journey.Types exposing (Journey, JourneyWalk, toJourney, walkFallbackPolyline)
import Data.Lookup.Decode exposing (decodeTripToConnectionResponse)
import Data.Lookup.Request exposing (encodeTripToConnection)
import Data.OSRM.Decode exposing (decodeOSRMViaRouteResponse)
import Data.OSRM.Request exposing (encodeOSRMViaRouteRequest)
import Data.OSRM.Types exposing (..)
import Data.PPR.Decode exposing (decodeFootRoutingResponse)
import Data.PPR.Request exposing (encodeFootRoutingRequest)
import Data.PPR.Types as PPR exposing (FootRoutingRequest, FootRoutingResponse, SearchOptions)
import Data.ScheduleInfo.Decode exposing (decodeScheduleInfoResponse)
import Data.ScheduleInfo.Request as ScheduleInfo
import Data.ScheduleInfo.Types exposing (ScheduleInfo)
import Date exposing (Date)
import Date.Extra.Compare
import Dom
import Dom.Scroll as Scroll
import Html exposing (..)
import Html.Attributes exposing (..)
import Html.Events exposing (..)
import Html.Lazy exposing (..)
import Http exposing (encodeUri)
import Json.Decode as Decode
import Json.Encode
import Localization.Base exposing (..)
import Localization.De exposing (..)
import Localization.En exposing (..)
import Maybe.Extra exposing (isJust, isNothing, orElse)
import Navigation exposing (Location)
import Port
import ProgramFlags exposing (..)
import Routes exposing (..)
import Task
import Time exposing (Time)
import UrlParser
import Util.Api as Api exposing (ApiError(..))
import Util.Core exposing ((=>))
import Util.Date exposing (combineDateTime, unixTime)
import Util.List exposing ((!!))
import Widgets.ConnectionDetails as ConnectionDetails
import Widgets.Connections as Connections
import Widgets.Map.Details as MapDetails
import Widgets.Map.RailViz as RailViz
import Widgets.Routing as Routing
import Widgets.SimTimePicker as SimTimePicker
import Widgets.StationEvents as StationEvents
import Widgets.TripSearch as TripSearch
import Widgets.Typeahead as Typeahead


main : Program ProgramFlags Model Msg
main =
    Navigation.programWithFlags locationToMsg
        { init = init
        , view = view
        , update = update
        , subscriptions = subscriptions
        }



-- MODEL


type alias Model =
    { routing : Routing.Model
    , railViz : RailViz.Model
    , connectionDetails : Maybe ConnectionDetails.State
    , tripDetails : Maybe ConnectionDetails.State
    , stationEvents : Maybe StationEvents.Model
    , tripSearch : TripSearch.Model
    , subView : Maybe SubView
    , selectedConnectionIdx : Maybe Int
    , scheduleInfo : Maybe ScheduleInfo
    , locale : Localization
    , apiEndpoint : String
    , currentTime : Date
    , timeOffset : Float
    , overlayVisible : Bool
    , stationSearch : Typeahead.Model
    , programFlags : ProgramFlags
    , simTimePicker : SimTimePicker.Model
    , updateSearchTime : Bool
    }


type SubView
    = TripDetailsView
    | StationEventsView
    | TripSearchView


init : ProgramFlags -> Location -> ( Model, Cmd Msg )
init flags initialLocation =
    let
        locale =
            getLocale flags.language

        remoteAddress =
            flags.apiEndpoint

        ( routingModel, routingCmd ) =
            Routing.init flags locale

        ( mapModel, mapCmd ) =
            RailViz.init remoteAddress

        ( stationSearchModel, stationSearchCmd ) =
            Typeahead.init remoteAddress ""

        ( tripSearchModel, tripSearchCmd ) =
            TripSearch.init remoteAddress locale

        ( simTimePickerModel, simTimePickerCmd ) =
            SimTimePicker.init locale

        initialModel =
            { routing = routingModel
            , railViz = mapModel
            , connectionDetails = Nothing
            , tripDetails = Nothing
            , stationEvents = Nothing
            , tripSearch = tripSearchModel
            , subView = Nothing
            , selectedConnectionIdx = Nothing
            , scheduleInfo = Nothing
            , locale = locale
            , apiEndpoint = remoteAddress
            , currentTime = Date.fromTime flags.currentTime
            , timeOffset = 0
            , overlayVisible = True
            , stationSearch = stationSearchModel
            , programFlags = flags
            , simTimePicker = simTimePickerModel
            , updateSearchTime = isJust flags.simulationTime
            }

        ( model1, cmd1 ) =
            update (locationToMsg initialLocation) initialModel

        ( model2, cmd2 ) =
            case flags.simulationTime of
                Just time ->
                    update (SetSimulationTime time) model1

                Nothing ->
                    ( model1, Cmd.none )
    in
    model2
        ! [ RailViz.setLocale locale
          , Cmd.map RoutingUpdate routingCmd
          , Cmd.map MapUpdate mapCmd
          , Cmd.map StationSearchUpdate stationSearchCmd
          , Cmd.map TripSearchUpdate tripSearchCmd
          , Cmd.map SimTimePickerUpdate simTimePickerCmd
          , requestScheduleInfo remoteAddress
          , Task.perform UpdateCurrentTime Time.now
          , cmd1
          , cmd2
          ]



-- UPDATE


type Msg
    = NoOp
    | RoutingUpdate Routing.Msg
    | MapUpdate RailViz.Msg
    | SelectConnection Int
    | ConnectionDetailsUpdate ConnectionDetails.Msg
    | TripDetailsUpdate ConnectionDetails.Msg
    | ConnectionDetailsGoBack
    | TripDetailsGoBack
    | CloseConnectionDetails
    | PrepareSelectTrip Int
    | LoadTrip TripId
    | SelectTripId TripId
    | TripToConnectionError TripId ApiError
    | TripToConnectionResponse TripId Connection
    | ScheduleInfoError ApiError
    | ScheduleInfoResponse ScheduleInfo
    | SetLocale Localization
    | NavigateTo Route
    | ReplaceLocation Route
    | SetRoutingResponses (List ( String, String ))
    | UpdateCurrentTime Time
    | SetSimulationTime Time
    | SetTimeOffset Time Time
    | StationEventsUpdate StationEvents.Msg
    | PrepareSelectStation Station (Maybe Date)
    | SelectStation String (Maybe Date)
    | StationEventsGoBack
    | ShowStationDetails String
    | ToggleOverlay
    | CloseSubOverlay
    | StationSearchUpdate Typeahead.Msg
    | HandleRailVizError Json.Encode.Value
    | ClearRailVizError
    | TripSearchUpdate TripSearch.Msg
    | ShowTripSearch
    | ToggleTripSearch
    | HandleRailVizPermalink Float Float Float Float Float Date
    | SimTimePickerUpdate SimTimePicker.Msg
    | OSRMError Int JourneyWalk ApiError
    | OSRMResponse Int JourneyWalk OSRMViaRouteResponse
    | PPRError Int JourneyWalk ApiError
    | PPRResponse Int JourneyWalk FootRoutingResponse
    | SelectWalk JourneyWalk


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            ( model, Cmd.none )

        RoutingUpdate msg_ ->
            let
                ( routingModel, routingCmd ) =
                    Routing.update msg_ model.routing
            in
            { model | routing = routingModel } ! [ Cmd.map RoutingUpdate routingCmd ]

        MapUpdate msg_ ->
            let
                ( railViz_, railVizCmd ) =
                    RailViz.update msg_ model.railViz

                ( model1, cmd1 ) =
                    ( { model | railViz = railViz_ }, Cmd.map MapUpdate railVizCmd )

                ( model2, cmd2 ) =
                    case msg_ of
                        RailViz.ToggleSimTimePicker ->
                            update (SimTimePickerUpdate (SimTimePicker.Toggle (getCurrentDate model1) (model1.timeOffset /= 0))) model1

                        RailViz.MapContextMenuFromHere ->
                            update
                                (RoutingUpdate
                                    (Routing.FromLocationUpdate
                                        (Typeahead.setToPosition
                                            (RailViz.getContextMenuPosition model.railViz)
                                        )
                                    )
                                )
                                model1

                        RailViz.MapContextMenuToHere ->
                            update
                                (RoutingUpdate
                                    (Routing.ToLocationUpdate
                                        (Typeahead.setToPosition
                                            (RailViz.getContextMenuPosition model.railViz)
                                        )
                                    )
                                )
                                model1

                        _ ->
                            model1 ! []
            in
            model2 ! [ cmd1, cmd2 ]

        SelectConnection idx ->
            let
                ( m, c ) =
                    selectConnection model idx
            in
            m
                ! [ c
                  , Task.attempt noop <| Scroll.toTop "overlay-content"
                  , Task.attempt noop <| Scroll.toTop "connection-journey"
                  ]

        ConnectionDetailsUpdate msg_ ->
            let
                ( m, c ) =
                    case model.connectionDetails of
                        Just state ->
                            let
                                ( m_, c_ ) =
                                    ConnectionDetails.update msg_ state
                            in
                            ( Just m_, c_ )

                        Nothing ->
                            Nothing ! []
            in
            { model | connectionDetails = m }
                ! [ Cmd.map ConnectionDetailsUpdate c ]

        TripDetailsUpdate msg_ ->
            let
                ( m, c ) =
                    case model.tripDetails of
                        Just state ->
                            let
                                ( m_, c_ ) =
                                    ConnectionDetails.update msg_ state
                            in
                            ( Just m_, c_ )

                        Nothing ->
                            Nothing ! []
            in
            { model | tripDetails = m }
                ! [ Cmd.map TripDetailsUpdate c ]

        CloseConnectionDetails ->
            closeSelectedConnection model

        ConnectionDetailsGoBack ->
            update (NavigateTo Connections) model

        TripDetailsGoBack ->
            model ! [ Navigation.back 1 ]

        PrepareSelectTrip tripIdx ->
            selectConnectionTrip model tripIdx

        LoadTrip tripId ->
            loadTripById model tripId

        SelectTripId tripId ->
            update (NavigateTo (tripDetailsRoute tripId)) model

        TripToConnectionError tripId err ->
            setFullTripError model tripId err

        TripToConnectionResponse tripId connection ->
            setFullTripConnection model tripId connection

        ScheduleInfoError err ->
            let
                ( routingModel, routingCmd ) =
                    Routing.update (Routing.ScheduleInfoError err) model.routing
            in
            { model
                | scheduleInfo = Nothing
                , routing = routingModel
            }
                ! [ Cmd.map RoutingUpdate routingCmd ]

        ScheduleInfoResponse si ->
            let
                ( routingModel, routingCmd ) =
                    Routing.update (Routing.ScheduleInfoResponse si) model.routing

                ( newTripSearch, _ ) =
                    TripSearch.update (TripSearch.UpdateScheduleInfo si) model.tripSearch

                ( newSimTimePicker, _ ) =
                    SimTimePicker.update (SimTimePicker.UpdateScheduleInfo si) model.simTimePicker

                model1 =
                    { model
                        | scheduleInfo = Just si
                        , routing = routingModel
                        , tripSearch = newTripSearch
                        , simTimePicker = newSimTimePicker
                    }

                currentDate =
                    getCurrentDate model1

                currentTimeInSchedule =
                    Date.Extra.Compare.is3 Date.Extra.Compare.BetweenOpen currentDate si.begin si.end

                ( model2, cmd1 ) =
                    if currentTimeInSchedule then
                        ( model1, Cmd.none )

                    else
                        update
                            (SetSimulationTime (Date.toTime (combineDateTime si.begin currentDate)))
                            { model1 | updateSearchTime = True }
            in
            model2 ! [ Cmd.map RoutingUpdate routingCmd, cmd1 ]

        SetLocale newLocale ->
            let
                ( routing_, _ ) =
                    Routing.update (Routing.SetLocale newLocale) model.routing

                ( tripSearch_, _ ) =
                    TripSearch.update (TripSearch.SetLocale newLocale) model.tripSearch

                ( simTimePicker_, _ ) =
                    SimTimePicker.update (SimTimePicker.SetLocale newLocale) model.simTimePicker
            in
            { model
                | locale = newLocale
                , routing = routing_
                , tripSearch = tripSearch_
                , simTimePicker = simTimePicker_
            }
                ! [ RailViz.setLocale newLocale ]

        NavigateTo route ->
            model ! [ Navigation.newUrl (toUrl route) ]

        ReplaceLocation route ->
            model ! [ Navigation.modifyUrl (toUrl route) ]

        SetRoutingResponses files ->
            let
                ( routingModel, routingCmd ) =
                    Routing.update (Routing.SetRoutingResponses files) model.routing

                ( _, navigation ) =
                    update (NavigateTo Connections) model
            in
            { model | routing = routingModel }
                ! [ Cmd.map RoutingUpdate routingCmd
                  , navigation
                  ]

        UpdateCurrentTime time ->
            { model | currentTime = Date.fromTime time } ! []

        SetSimulationTime simulationTime ->
            model
                ! [ Task.perform (SetTimeOffset simulationTime) Time.now ]

        SetTimeOffset simulationTime currentTime ->
            let
                offset =
                    simulationTime - currentTime

                model1 =
                    { model
                        | currentTime = Date.fromTime currentTime
                        , timeOffset = offset
                        , updateSearchTime = False
                    }

                newDate =
                    getCurrentDate model1

                updateSearchTime =
                    model.updateSearchTime

                ( model2, cmds1 ) =
                    update (MapUpdate (RailViz.SetTimeOffset offset)) model1

                ( model3, cmds2 ) =
                    if updateSearchTime then
                        update (RoutingUpdate (Routing.SetSearchTime newDate)) model2

                    else
                        model2 ! []

                ( model4, cmds3 ) =
                    if updateSearchTime then
                        update (TripSearchUpdate (TripSearch.SetTime newDate)) model3

                    else
                        model3 ! []
            in
            model4
                ! [ cmds1
                  , cmds2
                  , cmds3
                  , Port.setTimeOffset offset
                  ]

        StationEventsUpdate msg_ ->
            let
                ( m, c ) =
                    case model.stationEvents of
                        Just state ->
                            let
                                ( m_, c_ ) =
                                    StationEvents.update msg_ state
                            in
                            ( Just m_, c_ )

                        Nothing ->
                            Nothing ! []
            in
            { model | stationEvents = m }
                ! [ Cmd.map StationEventsUpdate c ]

        PrepareSelectStation station maybeDate ->
            case maybeDate of
                Just date ->
                    update (NavigateTo (StationEventsAt station.id date)) model

                Nothing ->
                    update (NavigateTo (StationEvents station.id)) model

        SelectStation stationId maybeDate ->
            let
                ( model_, cmds_ ) =
                    closeSubOverlay model

                date =
                    Maybe.withDefault (getCurrentDate model_) maybeDate

                ( m, c ) =
                    StationEvents.init model_.apiEndpoint stationId date
            in
            { model_
                | stationEvents = Just m
                , subView = Just StationEventsView
                , overlayVisible = True
            }
                ! [ cmds_, Cmd.map StationEventsUpdate c ]

        StationEventsGoBack ->
            model ! [ Navigation.back 1 ]

        ShowStationDetails id ->
            update (NavigateTo (StationEvents id)) model

        ToggleOverlay ->
            { model | overlayVisible = not model.overlayVisible } ! []

        CloseSubOverlay ->
            let
                ( model1, cmds1 ) =
                    closeSubOverlay model

                route =
                    case model1.selectedConnectionIdx of
                        Nothing ->
                            Connections

                        Just idx ->
                            ConnectionDetails idx

                ( model2, cmds2 ) =
                    update (NavigateTo route) model1
            in
            model2 ! [ cmds1, cmds2 ]

        StationSearchUpdate msg_ ->
            let
                ( m, c1 ) =
                    Typeahead.update msg_ model.stationSearch

                c2 =
                    case msg_ of
                        Typeahead.ItemSelected ->
                            case Typeahead.getSelectedSuggestion m of
                                Just (Typeahead.StationSuggestion station) ->
                                    Navigation.newUrl (toUrl (StationEvents station.id))

                                Just (Typeahead.AddressSuggestion address) ->
                                    RailViz.flyTo address.pos Nothing Nothing Nothing True

                                Just (Typeahead.PositionSuggestion pos) ->
                                    RailViz.flyTo pos Nothing Nothing Nothing True

                                Nothing ->
                                    Cmd.none

                        _ ->
                            Cmd.none
            in
            { model | stationSearch = m } ! [ Cmd.map StationSearchUpdate c1, c2 ]

        HandleRailVizError json ->
            let
                apiError =
                    case Decode.decodeValue Api.decodeErrorResponse json of
                        Ok value ->
                            MotisError value

                        Err msg ->
                            case Decode.decodeValue Decode.string json of
                                Ok errType ->
                                    case errType of
                                        "NetworkError" ->
                                            NetworkError

                                        "TimeoutError" ->
                                            TimeoutError

                                        _ ->
                                            DecodeError errType

                                Err msg_ ->
                                    DecodeError msg_

                ( railVizModel, _ ) =
                    RailViz.update (RailViz.SetApiError (Just apiError)) model.railViz
            in
            { model | railViz = railVizModel } ! []

        ClearRailVizError ->
            let
                ( railVizModel, _ ) =
                    RailViz.update (RailViz.SetApiError Nothing) model.railViz
            in
            { model | railViz = railVizModel } ! []

        TripSearchUpdate msg_ ->
            let
                ( m, c ) =
                    TripSearch.update msg_ model.tripSearch
            in
            { model | tripSearch = m }
                ! [ Cmd.map TripSearchUpdate c ]

        ShowTripSearch ->
            let
                model1 =
                    { model
                        | subView = Just TripSearchView
                        , overlayVisible = True
                    }

                cmd1 = MapDetails.setDetailFilter Nothing
            in
            model1
                ! [ cmd1
                  , Task.attempt noop (Dom.focus "trip-search-trainnr-input")
                  ]

        ToggleTripSearch ->
            case model.subView of
                Just TripSearchView ->
                    update (NavigateTo Connections) model

                _ ->
                    update (NavigateTo TripSearchRoute) model

        HandleRailVizPermalink lat lng zoom bearing pitch date ->
            let
                ( model1, cmd1 ) =
                    closeSelectedConnection { model | overlayVisible = False }

                ( model2, cmd2 ) =
                    update (SetSimulationTime (Date.toTime date)) model1

                pos =
                    { lat = lat, lng = lng }

                cmd3 =
                    RailViz.flyTo pos (Just zoom) (Just bearing) (Just pitch) False
            in
            model2 ! [ cmd1, cmd2, cmd3 ]

        SimTimePickerUpdate msg_ ->
            let
                ( m, c ) =
                    SimTimePicker.update msg_ model.simTimePicker

                ( model1, cmd1 ) =
                    ( { model | simTimePicker = m }, Cmd.map SimTimePickerUpdate c )

                ( model2, cmd2 ) =
                    case msg_ of
                        SimTimePicker.SetSimulationTime ->
                            let
                                pickedTime =
                                    SimTimePicker.getSelectedSimTime model1.simTimePicker
                            in
                            update (SetSimulationTime pickedTime) model1

                        SimTimePicker.DisableSimMode ->
                            let
                                currentTime =
                                    Date.toTime model1.currentTime
                            in
                            update (SetTimeOffset currentTime currentTime) model1

                        _ ->
                            model1 ! []
            in
            model2 ! [ cmd1, cmd2 ]

        OSRMError journeyIdx walk err ->
            let
                _ =
                    Debug.log "OSRMError" ( journeyIdx, walk, err )

                fallbackPolyline =
                    { coordinates = walkFallbackPolyline walk }

                response =
                    { time = 0, distance = 0.0, polyline = fallbackPolyline }
            in
            setWalkRoute model journeyIdx walk response False

        OSRMResponse journeyIdx walk response ->
            setWalkRoute model journeyIdx walk response True

        PPRError journeyIdx walk err ->
            let
                _ =
                    Debug.log "PPRError" ( journeyIdx, walk, err )

                fallbackPolyline =
                    { coordinates = walkFallbackPolyline walk }

                response =
                    { time = 0, distance = 0.0, polyline = fallbackPolyline }
            in
            setWalkRoute model journeyIdx walk response False

        PPRResponse journeyIdx walk response ->
            setPPRWalkRoute model journeyIdx walk response

        SelectWalk walk ->
            let
                coords =
                    walk.polyline
                        |> Maybe.withDefault (walkFallbackPolyline walk)
            in
            model ! [ RailViz.fitBounds coords ]


noop : a -> Msg
noop =
    \_ -> NoOp


selectConnection : Model -> Int -> ( Model, Cmd Msg )
selectConnection model idx =
    let
        journey =
            Connections.getJourney model.routing.connections idx

        ( newRouting, _ ) =
            -- TODO: ???
            Routing.update Routing.ResetNew model.routing
    in
    case journey of
        Just j ->
            let
                trips =
                    journeyTrips j
            in
            { model
                | connectionDetails =
                    Maybe.map (ConnectionDetails.init False False Nothing) (Just journey)
                , routing = newRouting
                , selectedConnectionIdx = Just idx
                , tripDetails = Nothing
                , stationEvents = Nothing
                , subView = Nothing
            }
                ! [ MapDetails.setDetailFilter ( Just j )
                  , requestWalkRoutes model.apiEndpoint
                        (Routing.getStartSearchProfile model.routing)
                        (Routing.getDestinationSearchProfile model.routing)
                        j
                        idx
                  ]

        Nothing ->
            update (ReplaceLocation Connections) model


requestWalkRoutes : String -> SearchOptions -> SearchOptions -> Journey -> Int -> Cmd Msg
requestWalkRoutes remoteAddress spStart spDestination journey idx =
    let
        requestWalk ( walkIdx, walk ) =
            if isNothing walk.polyline then
                case walk.mumoType of
                    "bike" ->
                        Just (sendOSRMViaRouteRequest remoteAddress walk idx)

                    "car" ->
                        Just (sendOSRMViaRouteRequest remoteAddress walk idx)

                    _ ->
                        let
                            searchProfile =
                                if walkIdx > 0 then
                                    spDestination

                                else
                                    spStart
                        in
                        Just (sendFootRoutingRequest remoteAddress searchProfile walk idx)

            else
                Nothing
    in
    journey.walks
        |> List.indexedMap (,)
        |> List.filterMap requestWalk
        |> Cmd.batch


setWalkRoute : Model -> Int -> JourneyWalk -> OSRMViaRouteResponse -> Bool -> ( Model, Cmd Msg )
setWalkRoute model journeyIdx walk response updateAll =
    let
        updateJourney j =
            { j
                | walks = List.map updateWalk j.walks
                , leadingWalk = Maybe.map updateWalk j.leadingWalk
                , trailingWalk = Maybe.map updateWalk j.trailingWalk
            }

        updateWalk w =
            if isSameWalk w then
                { w | polyline = Just response.polyline.coordinates }

            else
                w

        isSameWalk w =
            (w.from.station == walk.from.station)
                && (w.to.station == walk.to.station)
                && (w.mumoType == walk.mumoType)

        routing =
            model.routing

        connections_ =
            if updateAll then
                Connections.updateJourneys routing.connections updateJourney

            else
                Connections.updateJourney routing.connections journeyIdx updateJourney

        routing_ =
            { routing | connections = connections_ }

        journey =
            Connections.getJourney connections_ journeyIdx

        journeyComplete =
            journey
                |> Maybe.map (\j -> List.all (.polyline >> isJust) j.walks)
                |> Maybe.withDefault False

        updateConnectionDetails : ConnectionDetails.State -> ConnectionDetails.State
        updateConnectionDetails connectionDetails =
            case model.selectedConnectionIdx of
                Just journeyIdx ->
                    { connectionDetails
                        | journey = journey
                    }

                _ ->
                    connectionDetails

        connectionDetails_ =
            model.connectionDetails
                |> Maybe.map updateConnectionDetails

        cmd =
            case model.selectedConnectionIdx of
                Just journeyIdx ->
                    if journeyComplete then
                        journey
                            |> Maybe.map MapDetails.updateWalks
                            |> Maybe.withDefault Cmd.none

                    else
                        Cmd.none

                _ ->
                    Cmd.none
    in
    { model
        | routing = routing_
        , connectionDetails = connectionDetails_
    }
        ! [ cmd ]


getPPRRoute : FootRoutingResponse -> JourneyWalk -> Maybe PPR.Route
getPPRRoute response walk =
    let
        targetDuration =
            walk.duration.minute + walk.duration.hour * 60

        compareRoutes a b =
            case compare (abs (a.duration - targetDuration)) (abs (b.duration - targetDuration)) of
                LT ->
                    LT

                GT ->
                    GT

                EQ ->
                    compare (abs (a.accessibility - walk.accessibility)) (abs (b.accessibility - walk.accessibility))
    in
    response.routes
        |> List.concatMap .routes
        |> List.sortWith compareRoutes
        |> List.head


setPPRWalkRoute : Model -> Int -> JourneyWalk -> FootRoutingResponse -> ( Model, Cmd Msg )
setPPRWalkRoute model journeyIdx walk response =
    let
        route =
            getPPRRoute response walk

        polyline =
            Maybe.map .path route
                |> Maybe.withDefault { coordinates = [] }

        osrmResponse =
            { time = 0, distance = 0.0, polyline = polyline }
    in
    setWalkRoute model journeyIdx walk osrmResponse False


selectConnectionTrip : Model -> Int -> ( Model, Cmd Msg )
selectConnectionTrip model tripIdx =
    let
        journey =
            model.selectedConnectionIdx
                |> Maybe.andThen (Connections.getJourney model.routing.connections)

        trip =
            journey
                |> Maybe.andThen (\j -> j.trains !! tripIdx)
                |> Maybe.andThen .trip
    in
    case trip of
        Just tripId ->
            update (NavigateTo (tripDetailsRoute tripId)) model

        Nothing ->
            model ! []


loadTripById : Model -> TripId -> ( Model, Cmd Msg )
loadTripById model tripId =
    let
        tripDetails =
            ConnectionDetails.init True True (Just tripId) Nothing
    in
    { model
        | stationEvents = Nothing
        , tripDetails = Just tripDetails
        , subView = Just TripDetailsView
        , overlayVisible = True
    }
        ! [ sendTripRequest model.apiEndpoint tripId
          , Task.attempt noop <| Scroll.toTop "sub-overlay-content"
          , Task.attempt noop <| Scroll.toTop "sub-connection-journey"
          ]


setFullTripConnection : Model -> TripId -> Connection -> ( Model, Cmd Msg )
setFullTripConnection model tripId connection =
    let
        journey =
            toJourney connection

        tripJourney =
            { journey
                | isSingleCompleteTrip = True
                , trains = journey.trains |> List.map (\t -> { t | trip = Just tripId })
                , moves =
                    journey.moves
                        |> List.map
                            (\m ->
                                case m of
                                    Data.Journey.Types.TrainMove t ->
                                        Data.Journey.Types.TrainMove { t | trip = Just tripId }

                                    Data.Journey.Types.WalkMove w ->
                                        Data.Journey.Types.WalkMove w
                            )
            }



        ( tripDetails, _ ) =
            case model.tripDetails of
                Just td ->
                    ConnectionDetails.update (ConnectionDetails.SetJourney tripJourney True) td

                Nothing ->
                    ConnectionDetails.init True True (Just tripId) (Just tripJourney) ! []
    in
    { model
        | tripDetails = Just tripDetails
        , subView = Just TripDetailsView
    }
        ! [ MapDetails.setDetailFilter ( Just tripJourney )
          , Task.attempt noop <| Scroll.toTop "sub-overlay-content"
          , Task.attempt noop <| Scroll.toTop "sub-connection-journey"
          ]


setFullTripError : Model -> TripId -> ApiError -> ( Model, Cmd Msg )
setFullTripError model tripId error =
    case model.tripDetails of
        Just td ->
            let
                ( tripDetails, _ ) =
                    ConnectionDetails.update (ConnectionDetails.SetApiError error) td
            in
            model ! [ MapDetails.setDetailFilter Nothing ]

        Nothing ->
            model ! []


getCurrentTime : Model -> Time
getCurrentTime model =
    Date.toTime model.currentTime + model.timeOffset


getCurrentDate : Model -> Date
getCurrentDate model =
    Date.fromTime (getCurrentTime model)


getLocale : String -> Localization
getLocale language =
    case String.toLower language of
        "de" ->
            deLocalization

        "en" ->
            enLocalization

        _ ->
            deLocalization



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions model =
    Sub.batch
        [ Sub.map RoutingUpdate (Routing.subscriptions model.routing)
        , Sub.map MapUpdate (RailViz.subscriptions model.railViz)
        , Port.setRoutingResponses SetRoutingResponses
        , Port.showStationDetails ShowStationDetails
        , Port.showTripDetails SelectTripId
        , Port.setSimulationTime SetSimulationTime
        , Port.handleRailVizError HandleRailVizError
        , Port.clearRailVizError (always ClearRailVizError)
        , Time.every (2 * Time.second) UpdateCurrentTime
        ]



-- VIEW


view : Model -> Html Msg
view model =
    let
        permalink =
            getPermalink model
    in
    div [ class "app" ] <|
        [ Html.map MapUpdate (RailViz.view model.locale permalink model.railViz)
        , lazy overlayView model
        , lazy stationSearchView model
        , lazy simTimePickerView model
        ]


overlayView : Model -> Html Msg
overlayView model =
    let
        mainOverlayContent =
            case model.connectionDetails of
                Nothing ->
                    Routing.view model.locale model.routing
                        |> List.map (Html.map RoutingUpdate)

                Just c ->
                    connectionDetailsView model.locale (getCurrentDate model) c

        subOverlayContent =
            case model.subView of
                Just TripDetailsView ->
                    Maybe.map (tripDetailsView model.locale (getCurrentDate model)) model.tripDetails

                Just StationEventsView ->
                    Maybe.map (stationView model.locale) model.stationEvents

                Just TripSearchView ->
                    Just (tripSearchView model.locale model.tripSearch)

                Nothing ->
                    Nothing

        subOverlay =
            subOverlayContent
                |> Maybe.map
                    (\c ->
                        div [ class "sub-overlay" ]
                            [ div [ id "sub-overlay-content" ] c
                            , div [ class "sub-overlay-close", onClick CloseSubOverlay ]
                                [ i [ class "icon" ] [ text "close" ] ]
                            ]
                    )
                |> Maybe.withDefault (div [ class "sub-overlay hidden" ] [ div [ id "sub-overlay-content" ] [] ])
    in
    div
        [ classList
            [ "overlay-container" => True
            , "hidden" => not model.overlayVisible
            ]
        ]
        [ div [ class "overlay" ]
            [ div [ id "overlay-content" ]
                mainOverlayContent
            , subOverlay
            ]
        , div [ class "overlay-tabs" ]
            [ div [ class "overlay-toggle", onClick ToggleOverlay ]
                [ i [ class "icon" ] [ text "arrow_drop_down" ] ]
            , div
                [ classList
                    [ "trip-search-toggle" => True
                    , "enabled" => (model.subView == Just TripSearchView)
                    ]
                , onClick ToggleTripSearch
                ]
                [ i [ class "icon" ] [ text "train" ] ]
            ]
        ]


stationSearchView : Model -> Html Msg
stationSearchView model =
    div
        [ id "station-search"
        , classList
            [ "overlay-hidden" => not model.overlayVisible
            ]
        ]
        [ Html.map StationSearchUpdate <|
            Typeahead.view 10 "" (Just "place") model.stationSearch
        ]


simTimePickerView : Model -> Html Msg
simTimePickerView model =
    div [ class "sim-time-picker-container" ]
        [ Html.map SimTimePickerUpdate <|
            SimTimePicker.view model.locale model.simTimePicker
        ]


connectionDetailsView : Localization -> Date -> ConnectionDetails.State -> List (Html Msg)
connectionDetailsView locale currentTime state =
    [ ConnectionDetails.view connectionDetailsConfig locale currentTime state ]


connectionDetailsConfig : ConnectionDetails.Config Msg
connectionDetailsConfig =
    ConnectionDetails.Config
        { internalMsg = ConnectionDetailsUpdate
        , selectTripMsg = PrepareSelectTrip
        , selectStationMsg = PrepareSelectStation
        , selectWalkMsg = SelectWalk
        , goBackMsg = ConnectionDetailsGoBack
        }


tripDetailsView : Localization -> Date -> ConnectionDetails.State -> List (Html Msg)
tripDetailsView locale currentTime state =
    [ ConnectionDetails.view tripDetailsConfig locale currentTime state ]


tripDetailsConfig : ConnectionDetails.Config Msg
tripDetailsConfig =
    ConnectionDetails.Config
        { internalMsg = TripDetailsUpdate
        , selectTripMsg = PrepareSelectTrip
        , selectStationMsg = PrepareSelectStation
        , selectWalkMsg = SelectWalk
        , goBackMsg = TripDetailsGoBack
        }


stationView : Localization -> StationEvents.Model -> List (Html Msg)
stationView locale model =
    [ StationEvents.view stationConfig locale model ]


stationConfig : StationEvents.Config Msg
stationConfig =
    StationEvents.Config
        { internalMsg = StationEventsUpdate
        , selectTripMsg = SelectTripId
        , goBackMsg = StationEventsGoBack
        }


tripSearchView : Localization -> TripSearch.Model -> List (Html Msg)
tripSearchView locale model =
    [ TripSearch.view tripSearchConfig locale model ]


tripSearchConfig : TripSearch.Config Msg
tripSearchConfig =
    TripSearch.Config
        { internalMsg = TripSearchUpdate
        , selectTripMsg = SelectTripId
        , selectStationMsg = PrepareSelectStation
        }


getPermalink : Model -> String
getPermalink model =
    let
        date =
            getCurrentDate model

        urlBase =
            getBaseUrl model.programFlags date
    in
    case model.subView of
        Just TripDetailsView ->
            case Maybe.andThen ConnectionDetails.getTripId model.tripDetails of
                Just tripId ->
                    urlBase
                        ++ toUrl
                            (TripDetails
                                tripId.station_id
                                tripId.train_nr
                                tripId.time
                                tripId.target_station_id
                                tripId.target_time
                                tripId.line_id
                            )

                Nothing ->
                    urlBase

        Just StationEventsView ->
            case model.stationEvents of
                Just stationEvents ->
                    toUrl
                        (StationEventsAt
                            (StationEvents.getStationId stationEvents)
                            date
                        )

                Nothing ->
                    urlBase

        Just TripSearchView ->
            urlBase ++ toUrl TripSearchRoute

        Nothing ->
            RailViz.getMapPermalink model.railViz


getBaseUrl : ProgramFlags -> Date -> String
getBaseUrl flags date =
    let
        timestamp =
            unixTime date

        params1 =
            [ "time" => toString timestamp ]

        params2 =
            case flags.langParam of
                Just lang ->
                    ("lang" => lang) :: params1

                Nothing ->
                    params1

        params3 =
            case flags.motisParam of
                Just motis ->
                    ("motis" => motis) :: params2

                Nothing ->
                    params2

        urlBase =
            params3
                |> List.map (\( k, v ) -> encodeUri k ++ "=" ++ encodeUri v)
                |> String.join "&"
                |> (\s -> "?" ++ s)
    in
    urlBase



-- REQUESTS


requestScheduleInfo : String -> Cmd Msg
requestScheduleInfo remoteAddress =
    Api.sendRequest
        (remoteAddress ++ "?elm=requestScheduleInfo")
        decodeScheduleInfoResponse
        ScheduleInfoError
        ScheduleInfoResponse
        ScheduleInfo.request


sendTripRequest : String -> TripId -> Cmd Msg
sendTripRequest remoteAddress tripId =
    Api.sendRequest
        (remoteAddress ++ "?elm=tripRequest")
        decodeTripToConnectionResponse
        (TripToConnectionError tripId)
        (TripToConnectionResponse tripId)
        (encodeTripToConnection tripId)


sendOSRMViaRouteRequest : String -> JourneyWalk -> Int -> Cmd Msg
sendOSRMViaRouteRequest remoteAddress walk journeyIdx =
    let
        request =
            { profile = osrmProfile walk
            , waypoints =
                [ walk.from.station.pos
                , walk.to.station.pos
                ]
            }

        osrmProfile walk =
            case walk.mumoType of
                "bike" ->
                    "bike"

                "walk" ->
                    "foot"

                "car" ->
                    "car"

                _ ->
                    "foot"
    in
    Api.sendRequest
        (remoteAddress ++ "?elm=OSRMViaRouteRequest")
        decodeOSRMViaRouteResponse
        (OSRMError journeyIdx walk)
        (OSRMResponse journeyIdx walk)
        (encodeOSRMViaRouteRequest request)


sendFootRoutingRequest : String -> SearchOptions -> JourneyWalk -> Int -> Cmd Msg
sendFootRoutingRequest remoteAddress searchProfile walk journeyIdx =
    let
        request =
            { start = walk.from.station.pos
            , destinations = [ walk.to.station.pos ]
            , search_options = searchProfile
            , include_steps = True
            , include_edges = False
            , include_path = True
            }
    in
    Api.sendRequest
        (remoteAddress ++ "?elm=FootRoutingRequest")
        decodeFootRoutingResponse
        (PPRError journeyIdx walk)
        (PPRResponse journeyIdx walk)
        (encodeFootRoutingRequest request)



-- NAVIGATION


locationToMsg : Location -> Msg
locationToMsg location =
    case UrlParser.parseHash urlParser location of
        Just route ->
            routeToMsg route

        Nothing ->
            ReplaceLocation Connections


routeToMsg : Route -> Msg
routeToMsg route =
    case route of
        Connections ->
            CloseConnectionDetails

        ConnectionDetails idx ->
            SelectConnection idx

        TripDetails station trainNr time targetStation targetTime lineId ->
            LoadTrip
                { station_id = station
                , train_nr = trainNr
                , time = time
                , target_station_id = targetStation
                , target_time = targetTime
                , line_id = lineId
                }

        StationEvents stationId ->
            SelectStation stationId Nothing

        StationEventsAt stationId date ->
            SelectStation stationId (Just date)

        SimulationTime time ->
            SetSimulationTime (Date.toTime time)

        TripSearchRoute ->
            ShowTripSearch

        RailVizPermalink lat lng zoom bearing pitch date ->
            HandleRailVizPermalink lat lng zoom bearing pitch date


closeSelectedConnection : Model -> ( Model, Cmd Msg )
closeSelectedConnection model =
    let
        cmds = MapDetails.setDetailFilter Nothing
    in
    { model
        | connectionDetails = Nothing
        , selectedConnectionIdx = Nothing
        , tripDetails = Nothing
        , stationEvents = Nothing
        , subView = Nothing
    }
        ! [ Task.attempt noop <| Scroll.toY "connections" model.routing.connectionListScrollPos
          , cmds
          ]


closeSubOverlay : Model -> ( Model, Cmd Msg )
closeSubOverlay model =
    let
        cmds = MapDetails.setDetailFilter Nothing
    in
    ( { model
        | tripDetails = Nothing
        , stationEvents = Nothing
        , subView = Nothing
      }
    , cmds
    )


journeyTrips : Journey -> List TripId
journeyTrips journey =
    journey.connection.trips
        |> List.map .id
