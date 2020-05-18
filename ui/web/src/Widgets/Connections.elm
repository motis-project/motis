module Widgets.Connections exposing
    ( Config(..)
    , Model
    , Msg(..)
    , SearchAction(..)
    , connectionIdxToListIdx
    , getJourney
    , init
    , subscriptions
    , update
    , updateJourney
    , updateJourneys
    , view
    )

import Data.Connection.Types as Connection exposing (Connection, Stop, hasNoProblems)
import Data.Intermodal.Request as IntermodalRoutingRequest
    exposing
        ( IntermodalLocation(..)
        , PretripSearchOptions
        , encodeRequest
        , getInterval
        , setInterval
        , setPretripSearchOptions
        )
import Data.Intermodal.Types exposing (IntermodalRoutingRequest)
import Data.Journey.Types as Journey exposing (Journey, Train)
import Data.Routing.Decode exposing (decodeRoutingResponse)
import Data.Routing.Types exposing (Interval, RoutingResponse, SearchDirection(..))
import Data.ScheduleInfo.Types as ScheduleInfo exposing (ScheduleInfo)
import Date exposing (Date)
import Date.Extra.Duration as Duration exposing (Duration(..))
import Html exposing (Html, a, div, i, text)
import Html.Attributes exposing (..)
import Html.Events exposing (onClick, onMouseEnter, onMouseLeave)
import Html.Keyed
import Html.Lazy exposing (..)
import List.Extra exposing (updateAt)
import Localization.Base exposing (..)
import Maybe.Extra exposing (isJust)
import Util.Api as Api exposing (ApiError)
import Util.Core exposing ((=>))
import Util.Date exposing (unixTime)
import Util.DateFormat exposing (..)
import Util.List exposing ((!!))
import Widgets.DateHeaders exposing (..)
import Widgets.Helpers.ApiErrorUtil exposing (errorText)
import Widgets.Helpers.ConnectionUtil exposing (..)
import Widgets.JourneyTransportGraph as JourneyTransportGraph
import Widgets.LoadingSpinner as LoadingSpinner
import Widgets.Map.Port exposing (MapTooltip, RVConnectionSegmentTrip, RVConnectionSegmentWalk, mapSetTooltip)
import Widgets.Map.RailViz as RailViz



-- MODEL


type alias Model =
    { loading : Bool
    , loadingBefore : Bool
    , loadingAfter : Bool
    , remoteAddress : String
    , journeys : List LabeledJourney
    , journeyTransportGraphs : List JourneyTransportGraph.Model
    , indexOffset : Int
    , errorMessage : Maybe ApiError
    , errorBefore : Maybe ApiError
    , errorAfter : Maybe ApiError
    , scheduleInfo : Maybe ScheduleInfo
    , routingRequest : Maybe IntermodalRoutingRequest
    , newJourneys : List Int
    , allowExtend : Bool
    , labels : List String
    , fromName : Maybe String
    , toName : Maybe String
    , lastRequestId : Int
    , hoveredTripSegments : Maybe ( List RVConnectionSegmentTrip )
    , hoveredWalkSegment : Maybe RVConnectionSegmentWalk
    }


type alias LabeledJourney =
    { journey : Journey
    , labels : List String
    }


type Config msg
    = Config
        { internalMsg : Msg -> msg
        , selectMsg : Int -> msg
        }


init : String -> Model
init remoteAddress =
    { loading = False
    , loadingBefore = False
    , loadingAfter = False
    , remoteAddress = remoteAddress
    , journeys = []
    , journeyTransportGraphs = []
    , indexOffset = 0
    , errorMessage = Nothing
    , errorBefore = Nothing
    , errorAfter = Nothing
    , scheduleInfo = Nothing
    , routingRequest = Nothing
    , newJourneys = []
    , allowExtend = True
    , labels = []
    , fromName = Nothing
    , toName = Nothing
    , lastRequestId = 0
    , hoveredTripSegments = Nothing
    , hoveredWalkSegment = Nothing
    }


connectionIdxToListIdx : Model -> Int -> Int
connectionIdxToListIdx model connectionIdx =
    connectionIdx - model.indexOffset


getJourney : Model -> Int -> Maybe Journey
getJourney model connectionIdx =
    model.journeys
        !! connectionIdxToListIdx model connectionIdx
        |> Maybe.map .journey


updateJourney : Model -> Int -> (Journey -> Journey) -> Model
updateJourney model connectionIdx f =
    let
        idx =
            connectionIdxToListIdx model connectionIdx

        updateLabeledJourney lj =
            { lj | journey = f lj.journey }

        journeys_ =
            List.Extra.updateAt idx updateLabeledJourney model.journeys
    in
    case journeys_ of
        Just j ->
            { model | journeys = j }

        Nothing ->
            model


updateJourneys : Model -> (Journey -> Journey) -> Model
updateJourneys model f =
    let
        updateLabeledJourney lj =
            { lj | journey = f lj.journey }

        journeys_ =
            List.map updateLabeledJourney model.journeys
    in
    { model | journeys = journeys_ }



-- UPDATE


type Msg
    = NoOp
    | Search SearchAction IntermodalRoutingRequest (Maybe String) (Maybe String)
    | ExtendSearchInterval ExtendIntervalType
    | ReceiveResponse SearchAction IntermodalRoutingRequest Int RoutingResponse
    | ReceiveError SearchAction IntermodalRoutingRequest Int ApiError
    | UpdateScheduleInfo (Maybe ScheduleInfo)
    | ResetNew
    | JTGUpdate Int JourneyTransportGraph.Msg
    | SetRoutingResponses (List ( String, RoutingResponse ))
    | SetError ApiError
    | MapSetTooltip MapTooltip
    | MouseEnterConnection Int Journey
    | MouseLeaveConnection Int Journey


type ExtendIntervalType
    = ExtendBefore
    | ExtendAfter


type SearchAction
    = ReplaceResults
    | PrependResults
    | AppendResults


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            model ! []

        Search action req fromName toName ->
            { model
                | loading = True
                , loadingBefore = False
                , loadingAfter = False
                , routingRequest = Just req
                , fromName = fromName
                , toName = toName
            }
                |> sendRequest ReplaceResults req

        ExtendSearchInterval direction ->
            case model.routingRequest of
                Just baseRequest ->
                    let
                        ( newRequest, updatedFullRequest ) =
                            extendSearchInterval direction baseRequest

                        action =
                            case direction of
                                ExtendBefore ->
                                    PrependResults

                                ExtendAfter ->
                                    AppendResults

                        loadingBefore_ =
                            case direction of
                                ExtendBefore ->
                                    True

                                ExtendAfter ->
                                    model.loadingBefore

                        loadingAfter_ =
                            case direction of
                                ExtendBefore ->
                                    model.loadingAfter

                                ExtendAfter ->
                                    True
                    in
                    { model
                        | routingRequest = Just updatedFullRequest
                        , loadingBefore = loadingBefore_
                        , loadingAfter = loadingAfter_
                    }
                        |> sendRequest action newRequest

                Nothing ->
                    model ! []

        ReceiveResponse action request requestId response ->
            if requestId == model.lastRequestId then
                let
                    model_ =
                        { model | allowExtend = True }
                in
                updateModelWithNewResults
                    model_
                    action
                    request
                    [ ( model.remoteAddress, response ) ]

            else
                model ! []

        ReceiveError action request requestId msg_ ->
            if requestId == model.lastRequestId then
                handleRequestError model action msg_ ! []

            else
                model ! []

        SetRoutingResponses responses ->
            let
                placeholderStation =
                    IntermodalStation
                        { id = ""
                        , name = ""
                        , pos = { lat = 0, lng = 0 }
                        }

                request =
                    IntermodalRoutingRequest.initialRequest
                        0
                        placeholderStation
                        placeholderStation
                        []
                        []
                        (Date.fromTime 0)
                        Forward

                model_ =
                    { model
                        | allowExtend = False
                        , fromName = Nothing
                        , toName = Nothing
                    }
            in
            updateModelWithNewResults model_ ReplaceResults request responses

        UpdateScheduleInfo si ->
            { model | scheduleInfo = si } ! []

        ResetNew ->
            { model
                | newJourneys = []
                , journeyTransportGraphs =
                    List.map
                        JourneyTransportGraph.hideTooltips
                        model.journeyTransportGraphs
            }
                ! []

        JTGUpdate idx msg_ ->
            { model
                | journeyTransportGraphs =
                    updateAt (connectionIdxToListIdx model idx)
                        (JourneyTransportGraph.update msg_)
                        model.journeyTransportGraphs
                        |> Maybe.withDefault model.journeyTransportGraphs
            }
                ! []

        SetError err ->
            handleRequestError model ReplaceResults err ! []

        MapSetTooltip tt ->
            { model
                | hoveredTripSegments = tt.hoveredTripSegments
                , hoveredWalkSegment = tt.hoveredWalkSegment
            }
                ! []

        MouseEnterConnection idx journey ->
            model ! [ RailViz.highlightConnections [ ( idx, journey ) ] ]

        MouseLeaveConnection idx journey ->
            model ! [ RailViz.highlightConnections [] ]


extendSearchInterval :
    ExtendIntervalType
    -> IntermodalRoutingRequest
    -> ( IntermodalRoutingRequest, IntermodalRoutingRequest )
extendSearchInterval direction base =
    let
        extendBy =
            3600 * 2

        minConnectionCount =
            3

        previousInterval =
            getInterval base

        newIntervalBegin =
            case direction of
                ExtendBefore ->
                    previousInterval.begin - extendBy

                ExtendAfter ->
                    previousInterval.begin

        newIntervalEnd =
            case direction of
                ExtendBefore ->
                    previousInterval.end

                ExtendAfter ->
                    previousInterval.end + extendBy

        newRequest =
            case direction of
                ExtendBefore ->
                    setPretripSearchOptions base
                        { interval =
                            { begin = newIntervalBegin
                            , end = previousInterval.begin - 1
                            }
                        , minConnectionCount = minConnectionCount
                        , extendIntervalEarlier = True
                        , extendIntervalLater = False
                        }

                ExtendAfter ->
                    setPretripSearchOptions base
                        { interval =
                            { begin = previousInterval.end + 1
                            , end = newIntervalEnd
                            }
                        , minConnectionCount = minConnectionCount
                        , extendIntervalEarlier = False
                        , extendIntervalLater = True
                        }

        updatedFullRequest =
            setInterval base
                { begin = newIntervalBegin
                , end = newIntervalEnd
                }
    in
    ( newRequest, updatedFullRequest )


updateModelWithNewResults :
    Model
    -> SearchAction
    -> IntermodalRoutingRequest
    -> List ( String, RoutingResponse )
    -> ( Model, Cmd Msg )
updateModelWithNewResults model action request responses =
    let
        firstResponse =
            List.head responses
                |> Maybe.map Tuple.second

        updateInterval updateStart updateEnd routingRequest =
            case firstResponse of
                Just response ->
                    let
                        previousInterval =
                            getInterval routingRequest
                    in
                    setInterval routingRequest
                        { begin =
                            if updateStart then
                                unixTime response.intervalStart

                            else
                                previousInterval.begin
                        , end =
                            if updateEnd then
                                unixTime response.intervalEnd

                            else
                                previousInterval.end
                        }

                Nothing ->
                    routingRequest

        base =
            case action of
                ReplaceResults ->
                    { model
                        | loading = False
                        , errorMessage = Nothing
                        , errorBefore = Nothing
                        , errorAfter = Nothing
                        , routingRequest =
                            Maybe.map (updateInterval True True) model.routingRequest
                    }

                PrependResults ->
                    { model
                        | loadingBefore = False
                        , errorBefore = Nothing
                        , routingRequest =
                            Maybe.map (updateInterval True False) model.routingRequest
                    }

                AppendResults ->
                    { model
                        | loadingAfter = False
                        , errorAfter = Nothing
                        , routingRequest =
                            Maybe.map (updateInterval False True) model.routingRequest
                    }

        journeysToAdd : List LabeledJourney
        journeysToAdd =
            responses
                |> patchConnections base
                |> toLabeledJourneys
                |> List.filter (\lj -> hasNoProblems lj.journey.connection)

        newJourneys =
            case action of
                ReplaceResults ->
                    journeysToAdd

                PrependResults ->
                    journeysToAdd ++ model.journeys

                AppendResults ->
                    model.journeys ++ journeysToAdd

        newIndexOffset =
            case action of
                ReplaceResults ->
                    0

                PrependResults ->
                    model.indexOffset - List.length journeysToAdd

                AppendResults ->
                    model.indexOffset

        newNewJourneys =
            case action of
                ReplaceResults ->
                    []

                PrependResults ->
                    List.range newIndexOffset (newIndexOffset + List.length journeysToAdd - 1)

                AppendResults ->
                    List.range (newIndexOffset + List.length model.journeys) (newIndexOffset + List.length newJourneys - 1)

        sortedJourneys =
            sortJourneys newJourneys

        journeyTransportGraphs =
            List.map
                (\lj -> JourneyTransportGraph.init transportListViewWidth lj.journey)
                sortedJourneys

        labelsToAdd =
            responses
                |> List.map Tuple.first
                |> List.Extra.unique

        newLabels =
            case action of
                ReplaceResults ->
                    labelsToAdd

                _ ->
                    List.Extra.unique (model.labels ++ labelsToAdd)

        cmd =
            sortedJourneys
                |> List.map .journey
                |> List.indexedMap (\i j -> ( newIndexOffset + i, j ))
                |> RailViz.setConnections
    in
    { base
        | journeys = sortedJourneys
        , journeyTransportGraphs = journeyTransportGraphs
        , indexOffset = newIndexOffset
        , newJourneys = newNewJourneys
        , labels = newLabels
    }
        ! [ cmd ]


patchConnection : Model -> Connection -> Connection
patchConnection model connection =
    let
        swapStartDest =
            model.routingRequest
                |> Maybe.map (\r -> r.searchDir == Backward)
                |> Maybe.withDefault False

        ( startName, destName ) =
            if swapStartDest then
                ( model.toName, model.fromName )

            else
                ( model.fromName, model.toName )

        intermodalStartName =
            startName
                |> Maybe.withDefault "START"

        intermodalDestName =
            destName
                |> Maybe.withDefault "END"

        replaceStationName station name =
            { station | name = name }

        patchStop stop =
            case stop.station.id of
                "START" ->
                    { stop
                        | station =
                            replaceStationName stop.station intermodalStartName
                    }

                "END" ->
                    { stop
                        | station =
                            replaceStationName stop.station intermodalDestName
                    }

                _ ->
                    stop
    in
    { connection | stops = List.map patchStop connection.stops }


patchConnections :
    Model
    -> List ( String, RoutingResponse )
    -> List ( String, RoutingResponse )
patchConnections model responses =
    let
        patchResponse response =
            { response
                | connections = List.map (patchConnection model) response.connections
            }
    in
    responses
        |> List.map (\( l, r ) -> ( l, patchResponse r ))


toLabeledJourneys : List ( String, RoutingResponse ) -> List LabeledJourney
toLabeledJourneys responses =
    let
        journeys : List ( String, Journey )
        journeys =
            List.concatMap
                (\( label, r ) ->
                    List.map
                        (\c -> ( label, Journey.toJourney c ))
                        r.connections
                )
                responses

        labelJourneys : ( String, Journey ) -> List LabeledJourney -> List LabeledJourney
        labelJourneys ( label, journey ) labeled =
            labeled
                |> List.Extra.findIndex (\lj -> lj.journey == journey)
                |> Maybe.andThen
                    (\idx ->
                        List.Extra.updateAt
                            idx
                            (\lj ->
                                { lj | labels = label :: lj.labels }
                            )
                            labeled
                    )
                |> Maybe.withDefault
                    ({ journey = journey, labels = [ label ] }
                        :: labeled
                    )
    in
    List.foldr labelJourneys [] journeys


sortJourneys : List LabeledJourney -> List LabeledJourney
sortJourneys journeys =
    let
        departureTime stops =
            stops
                |> List.head
                |> Maybe.andThen (.departure >> .schedule_time)
                |> Maybe.map Date.toTime
                |> Maybe.withDefault 0

        arrivalTime stops =
            stops
                |> List.Extra.last
                |> Maybe.andThen (.arrival >> .schedule_time)
                |> Maybe.map Date.toTime
                |> Maybe.withDefault 0

        sortKey lj =
            let
                stops =
                    lj.journey.connection.stops
            in
            ( departureTime stops, arrivalTime stops )
    in
    List.sortBy sortKey journeys


handleRequestError :
    Model
    -> SearchAction
    -> ApiError
    -> Model
handleRequestError model action msg =
    let
        newModel =
            case action of
                ReplaceResults ->
                    { model
                        | loading = False
                        , errorMessage = Just msg
                        , journeys = []
                        , journeyTransportGraphs = []
                    }

                PrependResults ->
                    { model
                        | loadingBefore = False
                        , errorBefore = Just msg
                    }

                AppendResults ->
                    { model
                        | loadingAfter = False
                        , errorAfter = Just msg
                    }
    in
    newModel



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions model =
    Sub.batch
        [ mapSetTooltip MapSetTooltip
        ]



-- VIEW


view : Config msg -> Localization -> Model -> Html msg
view config locale model =
    if model.loading then
        div [ class "loading" ] [ LoadingSpinner.view ]

    else if List.isEmpty model.journeys then
        case model.errorMessage of
            Just err ->
                errorView "main-error" locale model err

            Nothing ->
                div [ class "no-results" ]
                    [ if isJust model.routingRequest then
                        div [] [ text locale.t.connections.noResults ]

                      else
                        text ""
                    , scheduleRangeView locale model
                    ]

    else
        lazy3 connectionsView config locale model


connectionsView : Config msg -> Localization -> Model -> Html msg
connectionsView config locale model =
    div [ class "connections" ]
        [ extendIntervalButton ExtendBefore config locale model
        , connectionsWithDateHeaders config locale model
        , div [ class "divider footer" ] []
        , extendIntervalButton ExtendAfter config locale model
        ]


connectionsWithDateHeaders : Config msg -> Localization -> Model -> Html msg
connectionsWithDateHeaders config locale model =
    let
        getDate ( idx, labeledJourney, _ ) =
            Connection.departureTime labeledJourney.journey.connection
                |> Maybe.withDefault (Date.fromTime 0)

        renderConnection ( idx, journey, jtg ) =
            ( "connection-" ++ toString idx
            , connectionView config
                locale
                ( model.hoveredTripSegments, model.hoveredWalkSegment )
                model.labels
                idx
                (List.member idx model.newJourneys)
                journey
                jtg
            )

        renderDateHeader date =
            ( "header-" ++ toString (Date.toTime date), dateHeader locale date )

        elements =
            List.map3 (\a b c -> ( a, b, c ))
                (List.range
                    model.indexOffset
                    (model.indexOffset + List.length model.journeys - 1)
                )
                model.journeys
                model.journeyTransportGraphs
    in
    Html.Keyed.node "div"
        [ class "connection-list" ]
        (withDateHeaders getDate renderConnection renderDateHeader elements)


connectionView :
    Config msg
    -> Localization
    -> ( Maybe (List RVConnectionSegmentTrip), Maybe RVConnectionSegmentWalk )
    -> List String
    -> Int
    -> Bool
    -> LabeledJourney
    -> JourneyTransportGraph.Model
    -> Html msg
connectionView (Config { internalMsg, selectMsg }) locale ( hoveredTrips, hoveredWalk ) allLabels idx new labeledJourney jtg =
    let
        j =
            labeledJourney.journey

        renderedLabels =
            if List.length allLabels > 1 then
                labelsView allLabels labeledJourney.labels

            else
                text ""

        isHighlighted =
            journeyIsHighlighted idx hoveredTrips hoveredWalk

        useHighlighting =
            isJust hoveredTrips || isJust hoveredWalk

        isFaded =
            not isHighlighted && useHighlighting
    in
    div
        [ classList
            [ "connection" => True
            , "new" => new
            , "highlighted" => isHighlighted
            , "faded" => isFaded
            ]
        , onClick (selectMsg idx)
        , onMouseEnter (internalMsg (MouseEnterConnection idx j))
        , onMouseLeave (internalMsg (MouseLeaveConnection idx j))
        ]
        [ renderedLabels
        , div [ class "pure-g" ]
            [ div [ class "pure-u-4-24 connection-times" ]
                [ div [ class "connection-departure" ]
                    [ text (Maybe.map formatTime (Connection.departureTime j.connection) |> Maybe.withDefault "?")
                    , text " "
                    , Maybe.map delay (Connection.departureEvent j.connection) |> Maybe.withDefault (text "")
                    ]
                , div [ class "connection-arrival" ]
                    [ text (Maybe.map formatTime (Connection.arrivalTime j.connection) |> Maybe.withDefault "?")
                    , text " "
                    , Maybe.map delay (Connection.arrivalEvent j.connection) |> Maybe.withDefault (text "")
                    ]
                ]
            , div [ class "pure-u-4-24 connection-duration" ]
                [ div [] [ text (Maybe.map durationText (Connection.duration j.connection) |> Maybe.withDefault "?") ] ]
            , div [ class "pure-u-16-24 connection-trains" ]
                [ Html.map (\m -> internalMsg (JTGUpdate idx m)) <|
                    JourneyTransportGraph.view locale ( useHighlighting, hoveredTrips, hoveredWalk ) jtg
                ]
            ]
        ]


labelsView : List String -> List String -> Html msg
labelsView allLabels journeyLabels =
    let
        labelClass label =
            List.Extra.elemIndex label allLabels
                |> Maybe.withDefault 0
                |> toString

        labelView label =
            div
                [ class ("connection-label with-tooltip label-" ++ labelClass label)
                , attribute "data-tooltip" label
                ]
                []
    in
    div [ class "labels" ]
        (List.map labelView journeyLabels)


transportListViewWidth : Int
transportListViewWidth =
    335


scheduleRangeView : Localization -> Model -> Html msg
scheduleRangeView { t } { scheduleInfo } =
    case scheduleInfo of
        Just si ->
            let
                begin =
                    si.begin

                end =
                    Duration.add Hour -12 si.end
            in
            div [ class "schedule-range" ]
                [ text <| t.connections.scheduleRange begin end ]

        Nothing ->
            text ""


errorView : String -> Localization -> Model -> ApiError -> Html msg
errorView divClass locale model err =
    let
        errorMsg =
            errorText locale err
    in
    div [ class divClass ]
        [ div [] [ text errorMsg ]
        , scheduleRangeView locale model
        ]


extendIntervalButton :
    ExtendIntervalType
    -> Config msg
    -> Localization
    -> Model
    -> Html msg
extendIntervalButton direction (Config { internalMsg }) locale model =
    let
        enabled =
            model.allowExtend
                && (case direction of
                        ExtendBefore ->
                            not model.loadingBefore

                        ExtendAfter ->
                            not model.loadingAfter
                   )

        divClass =
            case direction of
                ExtendBefore ->
                    "search-before"

                ExtendAfter ->
                    "search-after"

        title =
            case direction of
                ExtendBefore ->
                    locale.t.connections.extendBefore

                ExtendAfter ->
                    locale.t.connections.extendAfter

        clickHandler =
            if enabled then
                internalMsg <| ExtendSearchInterval direction

            else
                internalMsg NoOp

        err =
            case direction of
                ExtendBefore ->
                    model.errorBefore

                ExtendAfter ->
                    model.errorAfter
    in
    div
        [ classList
            [ "extend-search-interval" => True
            , divClass => True
            , "disabled" => not enabled
            , "error" => (enabled && isJust err)
            ]
        ]
        [ if enabled then
            case err of
                Nothing ->
                    a
                        [ onClick clickHandler ]
                        [ text title ]

                Just error ->
                    errorView "error" locale model error

          else if model.allowExtend then
            LoadingSpinner.view

          else
            text ""
        ]


journeyIsHighlighted : Int -> Maybe ( List RVConnectionSegmentTrip ) -> Maybe RVConnectionSegmentWalk -> Bool
journeyIsHighlighted idx mbTripSegs mbWalkSeg =
    let
        checkTrip tripSeg =
            tripSeg
                |> List.any (\t -> List.member idx t.connectionIds)

        checkWalk walkSeg =
            List.member idx walkSeg.connectionIds

        containsTrip =
            mbTripSegs
                |> Maybe.map checkTrip
                |> Maybe.withDefault False

        containsWalk =
            mbWalkSeg
                |> Maybe.map checkWalk
                |> Maybe.withDefault False
    in
    containsTrip || containsWalk



-- ROUTING REQUEST


sendRequest : SearchAction -> IntermodalRoutingRequest -> Model -> ( Model, Cmd Msg )
sendRequest action request model =
    let
        requestId =
            model.lastRequestId + 1

        model_ =
            { model | lastRequestId = requestId }
    in
    model_
        ! [ Api.sendRequest
                (model.remoteAddress ++ "?elm=IntermodalConnectionRequest")
                decodeRoutingResponse
                (ReceiveError action request requestId)
                (ReceiveResponse action request requestId)
                (encodeRequest request)
          ]
