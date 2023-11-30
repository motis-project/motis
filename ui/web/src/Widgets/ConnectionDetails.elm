module Widgets.ConnectionDetails exposing
    ( Config(..)
    , Msg(SetApiError, SetJourney)
    , State
    , getJourney
    , getTripId
    , init
    , update
    , view
    )

import Data.Connection.Types as Connection exposing (..)
import Data.Journey.Types as Journey exposing (..)
import Date exposing (Date)
import Date.Extra.Duration as Duration exposing (DeltaRecord)
import Html exposing (Html, div, i, span, text)
import Html.Attributes exposing (..)
import Html.Events exposing (onClick)
import Localization.Base exposing (..)
import String
import Util.Api exposing (ApiError)
import Util.Core exposing ((=>))
import Util.DateFormat exposing (..)
import Util.List exposing (..)
import Widgets.Helpers.ApiErrorUtil exposing (errorText)
import Widgets.Helpers.ConnectionUtil exposing (..)
import Widgets.LoadingSpinner as LoadingSpinner



-- MODEL


type alias State =
    { journey : Maybe Journey
    , expanded : List Bool
    , inSubOverlay : Bool
    , tripId : Maybe TripId
    , apiError : Maybe ApiError
    }


type Config msg
    = Config
        { internalMsg : Msg -> msg
        , selectTripMsg : Int -> msg
        , selectStationMsg : Station -> Maybe Date -> msg
        , selectWalkMsg : JourneyWalk -> msg
        , goBackMsg : msg
        }


init : Bool -> Bool -> Maybe TripId -> Maybe Journey -> State
init expanded inSubOverlay tripId journey =
    { journey = journey
    , expanded =
        journey
            |> Maybe.map (\j -> List.repeat (List.length j.trains) expanded)
            |> Maybe.withDefault []
    , inSubOverlay = inSubOverlay
    , tripId = tripId
    , apiError = Nothing
    }


type Msg
    = ToggleExpand Int
    | SetJourney Journey Bool
    | SetApiError ApiError


update : Msg -> State -> ( State, Cmd Msg )
update msg model =
    case msg of
        ToggleExpand idx ->
            { model | expanded = toggle model.expanded idx } ! []

        SetJourney journey expanded ->
            { model
                | journey = Just journey
                , expanded = List.repeat (List.length journey.trains) expanded
                , apiError = Nothing
            }
                ! []

        SetApiError error ->
            { model | apiError = Just error } ! []


toggle : List Bool -> Int -> List Bool
toggle list idx =
    case list of
        x :: xs ->
            if idx == 0 then
                not x :: xs

            else
                x :: toggle xs (idx - 1)

        [] ->
            []


getJourney : State -> Maybe Journey
getJourney state =
    state.journey


getTripId : State -> Maybe TripId
getTripId state =
    state.tripId



-- VIEW


view : Config msg -> Localization -> Date -> State -> Html msg
view config locale currentTime state =
    let
        content =
            case state.journey of
                Just journey ->
                    journeyView config locale currentTime state journey

                Nothing ->
                    emptyView config locale state

        isTripView =
            Maybe.map .isSingleCompleteTrip state.journey
                |> Maybe.withDefault True
    in
    div
        [ classList
            [ "connection-details" => True
            , "trip-view" => isTripView
            ]
        ]
        content


emptyView : Config msg -> Localization -> State -> List (Html msg)
emptyView (Config { internalMsg, selectTripMsg, selectStationMsg, goBackMsg }) locale state =
    let
        cjId =
            if state.inSubOverlay then
                "sub-connection-journey"

            else
                "connection-journey"

        content =
            case state.apiError of
                Just error ->
                    errorView locale error

                Nothing ->
                    div [ class "loading" ] [ LoadingSpinner.view ]
    in
    [ div [ class "connection-info" ]
        [ div [ class "header" ]
            [ div [ onClick goBackMsg, class "back" ]
                [ i [ class "icon" ] [ text "arrow_back" ] ]
            ]
        ]
    , div [ class "connection-journey empty", id cjId ]
        [ content ]
    ]


errorView : Localization -> ApiError -> Html msg
errorView locale err =
    let
        errorMsg =
            errorText locale err
    in
    div [ class "main-error" ]
        [ div [] [ text errorMsg ] ]


journeyView : Config msg -> Localization -> Date -> State -> Journey -> List (Html msg)
journeyView (Config { internalMsg, selectTripMsg, selectStationMsg, selectWalkMsg, goBackMsg }) locale currentTime state journey =
    let
        trains =
            trainsWithInterchangeInfo journey.trains

        indices =
            List.range 0 (List.length trains - 1)

        indexedTrains =
            List.map3 (\( train, ic ) idx ex -> ( idx, train, ic, ex )) trains indices state.expanded

        showMove move =
            case move of
                TrainMove t ->
                    True

                WalkMove w ->
                    isMumoWalk w

        moves =
            journey.moves
                |> List.filter showMove

        getTrain train =
            indexedTrains
                |> List.filter (\( idx, t, ic, ex ) -> t == train)
                |> List.head

        moveView move =
            case move of
                TrainMove t ->
                    case getTrain t of
                        Just ( idx, train, ic, ex ) ->
                            trainDetail
                                journey.isSingleCompleteTrip
                                internalMsg
                                selectTripMsg
                                selectStationMsg
                                locale
                                currentTime
                                ( train, ic )
                                idx
                                ex

                        Nothing ->
                            let
                                _ =
                                    Debug.log "Connection Details: Train not found!"
                            in
                            div [] []

                WalkMove w ->
                    walkDetail selectStationMsg selectWalkMsg locale currentTime w

        transportsView =
            moves
                |> List.map moveView

        cjId =
            if state.inSubOverlay then
                "sub-connection-journey"

            else
                "connection-journey"
    in
    [ connectionInfoView goBackMsg locale state.inSubOverlay journey.connection
    , div [ class "connection-journey", id cjId ]
        transportsView
    ]


connectionInfoView : msg -> Localization -> Bool -> Connection -> Html msg
connectionInfoView goBackMsg locale inSubOverlay connection =
    let
        depTime =
            departureTime connection |> Maybe.withDefault (Date.fromTime 0)

        arrTime =
            arrivalTime connection |> Maybe.withDefault (Date.fromTime 0)

        depTimeText =
            formatTime depTime

        arrTimeText =
            formatTime arrTime

        dateText =
            formatDate locale.dateConfig depTime

    in
    div [ class "connection-info" ]
        [ div [ class "header" ]
            [ div [ onClick goBackMsg, class "back" ]
                [ i [ class "icon" ] [ text "arrow_back" ] ]
            , div [ class "details" ]
                [ div [ class "date" ] [ text dateText ]
                , div [ class "connection-times" ]
                    [ div [ class "times" ]
                        [ div [ class "connection-departure" ]
                            [ text depTimeText
                            ]
                        , div [ class "connection-arrival" ]
                            [ text arrTimeText
                            ]
                        ]
                    , div [ class "locations" ]
                        [ div [] [ text (Maybe.map (.station >> .name) (List.head connection.stops) |> Maybe.withDefault "?") ]
                        , div [] [ text (Maybe.map (.station >> .name) (last connection.stops) |> Maybe.withDefault "?") ]
                        ]
                    ]
                , div [ class "summary" ]
                    [ span [ class "duration" ]
                        [ i [ class "icon" ] [ text "schedule" ]
                        , text (Maybe.map durationText (duration connection) |> Maybe.withDefault "?")
                        ]
                    , span [ class "interchanges" ]
                        [ i [ class "icon" ] [ text "transfer_within_a_station" ]
                        , text <| locale.t.connections.interchanges (interchanges connection)
                        ]
                    ]
                ]
            ]
        ]


type StopViewType
    = CompactStopView
    | DetailedStopView


stopView :
    (Station -> Maybe Date -> msg)
    -> StopViewType
    -> EventType
    -> Localization
    -> Date
    -> Maybe Int
    -> Stop
    -> Html msg
stopView selectStationMsg stopViewType eventType locale currentTime progress stop =
    let
        event : EventInfo
        event =
            if eventType == Departure then
                stop.departure

            else
                stop.arrival

        timeView e =
            span [ class (timestampClass currentTime e) ]
                [ text (Maybe.map formatTime e.schedule_time |> Maybe.withDefault "?") ]

        timeCell =
            case stopViewType of
                CompactStopView ->
                    [ timeView event ]

                DetailedStopView ->
                    [ div [ class "arrival" ] [ timeView stop.arrival ]
                    , div [ class "departure" ] [ timeView stop.departure ]
                    ]

        delayCell =
            case stopViewType of
                CompactStopView ->
                    [ delay event ]

                DetailedStopView ->
                    [ div [ class "arrival" ] [ delay stop.arrival ]
                    , div [ class "departure" ] [ delay stop.departure ]
                    ]

        trackCell =
            if String.isEmpty event.track then
                text ""

            else
                div [ class "track" ]
                    [ span []
                        [ text (locale.t.connections.track ++ " " ++ event.track) ]
                    ]

        stopTimestampClass =
            timestampClass currentTime event

        timelines =
            case progress of
                Just percentage ->
                    [ div [ class "timeline train-color-border bg" ] []
                    , div
                        [ class "timeline train-color-border progress"
                        , style [ "height" => (toString percentage ++ "%") ]
                        ]
                        []
                    ]

                Nothing ->
                    [ div [ class "timeline train-color-border" ] [] ]

        stationAttrs =
            if isRealStation stop.station then
                [ onClick (selectStationMsg stop.station event.time) ]

            else
                [ class "virtual" ]

        stationName =
            if String.startsWith "VIA" stop.station.id then
                locale.t.connections.parking

            else
                stop.station.name
    in
    div [ class ("stop " ++ stopTimestampClass) ]
        (timelines
            ++ [ div [ class "time" ] timeCell
               , div [ class "delay" ] delayCell
               , div [ class "station" ]
                    [ span stationAttrs
                        [ text stationName ]
                    ]
               , trackCell
               ]
        )


isRealStation : Station -> Bool
isRealStation station =
    station.id /= "START" && station.id /= "END" && not (String.startsWith "VIA" station.id)


timestampClass : Date -> EventInfo -> String
timestampClass currentTime event =
    if eventIsInThePast currentTime event then
        "past"

    else
        "future"


timelineProgress : Date -> EventInfo -> EventInfo -> Int
timelineProgress currentTime firstEvent lastEvent =
    let
        departed =
            eventIsInThePast currentTime firstEvent

        arrived =
            eventIsInThePast currentTime lastEvent

        calcProgress firstTime lastTime =
            let
                total =
                    Date.toTime lastTime - Date.toTime firstTime

                elapsed =
                    Date.toTime currentTime - Date.toTime firstTime
            in
            round (elapsed / total * 100.0)
    in
    if arrived then
        100

    else if departed then
        Maybe.map2
            calcProgress
            (getEventTime firstEvent)
            (getEventTime lastEvent)
            |> Maybe.withDefault 0

    else
        0


trainTopLine : Localization -> ( Train, InterchangeInfo ) -> String
trainTopLine locale ( train, ic ) =
    case ic.previousArrival of
        Just pa ->
            let
                prevArrivalTime : Maybe Date
                prevArrivalTime =
                    pa.schedule_time

                departureTime : Maybe Date
                departureTime =
                    List.head train.stops |> Maybe.andThen (.departure >> .schedule_time)

                arrivalTime : Maybe Date
                arrivalTime =
                    List.head train.stops |> Maybe.andThen (.arrival >> .schedule_time)

                d : Maybe DeltaRecord
                d =
                    if ic.walk then
                        Maybe.map2 Duration.diff arrivalTime prevArrivalTime

                    else
                        Maybe.map2 Duration.diff departureTime prevArrivalTime

                dText =
                    Maybe.map durationText d |> Maybe.withDefault "?"

                icText =
                    if ic.walk then
                        locale.t.connections.walkDuration dText

                    else
                        locale.t.connections.interchangeDuration dText
            in
            if String.isEmpty pa.track then
                icText

            else
                locale.t.connections.arrivalTrack pa.track ++ ", " ++ icText

        Nothing ->
            ""


directionView : String -> String -> Html msg
directionView additionalClass direction =
    div [ class ("direction " ++ additionalClass) ]
        [ div [ class "timeline train-color-border" ] []
        , i [ class "icon" ] [ text "arrow_forward" ]
        , text direction
        ]


trainDetail :
    Bool
    -> (Msg -> msg)
    -> (Int -> msg)
    -> (Station -> Maybe Date -> msg)
    -> Localization
    -> Date
    -> ( Train, InterchangeInfo )
    -> Int
    -> Bool
    -> Html msg
trainDetail isTripView internalMsg selectTripMsg selectStationMsg locale currentTime ( train, ic ) idx expanded =
    let
        transport =
            List.head train.transports

        departureStop =
            List.head train.stops

        departureTrack =
            Maybe.map (.departure >> .track) departureStop |> Maybe.withDefault ""

        arrivalStop =
            last train.stops

        arrivalTrack =
            Maybe.map (.arrival >> .track) arrivalStop |> Maybe.withDefault ""

        intermediateStops =
            train.stops |> List.drop 1 |> dropEnd 1

        hasIntermediateStops =
            not (List.isEmpty intermediateStops)

        topLine =
            trainTopLine locale ( train, ic )

        duration =
            Maybe.map2 Duration.diff
                (arrivalStop |> Maybe.andThen (.arrival >> .schedule_time))
                (departureStop |> Maybe.andThen (.departure >> .schedule_time))

        durationStr =
            Maybe.map durationText duration |> Maybe.withDefault "?"

        expandIcon =
            if hasIntermediateStops then
                if expanded then
                    [ i [ class "icon" ] [ text "expand_more" ]
                    , i [ class "icon" ] [ text "expand_less" ]
                    ]

                else
                    [ i [ class "icon" ] [ text "expand_less" ]
                    , i [ class "icon" ] [ text "expand_more" ]
                    ]

            else
                []

        intermediateText =
            locale.t.connections.tripIntermediateStops (List.length intermediateStops)

        intermediateToggleOnClick =
            if hasIntermediateStops then
                [ onClick (internalMsg (ToggleExpand idx)) ]

            else
                []

        direction =
            case Maybe.map .direction transport of
                Just d ->
                    if String.isEmpty d then
                        Nothing

                    else
                        Just d

                Nothing ->
                    Nothing

        trainBoxOnClick =
            if isTripView then
                []

            else
                [ onClick (selectTripMsg idx) ]

        intermediateStopViewType =
            if isTripView then
                DetailedStopView

            else
                CompactStopView

        directionClass =
            departureStop
                |> Maybe.map .departure
                |> Maybe.map (timestampClass currentTime)
                |> Maybe.withDefault ""

        intermediateToggleClass =
            if expanded then
                departureStop
                    |> Maybe.map .departure
                    |> Maybe.map (timestampClass currentTime)
                    |> Maybe.withDefault ""

            else
                arrivalStop
                    |> Maybe.map .arrival
                    |> Maybe.map (timestampClass currentTime)
                    |> Maybe.withDefault ""

        progressTimeline progress =
            [ div [ class "timeline train-color-border bg" ] []
            , div
                [ class "timeline train-color-border progress"
                , style [ "height" => (toString progress ++ "%") ]
                ]
                []
            ]

        intermediateProgress =
            if expanded then
                Maybe.map2
                    (timelineProgress currentTime)
                    (Maybe.map .departure departureStop)
                    (Maybe.map .arrival (List.head intermediateStops))

            else
                Maybe.map2
                    (timelineProgress currentTime)
                    (Maybe.map .departure departureStop)
                    (Maybe.map .arrival arrivalStop)

        timelines =
            case intermediateProgress of
                Just progress ->
                    progressTimeline progress

                Nothing ->
                    [ div [ class "timeline train-color-border" ] [] ]
    in
    case transport of
        Just t ->
            div
                [ class <| "train-detail train-class-" ++ toString t.class ]
                [ div [ class "top-border" ] []
                , div trainBoxOnClick
                    [ trainBox LongName locale t ]
                , if String.isEmpty topLine then
                    text ""

                  else
                    div [ class "train-top-line" ]
                        [ span [] [ text topLine ] ]
                , if String.isEmpty departureTrack then
                    text ""

                  else
                    div [ class "train-dep-track" ]
                        [ text <| locale.t.connections.track ++ " " ++ departureTrack ]
                , div [ class "first-stop" ]
                    [ Maybe.map
                        (stopView selectStationMsg CompactStopView Departure locale currentTime Nothing)
                        departureStop
                        |> Maybe.withDefault (text "")
                    ]
                , Maybe.map (directionView directionClass) direction
                    |> Maybe.withDefault (text "")
                , div
                    ([ classList
                        [ "intermediate-stops-toggle" => True
                        , "clickable" => hasIntermediateStops
                        , intermediateToggleClass => True
                        ]
                     ]
                        ++ intermediateToggleOnClick
                    )
                    [ div [ class "timeline-container" ] timelines
                    , div [ class "expand-icon" ] expandIcon
                    , span [] [ text (intermediateText ++ " (" ++ durationStr ++ ")") ]
                    ]
                , div
                    [ classList
                        [ "intermediate-stops" => True
                        , "expanded" => expanded
                        , "collapsed" => not expanded
                        ]
                    ]
                    (intermediateStopsWithProgress selectStationMsg intermediateStopViewType locale currentTime intermediateStops arrivalStop)
                , div [ class "last-stop" ]
                    [ Maybe.map
                        (stopView selectStationMsg CompactStopView Arrival locale currentTime Nothing)
                        arrivalStop
                        |> Maybe.withDefault (text "")
                    , if String.isEmpty arrivalTrack then
                        text ""

                      else
                        div [ class "train-arr-track" ]
                            [ text <| locale.t.connections.track ++ " " ++ arrivalTrack ]
                    ]
                ]

        Nothing ->
            text ""


intermediateStopsWithProgress :
    (Station -> Maybe Date -> msg)
    -> StopViewType
    -> Localization
    -> Date
    -> List Stop
    -> Maybe Stop
    -> List (Html msg)
intermediateStopsWithProgress selectStationMsg intermediateStopViewType locale currentTime intermediateStops maybeLastStop =
    case maybeLastStop of
        Just lastStop ->
            let
                render stop nextStop =
                    stopView
                        selectStationMsg
                        intermediateStopViewType
                        Departure
                        locale
                        currentTime
                        (Just (timelineProgress currentTime stop.departure nextStop.arrival))
                        stop

                f stop ( views, nextStop ) =
                    ( render stop nextStop :: views, stop )

                ( result, _ ) =
                    List.foldr f ( [], lastStop ) intermediateStops
            in
            result

        Nothing ->
            List.map
                (stopView selectStationMsg intermediateStopViewType Departure locale currentTime Nothing)
                intermediateStops


walkDetail : (Station -> Maybe Date -> msg) -> (JourneyWalk -> msg) -> Localization -> Date -> JourneyWalk -> Html msg
walkDetail selectStationMsg selectWalkMsg locale currentTime walk =
    let
        durationStr =
            durationText walk.duration

        progress =
            timelineProgress currentTime walk.from.departure walk.to.arrival

        txt =
            case walk.mumoType of
                "bike" ->
                    locale.t.connections.tripBike durationStr

                "car" ->
                    locale.t.connections.tripCar durationStr

                _ ->
                    locale.t.connections.tripWalk durationStr

        walkTypeClass =
            case walk.walkType of
                InitialWalk ->
                    "initial-walk"

                TrailingWalk ->
                    "trailing-walk"

                TransferWalk ->
                    "transfer-walk"
    in
    div [ class <| "train-detail train-class-walk " ++ walkTypeClass ] <|
        [ div [ class "top-border" ] []
        , div [ onClick (selectWalkMsg walk) ] [ walkBox walk.mumoType ]
        , div [ class "first-stop" ]
            [ stopView selectStationMsg CompactStopView Departure locale currentTime Nothing walk.from ]
        , div [ class "intermediate-stops-toggle" ]
            [ div [ class "timeline-container" ]
                [ div [ class "timeline train-color-border bg" ] []
                , div
                    [ class "timeline train-color-border progress"
                    , style [ "height" => (toString progress ++ "%") ]
                    ]
                    []
                ]
            , div [ class "expand-icon" ] []
            , span [] [ text <| txt ]
            ]
        , div [ class "last-stop" ]
            [ stopView selectStationMsg CompactStopView Arrival locale currentTime Nothing walk.to ]
        ]
