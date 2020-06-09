module Data.Journey.Types exposing
    ( EventType(..)
    , InterchangeInfo
    , Journey
    , JourneyMove(..)
    , JourneyWalk
    , Train
    , WalkType(..)
    , isMumoWalk
    , toJourney
    , trainsWithInterchangeInfo
    , walkFallbackPolyline
    )

import Data.Connection.Types exposing (..)
import Date.Extra.Duration as Duration
import Util.List exposing (..)


type alias Journey =
    { connection : Connection
    , trains : List Train
    , leadingWalk : Maybe JourneyWalk
    , trailingWalk : Maybe JourneyWalk
    , walks : List JourneyWalk
    , moves : List JourneyMove
    , isSingleCompleteTrip : Bool
    , accessibility : Int
    }


type alias Train =
    { stops : List Stop
    , transports : List TransportInfo
    , trip : Maybe TripId
    , from_idx : Int
    , to_idx : Int
    }


type alias JourneyWalk =
    { from : Stop
    , from_idx : Int
    , to : Stop
    , to_idx : Int
    , duration : Duration.DeltaRecord
    , mumoType : String
    , mumoId : Int
    , polyline : Maybe (List Float)
    , accessibility : Int
    , walkType : WalkType
    }


type WalkType
    = InitialWalk
    | TrailingWalk
    | TransferWalk


type JourneyMove
    = TrainMove Train
    | WalkMove JourneyWalk


type EventType
    = Departure
    | Arrival


type alias InterchangeInfo =
    { previousArrival : Maybe EventInfo
    , walk : Bool
    }


toJourney : Connection -> Journey
toJourney connection =
    let
        trains =
            groupTrains connection

        walks =
            extractWalks connection

        moves =
            getMoves trains walks
    in
    { connection = connection
    , trains = trains
    , leadingWalk = extractLeadingWalk connection
    , trailingWalk = extractTrailingWalk connection
    , walks = walks
    , moves = moves
    , isSingleCompleteTrip = False
    , accessibility = getAccessibility connection
    }


groupTrains : Connection -> List Train
groupTrains connection =
    let
        indexedStops : List ( Int, Stop )
        indexedStops =
            List.indexedMap (,) connection.stops

        add_stop : List Train -> Stop -> List Train
        add_stop trains stop =
            case List.head trains of
                Just train ->
                    { train | stops = stop :: train.stops }
                        :: (List.tail trains |> Maybe.withDefault [])

                Nothing ->
                    -- should not happen
                    Debug.log "groupTrains: add_stop empty list" []

        finish_train : List Train -> Int -> Int -> List Train
        finish_train trains start_idx end_idx =
            case List.head trains of
                Just train ->
                    let
                        transports =
                            transportsForRange connection start_idx end_idx

                        trip =
                            tripsForRange connection start_idx end_idx
                                |> List.head
                    in
                    { train
                        | transports = transports
                        , trip = trip
                        , from_idx = start_idx
                        , to_idx = end_idx
                    }
                        :: (List.tail trains |> Maybe.withDefault [])

                Nothing ->
                    -- should not happen
                    Debug.log "groupTrains: finish_train empty list" []

        group : ( Int, Stop ) -> ( List Train, Bool, Int ) -> ( List Train, Bool, Int )
        group ( idx, stop ) ( trains, in_train, end_idx ) =
            let
                ( trains_, in_train_, end_idx_ ) =
                    if stop.enter then
                        ( finish_train (add_stop trains stop) idx end_idx, False, -1 )

                    else
                        ( trains, in_train, end_idx )
            in
            if stop.exit then
                ( add_stop (Train [] [] Nothing -1 -1 :: trains_) stop, True, idx )

            else if in_train_ then
                ( add_stop trains_ stop, in_train_, end_idx_ )

            else
                ( trains_, in_train_, end_idx_ )

        ( trains, _, _ ) =
            List.foldr group ( [], False, -1 ) indexedStops
    in
    trains


extractLeadingWalk : Connection -> Maybe JourneyWalk
extractLeadingWalk connection =
    getWalkFrom 0 connection |> Maybe.andThen (toJourneyWalk connection)


extractTrailingWalk : Connection -> Maybe JourneyWalk
extractTrailingWalk connection =
    let
        lastStopIdx =
            List.length connection.stops - 1
    in
    getWalkTo lastStopIdx connection |> Maybe.andThen (toJourneyWalk connection)


extractWalks : Connection -> List JourneyWalk
extractWalks connection =
    let
        convertWalk : Move -> Maybe JourneyWalk
        convertWalk move =
            case move of
                Walk walk ->
                    toJourneyWalk connection walk

                Transport _ ->
                    Nothing
    in
    List.filterMap convertWalk connection.transports



-- TODO: combine adjacent walks


getWalk : (WalkInfo -> Bool) -> Connection -> Maybe WalkInfo
getWalk filter connection =
    let
        checkMove : Move -> Maybe WalkInfo
        checkMove move =
            case move of
                Walk walk ->
                    if filter walk then
                        Just walk

                    else
                        Nothing

                Transport _ ->
                    Nothing
    in
    List.filterMap checkMove connection.transports |> List.head


getWalkTo : Int -> Connection -> Maybe WalkInfo
getWalkTo to =
    getWalk (\w -> w.range.to == to)


getWalkFrom : Int -> Connection -> Maybe WalkInfo
getWalkFrom to =
    getWalk (\w -> w.range.from == to)


toJourneyWalk : Connection -> WalkInfo -> Maybe JourneyWalk
toJourneyWalk connection walkInfo =
    let
        fromStop =
            connection.stops !! walkInfo.range.from

        toStop =
            connection.stops !! walkInfo.range.to

        getWalkDuration from to =
            Maybe.map2
                Duration.diff
                to.arrival.schedule_time
                from.departure.schedule_time
                |> Maybe.withDefault Duration.zeroDelta

        makeJourneyWalk from to =
            { from = from
            , from_idx = walkInfo.range.from
            , to = to
            , to_idx = walkInfo.range.to
            , duration = getWalkDuration from to
            , mumoType = walkInfo.mumo_type
            , mumoId = walkInfo.mumo_id
            , polyline = Nothing
            , accessibility = walkInfo.accessibility
            , walkType = getWalkType connection walkInfo
            }
    in
    Maybe.map2 makeJourneyWalk fromStop toStop


getWalkType : Connection -> WalkInfo -> WalkType
getWalkType connection walkInfo =
    let
        extractTransportInfo move =
            case move of
                Transport ti ->
                    Just ti

                Walk _ ->
                    Nothing

        transports =
            connection.transports
                |> List.filterMap extractTransportInfo

        trainsBefore =
            transports
                |> List.filter (\t -> t.range.to <= walkInfo.range.from)
                |> List.length

        trainsAfter =
            transports
                |> List.filter (\t -> t.range.from >= walkInfo.range.to)
                |> List.length
    in
    case ( trainsBefore, trainsAfter ) of
        ( 0, _ ) ->
            InitialWalk

        ( _, 0 ) ->
            TrailingWalk

        _ ->
            TransferWalk


getMoves : List Train -> List JourneyWalk -> List JourneyMove
getMoves trains walks =
    let
        trainMoves =
            List.map TrainMove trains

        walkMoves =
            List.map WalkMove walks

        getFrom move =
            case move of
                TrainMove t ->
                    t.from_idx

                WalkMove w ->
                    w.from_idx
    in
    trainMoves
        ++ walkMoves
        |> List.sortBy getFrom


isMumoWalk : JourneyWalk -> Bool
isMumoWalk walk =
    walk.mumoType /= ""


trainsWithInterchangeInfo : List Train -> List ( Train, InterchangeInfo )
trainsWithInterchangeInfo trains =
    let
        arrival : Train -> Maybe EventInfo
        arrival train =
            Maybe.map .arrival (last train.stops)

        hasWalk : Train -> Train -> Bool
        hasWalk from to =
            let
                arrivalStation =
                    Maybe.map .station (last from.stops)

                departureStation =
                    Maybe.map .station (List.head to.stops)
            in
            arrivalStation /= departureStation

        foldTrains : Train -> List ( Train, InterchangeInfo ) -> List ( Train, InterchangeInfo )
        foldTrains train list =
            case last list of
                Just ( lastTrain, _ ) ->
                    list
                        ++ [ ( train
                             , { previousArrival = arrival lastTrain
                               , walk = hasWalk lastTrain train
                               }
                             )
                           ]

                Nothing ->
                    [ ( train
                      , { previousArrival = Nothing
                        , walk = False
                        }
                      )
                    ]
    in
    List.foldl foldTrains [] trains


walkFallbackPolyline : JourneyWalk -> List Float
walkFallbackPolyline walk =
    [ walk.from.station.pos.lat
    , walk.from.station.pos.lng
    , walk.to.station.pos.lat
    , walk.to.station.pos.lng
    ]


getAccessibility : Connection -> Int
getAccessibility connection =
    let
        accessibility move =
            case move of
                Walk walk ->
                    walk.accessibility

                Transport _ ->
                    0
    in
    connection.transports
        |> List.map accessibility
        |> List.sum
