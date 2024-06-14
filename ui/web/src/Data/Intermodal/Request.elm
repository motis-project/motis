module Data.Intermodal.Request exposing
    ( IntermodalLocation(..)
    , PretripSearchOptions
    , destinationToIntermodalLocation
    , encodeIntermodalDestination
    , encodeIntermodalStart
    , encodeInterval
    , encodeMode
    , encodeRequest
    , fixStartDestination
    , getInterval
    , initialRequest
    , setInterval
    , setPretripSearchOptions
    , startToIntermodalLocation
    , swapStartDestination
    , toIntermodalDestination
    , toIntermodalStart
    )

import Data.Connection.Types exposing (Position, Station)
import Data.Intermodal.Types exposing (..)
import Data.PPR.Request exposing (encodeSearchOptions)
import Data.RailViz.Request exposing (encodePosition)
import Data.Routing.Request
    exposing
        ( encodeInputStation
        , encodeSearchDirection
        , encodeSearchType
        )
import Data.Routing.Types
    exposing
        ( Interval
        , SearchDirection(..)
        , SearchType(..)
        )
import Date exposing (Date)
import Json.Encode as Encode
import Util.Core exposing ((=>))
import Util.Date exposing (unixTime)


type IntermodalLocation
    = IntermodalStation Station
    | IntermodalPosition Position


type alias PretripSearchOptions =
    { interval : Interval
    , minConnectionCount : Int
    , extendIntervalEarlier : Bool
    , extendIntervalLater : Bool
    }


initialRequest :
    Int
    -> IntermodalLocation
    -> IntermodalLocation
    -> List Mode
    -> List Mode
    -> Date
    -> SearchDirection
    -> IntermodalRoutingRequest
initialRequest minConnectionCount from to startModes destModes date searchDirection =
    let
        selectedTime =
            unixTime date

        interval =
            { begin = selectedTime - 3600
            , end = selectedTime + 3600
            }

        options =
            { interval = interval
            , minConnectionCount = minConnectionCount
            , extendIntervalEarlier = True
            , extendIntervalLater = True
            }

        start =
            toIntermodalStart from options

        destination =
            toIntermodalDestination to
    in
    { start = start
    , startModes = startModes
    , destination = destination
    , destinationModes = destModes
    , searchType = AccessibilitySearchType
    , searchDir = searchDirection
    }


toIntermodalStart : IntermodalLocation -> PretripSearchOptions -> IntermodalStart
toIntermodalStart location options =
    case location of
        IntermodalStation s ->
            PretripStart
                { station = s
                , interval = options.interval
                , minConnectionCount = options.minConnectionCount
                , extendIntervalEarlier = options.extendIntervalEarlier
                , extendIntervalLater = options.extendIntervalLater
                }

        IntermodalPosition p ->
            IntermodalPretripStart
                { position = p
                , interval = options.interval
                , minConnectionCount = options.minConnectionCount
                , extendIntervalEarlier = options.extendIntervalEarlier
                , extendIntervalLater = options.extendIntervalLater
                }


toIntermodalDestination : IntermodalLocation -> IntermodalDestination
toIntermodalDestination location =
    case location of
        IntermodalStation s ->
            InputStation s

        IntermodalPosition p ->
            InputPosition p


startToIntermodalLocation : IntermodalStart -> IntermodalLocation
startToIntermodalLocation start =
    case start of
        PretripStart i ->
            IntermodalStation i.station

        IntermodalPretripStart i ->
            IntermodalPosition i.position


destinationToIntermodalLocation : IntermodalDestination -> IntermodalLocation
destinationToIntermodalLocation dest =
    case dest of
        InputStation s ->
            IntermodalStation s

        InputPosition p ->
            IntermodalPosition p


getInterval : IntermodalRoutingRequest -> Interval
getInterval req =
    case req.start of
        IntermodalPretripStart i ->
            i.interval

        PretripStart i ->
            i.interval


setInterval : IntermodalRoutingRequest -> Interval -> IntermodalRoutingRequest
setInterval req interval =
    let
        newStart =
            case req.start of
                IntermodalPretripStart i ->
                    IntermodalPretripStart { i | interval = interval }

                PretripStart i ->
                    PretripStart { i | interval = interval }
    in
    { req | start = newStart }


setPretripSearchOptions :
    IntermodalRoutingRequest
    -> PretripSearchOptions
    -> IntermodalRoutingRequest
setPretripSearchOptions req options =
    let
        newStart =
            case req.start of
                IntermodalPretripStart i ->
                    IntermodalPretripStart
                        { i
                            | interval = options.interval
                            , minConnectionCount = options.minConnectionCount
                            , extendIntervalEarlier = options.extendIntervalEarlier
                            , extendIntervalLater = options.extendIntervalLater
                        }

                PretripStart i ->
                    PretripStart
                        { i
                            | interval = options.interval
                            , minConnectionCount = options.minConnectionCount
                            , extendIntervalEarlier = options.extendIntervalEarlier
                            , extendIntervalLater = options.extendIntervalLater
                        }
    in
    { req | start = newStart }


swapStartDestination : IntermodalStart -> IntermodalDestination -> ( IntermodalStart, IntermodalDestination )
swapStartDestination origStart origDest =
    let
        start =
            case origDest of
                InputStation s ->
                    case origStart of
                        IntermodalPretripStart ps ->
                            PretripStart
                                { station = s
                                , interval = ps.interval
                                , minConnectionCount = ps.minConnectionCount
                                , extendIntervalEarlier = ps.extendIntervalEarlier
                                , extendIntervalLater = ps.extendIntervalLater
                                }

                        PretripStart ps ->
                            PretripStart { ps | station = s }

                InputPosition p ->
                    case origStart of
                        IntermodalPretripStart ps ->
                            IntermodalPretripStart { ps | position = p }

                        PretripStart ps ->
                            IntermodalPretripStart
                                { position = p
                                , interval = ps.interval
                                , minConnectionCount = ps.minConnectionCount
                                , extendIntervalEarlier = ps.extendIntervalEarlier
                                , extendIntervalLater = ps.extendIntervalLater
                                }

        dest =
            case origStart of
                IntermodalPretripStart ps ->
                    InputPosition ps.position

                PretripStart ps ->
                    InputStation ps.station
    in
    ( start, dest )


fixStartDestination : IntermodalStart -> IntermodalDestination -> SearchDirection -> ( IntermodalStart, IntermodalDestination )
fixStartDestination origStart origDest dir =
    case dir of
        Forward ->
            ( origStart, origDest )

        Backward ->
            swapStartDestination origStart origDest


encodeRequest : IntermodalRoutingRequest -> Encode.Value
encodeRequest request =
    let
        ( fixedStart, fixedDestination ) =
            fixStartDestination request.start request.destination request.searchDir

        ( startType, start ) =
            encodeIntermodalStart fixedStart

        ( destinationType, destination ) =
            encodeIntermodalDestination fixedDestination

        ( startModes, destinationModes ) =
            case request.searchDir of
                Forward ->
                    ( request.startModes, request.destinationModes )

                Backward ->
                    ( request.destinationModes, request.startModes )
    in
    Encode.object
        [ "destination"
            => Encode.object
                [ "type" => Encode.string "Module"
                , "target" => Encode.string "/intermodal"
                ]
        , "content_type" => Encode.string "IntermodalRoutingRequest"
        , "content"
            => Encode.object
                [ "start_type" => startType
                , "start" => start
                , "start_modes"
                    => Encode.list (List.map encodeMode startModes)
                , "destination_type" => destinationType
                , "destination" => destination
                , "destination_modes"
                    => Encode.list (List.map encodeMode destinationModes)
                , "search_type" => encodeSearchType request.searchType
                , "search_dir" => encodeSearchDirection request.searchDir
                , "router" => Encode.string ""
                ]
        ]


encodeIntermodalStart : IntermodalStart -> ( Encode.Value, Encode.Value )
encodeIntermodalStart start =
    case start of
        IntermodalPretripStart info ->
            ( Encode.string "IntermodalPretripStart"
            , Encode.object
                [ "position" => encodePosition info.position
                , "interval" => encodeInterval info.interval
                , "min_connection_count"
                    => Encode.int info.minConnectionCount
                , "extend_interval_earlier"
                    => Encode.bool info.extendIntervalEarlier
                , "extend_interval_later"
                    => Encode.bool info.extendIntervalLater
                ]
            )

        PretripStart info ->
            ( Encode.string "PretripStart"
            , Encode.object
                [ "station" => encodeInputStation info.station
                , "interval" => encodeInterval info.interval
                , "min_connection_count"
                    => Encode.int info.minConnectionCount
                , "extend_interval_earlier"
                    => Encode.bool info.extendIntervalEarlier
                , "extend_interval_later"
                    => Encode.bool info.extendIntervalLater
                ]
            )


encodeIntermodalDestination : IntermodalDestination -> ( Encode.Value, Encode.Value )
encodeIntermodalDestination start =
    case start of
        InputStation station ->
            ( Encode.string "InputStation"
            , encodeInputStation station
            )

        InputPosition pos ->
            ( Encode.string "InputPosition"
            , encodePosition pos
            )


encodeInterval : Interval -> Encode.Value
encodeInterval interval =
    Encode.object
        [ "begin" => Encode.int interval.begin
        , "end" => Encode.int interval.end
        ]


encodeMode : Mode -> Encode.Value
encodeMode mode =
    case mode of
        Foot info ->
            Encode.object
                [ "mode_type" => Encode.string "Foot"
                , "mode"
                    => Encode.object
                        [ "max_duration" => Encode.int info.maxDuration ]
                ]

        Bike info ->
            Encode.object
                [ "mode_type" => Encode.string "Bike"
                , "mode"
                    => Encode.object
                        [ "max_duration" => Encode.int info.maxDuration ]
                ]

        GBFS info ->
            Encode.object
                [ "mode_type" => Encode.string "GBFS"
                , "mode"
                    => Encode.object
                        [ "max_walk_duration" => Encode.int info.maxWalkDuration
                        , "max_vehicle_duration" => Encode.int info.maxVehicleDuration
                        , "provider" => Encode.string info.provider
                        ]
                ]

        Car info ->
            Encode.object
                [ "mode_type" => Encode.string "Car"
                , "mode" => Encode.object [ "max_duration" => Encode.int info.maxDuration ]
                ]

        FootPPR info ->
            Encode.object
                [ "mode_type" => Encode.string "FootPPR"
                , "mode"
                    => Encode.object
                        [ "search_options" => encodeSearchOptions info.searchOptions ]
                ]

        CarParking info ->
            Encode.object
                [ "mode_type" => Encode.string "CarParking"
                , "mode"
                    => Encode.object
                        [ "max_car_duration" => Encode.int info.maxCarDuration
                        , "ppr_search_options" => encodeSearchOptions info.pprSearchOptions
                        ]
                ]
