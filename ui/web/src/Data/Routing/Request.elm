module Data.Routing.Request exposing
    ( encodeInputStation
    , encodeRequest
    , encodeSearchDirection
    , encodeSearchType
    , initialRequest
    )

import Data.Connection.Types exposing (Station)
import Data.Routing.Types exposing (RoutingRequest, SearchDirection(..), SearchType(..))
import Date exposing (Date)
import Json.Encode as Encode
import Util.DateUtil exposing (unixTime)


initialRequest :
    Int
    -> Station
    -> Station
    -> Date
    -> SearchDirection
    -> RoutingRequest
initialRequest minConnectionCount from to date searchDirection =
    let
        selectedTime =
            unixTime date

        startTime =
            selectedTime - 3600

        endTime =
            selectedTime + 3600
    in
    { from = from
    , to = to
    , intervalStart = startTime
    , intervalEnd = endTime
    , minConnectionCount = minConnectionCount
    , searchDirection = searchDirection
    , extendIntervalEarlier = True
    , extendIntervalLater = True
    }


encodeRequest : RoutingRequest -> Encode.Value
encodeRequest request =
    Encode.object
        [ "destination"
            , Encode.object
                [ "type" , Encode.string "Module"
                , "target" , Encode.string "/routing"
                ]
        , "content_type" , Encode.string "RoutingRequest"
        , "content"
            , Encode.object
                [ "start_type" , Encode.string "PretripStart"
                , "start"
                    , Encode.object
                        [ "station" , encodeInputStation request.from
                        , "interval"
                            , Encode.object
                                [ "begin" , Encode.int request.intervalStart
                                , "end" , Encode.int request.intervalEnd
                                ]
                        , "min_connection_count"
                            , Encode.int request.minConnectionCount
                        , "extend_interval_earlier"
                            , Encode.bool request.extendIntervalEarlier
                        , "extend_interval_later"
                            , Encode.bool request.extendIntervalLater
                        ]
                , "destination" , encodeInputStation request.to
                , "search_type" , encodeSearchType DefaultSearchType
                , "search_dir" , encodeSearchDirection request.searchDirection
                , "via" , Encode.list []
                , "additional_edges" , Encode.list []
                ]
        ]


encodeInputStation : Station -> Encode.Value
encodeInputStation station =
    Encode.object
        [ "name" , Encode.string station.name
        , "id" , Encode.string station.id
        ]


encodeSearchDirection : SearchDirection -> Encode.Value
encodeSearchDirection direction =
    case direction of
        Forward ->
            Encode.string "Forward"

        Backward ->
            Encode.string "Backward"


encodeSearchType : SearchType -> Encode.Value
encodeSearchType t =
    case t of
        DefaultSearchType ->
            Encode.string "Default"

        SingleCriterion ->
            Encode.string "SingleCriterion"

        SingleCriterionNoIntercity ->
            Encode.string "SingleCriterionNoIntercity"

        LateConnections ->
            Encode.string "LateConnections"

        LateConnectionsTest ->
            Encode.string "LateConnectionsTest"

        AccessibilitySearchType ->
            Encode.string "Accessibility"
