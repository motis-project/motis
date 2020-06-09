module Routes exposing
    ( Route(..)
    , routeToTripId
    , toUrl
    , tripDetailsRoute
    , urlParser
    )

import Data.Connection.Types exposing (TripId)
import Date exposing (Date)
import Http
import UrlParser exposing ((</>), Parser, custom, int, map, oneOf, parseHash, s, string, top)
import Util.Date exposing (unixTime)


type Route
    = Connections
    | ConnectionDetails Int
    | TripDetails String Int Int String Int String
    | StationEvents String
    | StationEventsAt String Date
    | SimulationTime Date
    | TripSearchRoute
    | RailVizPermalink Float Float Float Float Float Date


urlParser : Parser (Route -> a) a
urlParser =
    oneOf
        [ map Connections top
        , map ConnectionDetails (s "connection" </> int)
        , map TripDetails (s "trip" </> encodedString </> int </> int </> encodedString </> int </> encodedString)
        , map StationEvents (s "station" </> encodedString)
        , map StationEventsAt (s "station" </> encodedString </> date)
        , map SimulationTime (s "time" </> date)
        , map TripSearchRoute (s "trips")
        , map RailVizPermalink (s "railviz" </> float </> float </> float </> float </> float </> date)
        ]


date : Parser (Date -> a) a
date =
    oneOf
        [ unixTimestamp
        , nativeDate
        ]


float : Parser (Float -> a) a
float =
    custom "FLOAT" String.toFloat


unixTimestamp : Parser (Date -> a) a
unixTimestamp =
    custom "UNIX_TIMESTAMP" (String.toFloat >> Result.map (\u -> Date.fromTime (u * 1000.0)))


nativeDate : Parser (Date -> a) a
nativeDate =
    custom "DATE_STR" Date.fromString


encodedString : Parser (String -> a) a
encodedString =
    custom "ENCODED_STRING" (Http.decodeUri >> Result.fromMaybe "decodeUri error")


dateToUrl : Date -> String
dateToUrl d =
    toString (round (Date.toTime d / 1000.0))


toUrl : Route -> String
toUrl route =
    case route of
        Connections ->
            "#/"

        ConnectionDetails idx ->
            "#/connection/" ++ toString idx

        TripDetails station trainNr time targetStation targetTime lineId ->
            "#/trip/"
                ++ Http.encodeUri station
                ++ "/"
                ++ toString trainNr
                ++ "/"
                ++ toString time
                ++ "/"
                ++ Http.encodeUri targetStation
                ++ "/"
                ++ toString targetTime
                ++ "/"
                ++ Http.encodeUri lineId

        StationEvents stationId ->
            "#/station/" ++ Http.encodeUri stationId

        StationEventsAt stationId date ->
            "#/station/" ++ Http.encodeUri stationId ++ "/" ++ toString (unixTime date)

        SimulationTime time ->
            "#/time/" ++ dateToUrl time

        TripSearchRoute ->
            "#/trips/"

        RailVizPermalink lat lng zoom bearing pitch date ->
            "#/railviz/"
                ++ toString lat ++ "/"
                ++ toString lng ++ "/"
                ++ toString zoom ++ "/"
                ++ toString bearing ++ "/"
                ++ toString pitch ++ "/"
                ++ toString (unixTime date)


tripDetailsRoute : TripId -> Route
tripDetailsRoute trip =
    TripDetails
        trip.station_id
        trip.train_nr
        trip.time
        trip.target_station_id
        trip.target_time
        trip.line_id


routeToTripId : Route -> Maybe TripId
routeToTripId route =
    case route of
        TripDetails station trainNr time targetStation targetTime lineId ->
            Just
                { station_id = station
                , train_nr = trainNr
                , time = time
                , target_station_id = targetStation
                , target_time = targetTime
                , line_id = lineId
                }

        _ ->
            Nothing
