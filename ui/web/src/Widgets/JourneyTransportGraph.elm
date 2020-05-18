module Widgets.JourneyTransportGraph exposing
    ( Model
    , Msg
    , hideTooltips
    , init
    , update
    , view
    )

import Data.Connection.Types exposing (Station, Stop, TripId)
import Data.Journey.Types as Journey exposing (Journey, JourneyMove(..), JourneyWalk, Train, isMumoWalk)
import Date exposing (Date)
import Date.Extra.Duration as Duration exposing (DeltaRecord)
import Html exposing (Html, div)
import Html.Attributes
import Html.Lazy
import List.Extra
import Localization.Base exposing (..)
import String
import Svg exposing (..)
import Svg.Attributes exposing (..)
import Svg.Events exposing (..)
import Util.Core exposing ((=>))
import Util.DateFormat exposing (formatTime)
import Util.List exposing (last)
import Widgets.Helpers.ConnectionUtil exposing (..)
import Widgets.Map.Port exposing (RVConnectionSegmentTrip, RVConnectionSegmentWalk, RVConnectionWalk)



-- MODEL


type alias Model =
    { displayParts : List DisplayPart
    , totalWidth : Int
    , hover : Maybe DisplayPart
    }


type alias Part =
    { icon : String
    , colorClass : String
    , duration : Int
    , longName : String
    , shortName : String
    , departureStation : Station
    , departureTime : Date
    , arrivalStation : Station
    , arrivalTime : Date
    , accessibility : Int
    , train : Maybe Train
    , walk : Maybe JourneyWalk
    }


type alias DisplayPart =
    { part : Part
    , position : Float
    , barLength : Float
    , nameDisplayType : NameDisplayType
    }


type NameDisplayType
    = LongName
    | NoName


init : Int -> Journey -> Model
init totalWidth journey =
    let
        parts =
            journeyParts journey |> layoutParts totalWidth LongName
    in
    { displayParts = parts
    , totalWidth = totalWidth
    , hover = Nothing
    }



-- UPDATE


type Msg
    = MouseOver DisplayPart
    | MouseOut DisplayPart


update : Msg -> Model -> Model
update msg model =
    case msg of
        MouseOver part ->
            { model | hover = Just part }

        MouseOut part ->
            case model.hover of
                Just hoveredPart ->
                    if hoveredPart == part then
                        { model | hover = Nothing }

                    else
                        model

                _ ->
                    model


hideTooltips : Model -> Model
hideTooltips model =
    { model | hover = Nothing }



-- VIEW


view : Localization -> ( Bool, Maybe (List RVConnectionSegmentTrip), Maybe RVConnectionSegmentWalk ) -> Model -> Html Msg
view locale highlighting model =
    Html.Lazy.lazy3 graphView locale highlighting model


graphView : Localization -> ( Bool, Maybe ( List RVConnectionSegmentTrip), Maybe RVConnectionSegmentWalk ) -> Model -> Html Msg
graphView locale ( useHighlighting, hoveredTrips, hoveredWalk ) model =
    div
        [ Html.Attributes.classList
            [ "transport-graph" => True
            , "highlighting" => useHighlighting
            ]
        ]
        (transportsView locale ( useHighlighting, hoveredTrips, hoveredWalk ) model)


transportsView :
    Localization
    -> ( Bool, Maybe ( List RVConnectionSegmentTrip ), Maybe RVConnectionSegmentWalk )
    -> Model
    -> List (Html Msg)
transportsView locale ( useHighlighting, hoveredTrips, hoveredWalk ) model =
    let
        isHovered displayPart =
            case model.hover of
                Just hoveredPart ->
                    hoveredPart == displayPart

                Nothing ->
                    isPartHighlighted hoveredTrips hoveredWalk displayPart.part

        renderedParts =
            List.map
                (\p -> partView locale model.totalWidth (isHovered p) useHighlighting p)
                model.displayParts
    in
    [ svg
        [ width (toString model.totalWidth)
        , height (toString totalHeight)
        , viewBox <| "0 0 " ++ toString model.totalWidth ++ " " ++ toString totalHeight
        ]
        [ g [] (List.map Tuple.first renderedParts)
        , destinationView model.totalWidth
        ]
    ]
        ++ List.map Tuple.second renderedParts


isPartHighlighted : Maybe (List RVConnectionSegmentTrip ) -> Maybe RVConnectionSegmentWalk -> Part -> Bool
isPartHighlighted hoveredTrips hoveredWalk part =
    case ( hoveredTrips, part.train ) of
        ( Just segments, Just train ) ->
            segments
                |> List.any (\seg -> isMatchingTrain seg train)

        _ ->
            case ( hoveredWalk, part.walk ) of
                ( Just seg, Just _ ) ->
                    isMatchingWalk seg.walk part

                _ ->
                    False


isMatchingTrain : RVConnectionSegmentTrip -> Train -> Bool
isMatchingTrain seg train =
    let
        checkTrip trip =
            seg.trip
                |> List.any (\t -> t == trip)

        containsStop stationId =
            train.stops
                |> List.any (\stop -> stop.station.id == stationId)

        checkStations r =
            r
                && containsStop seg.d_station_id
                && containsStop seg.a_station_id
    in
    train.trip
        |> Maybe.map checkTrip
        |> Maybe.map checkStations
        |> Maybe.withDefault False


isMatchingWalk : RVConnectionWalk -> Part -> Bool
isMatchingWalk walk part =
    (walk.departureStation == part.departureStation)
        && (walk.arrivalStation == part.arrivalStation)
        && (walk.duration == part.duration)
        && (walk.accessibility == part.accessibility)


destinationView : Int -> Svg msg
destinationView totalWidth =
    g [ class "destination" ]
        [ circle
            [ cx (toString <| totalWidth - destinationRadius)
            , cy (toString circleRadius)
            , r (toString destinationRadius)
            ]
            []
        ]


partView : Localization -> Int -> Bool -> Bool -> DisplayPart -> ( Svg Msg, Html Msg )
partView locale totalWidth tooltipVisible useHighlighting displayPart =
    let
        { part, position, barLength, nameDisplayType } =
            displayPart

        radius =
            toString circleRadius

        partWidth =
            (circleRadius * 2) + barLength

        lineEnd =
            position + partWidth + (destinationRadius / 2)

        trainName =
            case nameDisplayType of
                LongName ->
                    text_
                        [ x (toString <| position)
                        , y (toString <| textOffset + textHeight)
                        , textAnchor "start"
                        , class "train-name"
                        ]
                        [ text part.longName ]

                NoName ->
                    text ""

        accessibilityClass =
            if part.accessibility == 0 then
                "acc-0"

            else if part.accessibility < 30 then
                "acc-1"

            else
                "acc-2"

        highlightClass =
            if useHighlighting then
                if tooltipVisible then
                    " highlighted"

                else
                    " faded"

            else
                ""

        graphPart =
            g
                [ class <| "part train-class-" ++ part.colorClass ++ " " ++ accessibilityClass ++ highlightClass ]
            <|
                [ line
                    [ x1 (toString <| position)
                    , y1 radius
                    , x2 (toString <| lineEnd)
                    , y2 radius
                    , class "train-line"
                    ]
                    []
                , circle
                    [ cx (toString <| position + circleRadius)
                    , cy radius
                    , r radius
                    , class "train-circle"
                    ]
                    []
                , use
                    [ xlinkHref <| "#" ++ part.icon
                    , class "train-icon"
                    , x (toString <| position + iconOffset)
                    , y (toString <| iconOffset)
                    , width (toString <| iconSize)
                    , height (toString <| iconSize)
                    ]
                    []
                , trainName
                , rect
                    [ x (position |> toString)
                    , y "0"
                    , width (position + partWidth |> toString)
                    , height (circleRadius * 2 |> toString)
                    , class "tooltipTrigger"
                    , onMouseOver (MouseOver displayPart)
                    , onMouseOut (MouseOut displayPart)
                    ]
                    []
                ]

        tooltipX =
            Basics.min position (toFloat totalWidth - tooltipWidth)

        tooltipTransportName =
            case part.icon of
                "walk" ->
                    locale.t.connections.walk

                "bike" ->
                    locale.t.connections.bike

                "car" ->
                    locale.t.connections.car

                _ ->
                    part.longName

        getStationName station =
            if String.startsWith "VIA" station.id then
                locale.t.connections.parking

            else
                station.name

        departureStationName =
            getStationName part.departureStation

        arrivalStationName =
            getStationName part.arrivalStation

        tooltip =
            Html.div
                [ Html.Attributes.classList
                    [ "tooltip" => True
                    , "visible" => tooltipVisible
                    ]
                , Html.Attributes.style
                    [ "position" => "absolute"
                    , "left" => (toString tooltipX ++ "px")
                    , "top" => (toString (textOffset - 5) ++ "px")
                    , "width" => toString tooltipWidth
                    , "height" => "50"
                    ]
                ]
                [ Html.div [ Html.Attributes.class "stations" ]
                    [ Html.div [ Html.Attributes.class "departure" ]
                        [ Html.span [ Html.Attributes.class "station" ]
                            [ text departureStationName ]
                        , Html.span [ Html.Attributes.class "time" ]
                            [ text (formatTime part.departureTime) ]
                        ]
                    , Html.div [ Html.Attributes.class "arrival" ]
                        [ Html.span [ Html.Attributes.class "station" ]
                            [ text arrivalStationName ]
                        , Html.span [ Html.Attributes.class "time" ]
                            [ text (formatTime part.arrivalTime) ]
                        ]
                    ]
                , Html.div [ Html.Attributes.class "transport-name" ]
                    [ Html.span [] [ Html.text tooltipTransportName ] ]
                ]
    in
    ( graphPart, tooltip )


getTotalDuration : List Part -> Int
getTotalDuration parts =
    List.map .duration parts |> List.sum


journeyParts : Journey -> List Part
journeyParts { moves } =
    let
        toPart move =
            case move of
                TrainMove t ->
                    trainPart t

                WalkMove w ->
                    walkPart w

        showMove move =
            case move of
                TrainMove t ->
                    True

                WalkMove w ->
                    isMumoWalk w
    in
    moves
        |> List.filter showMove
        |> List.map toPart


missingStation : Station
missingStation =
    { id = ""
    , name = ""
    , pos = { lat = 0, lng = 0 }
    }


trainPart : Train -> Part
trainPart train =
    let
        transport =
            List.head train.transports

        departure =
            List.head train.stops

        arrival =
            last train.stops

        ( departureStation, departureTime ) =
            Maybe.map departureInfo departure
                |> Maybe.withDefault ( missingStation, Date.fromTime 0 )

        ( arrivalStation, arrivalTime ) =
            Maybe.map arrivalInfo arrival
                |> Maybe.withDefault ( missingStation, Date.fromTime 0 )

        base =
            { icon = "train"
            , colorClass = "0"
            , duration = trainDuration train
            , longName = ""
            , shortName = ""
            , departureStation = departureStation
            , departureTime = departureTime
            , arrivalStation = arrivalStation
            , arrivalTime = arrivalTime
            , accessibility = 0
            , train = Just train
            , walk = Nothing
            }
    in
    case transport of
        Just t ->
            { base
                | icon = trainIcon t.class
                , colorClass = toString t.class
                , longName = longTransportNameWithoutIcon t
                , shortName = shortTransportName t
            }

        Nothing ->
            base


walkPart : JourneyWalk -> Part
walkPart walk =
    let
        ( departureStation, departureTime ) =
            departureInfo walk.from

        ( arrivalStation, arrivalTime ) =
            arrivalInfo walk.to

        icon =
            case walk.mumoType of
                "bike" ->
                    "bike"

                "car" ->
                    "car"

                _ ->
                    "walk"
    in
    { icon = icon
    , colorClass = icon
    , duration = deltaRecordToMinutes walk.duration
    , longName = ""
    , shortName = ""
    , departureStation = departureStation
    , departureTime = departureTime
    , arrivalStation = arrivalStation
    , arrivalTime = arrivalTime
    , accessibility = walk.accessibility
    , train = Nothing
    , walk = Just walk
    }


departureInfo : Stop -> ( Station, Date )
departureInfo stop =
    let
        station =
            stop.station

        time =
            stop.departure.schedule_time
                |> Maybe.withDefault (Date.fromTime 0)
    in
    ( station, time )


arrivalInfo : Stop -> ( Station, Date )
arrivalInfo stop =
    let
        station =
            stop.station

        time =
            stop.arrival.schedule_time
                |> Maybe.withDefault (Date.fromTime 0)
    in
    ( station, time )


deltaRecordToMinutes : DeltaRecord -> Int
deltaRecordToMinutes dr =
    dr.minute + 60 * dr.hour + 1440 * dr.day


walkDuration : JourneyWalk -> Int
walkDuration walk =
    deltaRecordToMinutes walk.duration


trainDuration : Train -> Int
trainDuration { stops } =
    let
        departure : Maybe Date
        departure =
            List.head stops |> Maybe.andThen (.departure >> .schedule_time)

        arrival : Maybe Date
        arrival =
            last stops |> Maybe.andThen (.arrival >> .schedule_time)

        minutesBetween : Date -> Date -> Int
        minutesBetween from to =
            deltaRecordToMinutes <| Duration.diff to from
    in
    Maybe.map2 minutesBetween departure arrival |> Maybe.withDefault 0


type alias LayoutPart =
    { part : Part
    , minWidth : Float
    , idealWidth : Float
    , finalWidth : Float
    }


isFinal : LayoutPart -> Bool
isFinal part =
    part.finalWidth > 0


isNotFinal : LayoutPart -> Bool
isNotFinal part =
    part.finalWidth == 0


layoutParts : Int -> NameDisplayType -> List Part -> List DisplayPart
layoutParts totalWidth desiredNameDisplayType parts =
    let
        destinationWidth =
            destinationRadius * 2

        ( parts_, nameDisplayType ) =
            calcPartWidths
                (totalWidth - destinationWidth)
                desiredNameDisplayType
                parts

        setPosition lp ( pos, results ) =
            let
                displayPart =
                    { part = lp.part
                    , position = pos
                    , barLength = lp.finalWidth - basePartSize
                    , nameDisplayType = nameDisplayType
                    }

                nextPos =
                    pos + basePartSize + displayPart.barLength
            in
            ( nextPos, results ++ [ displayPart ] )

        ( _, displayParts ) =
            List.foldl setPosition ( 0, [] ) parts_
    in
    displayParts


calcPartWidths : Int -> NameDisplayType -> List Part -> ( List LayoutPart, NameDisplayType )
calcPartWidths availableWidth nameDisplayType parts =
    let
        baseBarLength =
            2

        avgCharWidth =
            7

        getMinLength part =
            case nameDisplayType of
                LongName ->
                    Basics.max
                        (basePartSize + baseBarLength)
                        (avgCharWidth * String.length part.longName)
                        |> toFloat

                NoName ->
                    basePartSize + baseBarLength |> toFloat

        initialLayoutPart part =
            { part = part
            , minWidth = getMinLength part
            , idealWidth = 0
            , finalWidth = 0
            }

        initialLayoutParts =
            List.map initialLayoutPart parts

        minRequiredSpace =
            initialLayoutParts
                |> List.map .minWidth
                |> List.sum
    in
    if minRequiredSpace <= toFloat availableWidth then
        ( calcFinalPartWidths
            (toFloat availableWidth)
            nameDisplayType
            initialLayoutParts
        , nameDisplayType
        )

    else
        calcPartWidths availableWidth NoName parts


calcFinalPartWidths : Float -> NameDisplayType -> List LayoutPart -> List LayoutPart
calcFinalPartWidths totalAvailableWidth nameDisplayType parts =
    let
        remainingDuration =
            parts
                |> List.filter isNotFinal
                |> List.map (.part >> .duration)
                |> List.sum
                |> toFloat

        finalPartsWidth p =
            p
                |> List.map .finalWidth
                |> List.sum

        availableWidth =
            totalAvailableWidth - finalPartsWidth parts

        getIdealWidth part =
            toFloat part.part.duration / remainingDuration * availableWidth

        idealParts =
            List.Extra.updateIf
                isNotFinal
                (\p -> { p | idealWidth = getIdealWidth p })
                parts

        finalizedParts =
            List.Extra.updateIf
                (\p -> isNotFinal p && p.minWidth >= p.idealWidth)
                (\p -> { p | finalWidth = p.minWidth })
                idealParts

        done =
            idealParts == finalizedParts
    in
    if done then
        finalizedParts
            |> List.Extra.updateIf
                isNotFinal
                (\p -> { p | finalWidth = p.idealWidth })

    else
        calcFinalPartWidths totalAvailableWidth nameDisplayType finalizedParts


iconSize : number
iconSize =
    16


circleRadius : number
circleRadius =
    12


basePartSize : number
basePartSize =
    circleRadius * 2


iconOffset : Float
iconOffset =
    ((circleRadius * 2) - iconSize) / 2


destinationRadius : number
destinationRadius =
    6


textOffset : number
textOffset =
    circleRadius * 2 + 4


textHeight : number
textHeight =
    12


totalHeight : number
totalHeight =
    textOffset + textHeight


tooltipWidth : number
tooltipWidth =
    240
