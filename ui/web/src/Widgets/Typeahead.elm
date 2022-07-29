module Widgets.Typeahead exposing
    ( Model
    , Msg(..)
    , Suggestion(..)
    , getSelectedAddress
    , getSelectedStation
    , getSelectedSuggestion
    , getShortSuggestionName
    , getSuggestionName
    , getSuggestionPosition
    , init
    , saveSelection
    , setToPosition
    , update
    , view
    )

import Data.Address.Decode exposing (decodeAddress, decodeAddressResponse)
import Data.Address.Request exposing (encodeAddress, encodeAddressRequest)
import Data.Address.Types exposing (..)
import Data.Connection.Decode exposing (decodePosition, decodeStation)
import Data.Connection.Request exposing (encodePosition, encodeStation)
import Data.Connection.Types exposing (Position, Station)
import Data.StationGuesser.Decode exposing (decodeStationGuesserResponse)
import Data.StationGuesser.Request as StationGuesser
import Debounce
import Dict exposing (..)
import Html exposing (Html, div, i, li, span, text, ul)
import Html.Attributes exposing (..)
import Html.Events exposing (keyCode, on, onClick, onFocus, onInput, onMouseOver)
import Html.Lazy exposing (..)
import Json.Decode as Decode
import Json.Decode.Pipeline as JDP exposing (decode, hardcoded, optional, required, requiredAt)
import Json.Encode as Encode
import List.Extra
import Maybe.Extra
import String
import Task
import Util.Api as Api exposing (ApiError(..))
import Util.Core exposing ((=>))
import Util.List exposing ((!!), last)
import Util.View exposing (onStopAll)
import Widgets.Input as Input



-- MODEL


type alias Model =
    { stationSuggestions : List Station
    , addressSuggestions : List Address
    , positionSuggestions : List Position
    , suggestions : List Suggestion
    , input : String
    , hoverIndex : Int
    , selectedSuggestion : Maybe Suggestion
    , visible : Bool
    , inputWidget : Input.Model
    , remoteAddress : String
    , debounce : Debounce.State
    }


type Suggestion
    = StationSuggestion Station
    | AddressSuggestion Address
    | PositionSuggestion Position


init : String -> String -> ( Model, Cmd Msg )
init remoteAddress initialValue =
    let
        suggestion =
            restoreSelection initialValue

        inputText =
            suggestion
                |> Maybe.map getSuggestionName
                |> Maybe.withDefault initialValue
    in
    { suggestions = []
    , stationSuggestions = []
    , addressSuggestions = []
    , positionSuggestions = []
    , input = inputText
    , hoverIndex = 0
    , selectedSuggestion = suggestion
    , visible = False
    , inputWidget = Input.init
    , remoteAddress = remoteAddress
    , debounce = Debounce.init
    }
        ! (if String.isEmpty inputText then
            []

           else
            [ requestSuggestions remoteAddress inputText ]
          )



-- UPDATE


type Msg
    = NoOp
    | StationSuggestionsResponse (List Station)
    | StationSuggestionsError ApiError
    | AddressSuggestionsResponse AddressResponse
    | AddressSuggestionsError ApiError
    | PositionSuggestionsResponse (Maybe Position)
    | InputChange String
    | SetText String
    | EnterSelection
    | ClickElement Int
    | SelectionUp
    | SelectionDown
    | Select Int
    | Hide
    | InputUpdate Input.Msg
    | Deb (Debounce.Msg Msg)
    | RequestSuggestions
    | ItemSelected
    | Empty


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            model ! []

        Empty ->
            model ! []

        StationSuggestionsResponse suggestions ->
            updateSuggestions { model | stationSuggestions = suggestions } ! []

        StationSuggestionsError _ ->
            updateSuggestions { model | stationSuggestions = [] } ! []

        AddressSuggestionsResponse response ->
            updateSuggestions
                { model
                    | addressSuggestions = filterAddressSuggestions response.guesses
                }
                ! []

        AddressSuggestionsError err ->
            let
                _ =
                    Debug.log "AddressSuggestionsError" err
            in
            updateSuggestions { model | addressSuggestions = [] } ! []

        PositionSuggestionsResponse maybePos ->
            updateSuggestions { model | positionSuggestions = Maybe.Extra.maybeToList maybePos } ! []

        InputChange str ->
            { model
                | input = str
                , visible = True
                , selectedSuggestion = getSuggestionByName model str
            }
                ! [ Debounce.debounceCmd debounceCfg RequestSuggestions ]

        SetText str ->
            { model
                | input = str
                , visible = False
                , selectedSuggestion = getSuggestionByName model str
            }
                ! [ Debounce.debounceCmd debounceCfg RequestSuggestions ]

        EnterSelection ->
            { model
                | visible = False
                , input = getEntryName model model.hoverIndex
                , selectedSuggestion = model.suggestions !! model.hoverIndex
            }
                ! [ Debounce.debounceCmd debounceCfg RequestSuggestions
                  , Task.perform identity (Task.succeed ItemSelected)
                  ]

        ClickElement i ->
            { model
                | visible = False
                , hoverIndex = 0
                , input = getEntryName model i
                , selectedSuggestion = model.suggestions !! i
            }
                ! [ Debounce.debounceCmd debounceCfg RequestSuggestions
                  , Task.perform identity (Task.succeed ItemSelected)
                  ]

        SelectionUp ->
            { model | hoverIndex = (model.hoverIndex - 1) % List.length model.suggestions } ! []

        SelectionDown ->
            { model | hoverIndex = (model.hoverIndex + 1) % List.length model.suggestions } ! []

        Select index ->
            { model | hoverIndex = index } ! []

        Hide ->
            { model | visible = False, hoverIndex = 0 }
                ! [ if String.isEmpty model.input then
                        Task.perform identity (Task.succeed Empty)

                    else
                        Cmd.none
                  ]

        InputUpdate msg_ ->
            let
                updated =
                    case msg_ of
                        Input.Focus ->
                            { model | visible = True }

                        Input.Click ->
                            { model | visible = True }

                        Input.Blur ->
                            { model | visible = False, hoverIndex = 0 }
            in
            { updated | inputWidget = Input.update msg_ model.inputWidget }
                ! [ if not updated.visible && String.isEmpty updated.input then
                        Task.perform identity (Task.succeed Empty)

                    else
                        Cmd.none
                  ]

        Deb a ->
            Debounce.update debounceCfg a model

        RequestSuggestions ->
            model
                ! [ if String.length model.input > 2 then
                        requestSuggestions model.remoteAddress model.input

                    else
                        Cmd.none
                  ]

        ItemSelected ->
            model ! []


updateSuggestions : Model -> Model
updateSuggestions model =
    let
        stations =
            List.map StationSuggestion model.stationSuggestions

        addresses =
            List.map AddressSuggestion model.addressSuggestions

        positions =
            List.map PositionSuggestion model.positionSuggestions

        model1 =
            { model | suggestions = positions ++ stations ++ addresses }

        model2 =
            case model1.selectedSuggestion of
                Nothing ->
                    { model1
                        | selectedSuggestion = getSuggestionByName model1 model1.input
                    }

                Just _ ->
                    model1
    in
    model2


getEntryName : Model -> Int -> String
getEntryName { suggestions } idx =
    suggestions
        !! idx
        |> Maybe.map getSuggestionName
        |> Maybe.withDefault ""


getSuggestionName : Suggestion -> String
getSuggestionName suggestion =
    case suggestion of
        StationSuggestion station ->
            station.name

        AddressSuggestion address ->
            getAddressStr address

        PositionSuggestion pos ->
            getPositionStr pos


getShortSuggestionName : Suggestion -> String
getShortSuggestionName suggestion =
    case suggestion of
        StationSuggestion station ->
            station.name

        AddressSuggestion address ->
            getShortAddressStr address

        PositionSuggestion pos ->
            getPositionStr pos


getSuggestionPosition : Suggestion -> Position
getSuggestionPosition suggestion =
    case suggestion of
        StationSuggestion station ->
            station.pos

        AddressSuggestion address ->
            address.pos

        PositionSuggestion pos ->
            pos


getAddressStr : Address -> String
getAddressStr =
    getShortAddressStr


getShortAddressStr : Address -> String
getShortAddressStr address =
    case getCity address of
        Just city ->
            address.name ++ ", " ++ city

        Nothing ->
            address.name


getRegionStr : Address -> String
getRegionStr address =
    let
        city =
            getCity address

        country =
            getCountry address
    in
    [ city, country ]
        |> Maybe.Extra.values
        |> String.join ", "


getCity : Address -> Maybe String
getCity address =
    address.regions
        |> List.filter (\a -> a.adminLevel <= 8)
        |> List.head
        |> Maybe.map .name


getCountry : Address -> Maybe String
getCountry address =
    address.regions
        |> List.filter (\a -> a.adminLevel == 2)
        |> List.head
        |> Maybe.map .name


getPositionStr : Position -> String
getPositionStr pos =
    toString pos.lat ++ ";" ++ toString pos.lng


getSelectedSuggestion : Model -> Maybe Suggestion
getSelectedSuggestion model =
    model.selectedSuggestion


getSelectedStation : Model -> Maybe Station
getSelectedStation model =
    case getSelectedSuggestion model of
        Just (StationSuggestion station) ->
            Just station

        _ ->
            Nothing


getSelectedAddress : Model -> Maybe Address
getSelectedAddress model =
    case getSelectedSuggestion model of
        Just (AddressSuggestion address) ->
            Just address

        _ ->
            Nothing


filterAddressSuggestions : List Address -> List Address
filterAddressSuggestions suggestions =
    suggestions
        |> List.Extra.uniqueBy getAddressStr


getSuggestionByName : Model -> String -> Maybe Suggestion
getSuggestionByName model rawInput =
    let
        input =
            rawInput |> String.trim |> String.toLower

        checkEntry entry =
            String.toLower (getSuggestionName entry) == input
    in
    model.suggestions
        |> List.filter checkEntry
        |> List.head


debounceCfg : Debounce.Config Model Msg
debounceCfg =
    Debounce.config
        .debounce
        (\model s -> { model | debounce = s })
        Deb
        100


setToPosition : Position -> Msg
setToPosition pos =
    (toString pos.lat ++ ";" ++ toString pos.lng)
        |> SetText



-- VIEW


up : Int
up =
    38


down : Int
down =
    40


enter : Int
enter =
    13


escape : Int
escape =
    27


onKey : Msg -> Dict Int Msg -> Html.Attribute Msg
onKey fail msgs =
    let
        tagger code =
            Dict.get code msgs |> Maybe.withDefault fail
    in
    on "keydown" (Decode.map tagger keyCode)


proposalView : Int -> Int -> Suggestion -> Html Msg
proposalView hoverIndex index suggestion =
    let
        fullName =
            getSuggestionName suggestion

        content =
            case suggestion of
                StationSuggestion station ->
                    stationView station

                AddressSuggestion address ->
                    addressView address

                PositionSuggestion pos ->
                    positionView pos
    in
    li
        [ classList [ ( "selected", hoverIndex == index ) ]
        , onClick (ClickElement index)
        , onMouseOver (Select index)
        , title fullName
        ]
        content


stationView : Station -> List (Html Msg)
stationView station =
    let
        name =
            station.name
    in
    [ i [ class "icon" ] [ text "train" ]
    , span [ class "station" ] [ text name ]
    ]


addressView : Address -> List (Html Msg)
addressView address =
    let
        name =
            address.name

        region =
            getRegionStr address
    in
    [ i [ class "icon" ] [ text "place" ]
    , span [ class "address-name" ] [ text name ]
    , span [ class "address-region" ] [ text region ]
    ]


positionView : Position -> List (Html Msg)
positionView pos =
    let
        str =
            getPositionStr pos
    in
    [ i [ class "icon" ] [ text "my_location" ]
    , span [ class "address-name" ] [ text str ]
    ]


typeaheadView : ( Int, String, Maybe String ) -> Model -> Html Msg
typeaheadView ( tabIndex, label, icon ) model =
    div []
        [ Input.view InputUpdate
            [ value model.input
            , onInput InputChange
            , onKey NoOp
                (Dict.fromList
                    [ ( down, SelectionDown )
                    , ( up, SelectionUp )
                    , ( enter, EnterSelection )
                    , ( escape, Hide )
                    ]
                )
            , tabindex tabIndex
            ]
            label
            Nothing
            icon
            model.inputWidget
        , div
            [ classList
                [ ( "paper", True )
                , ( "hide", not model.visible || List.length model.suggestions == 0 )
                ]
            , onStopAll "mousedown" NoOp
            ]
            [ ul [ class "proposals" ]
                (List.indexedMap (proposalView model.hoverIndex) model.suggestions)
            ]
        ]


view : Int -> String -> Maybe String -> Model -> Html Msg
view tabIndex label icon model =
    lazy2 typeaheadView ( tabIndex, label, icon ) model



-- SUBSCRIPTIONS
{- no subs atm -}
-- REMOTE SUGGESTIONS


requestSuggestions : String -> String -> Cmd Msg
requestSuggestions remoteAddress input =
    Cmd.batch
        [ requestStationSuggestions remoteAddress input
        , requestAddressSuggestions remoteAddress input
        , requestPositionSuggestions input
        ]


requestStationSuggestions : String -> String -> Cmd Msg
requestStationSuggestions remoteAddress input =
    Api.sendRequest
        (remoteAddress ++ "?elm=StationSuggestions")
        decodeStationGuesserResponse
        StationSuggestionsError
        StationSuggestionsResponse
        (StationGuesser.encodeRequest 6 input)


requestAddressSuggestions : String -> String -> Cmd Msg
requestAddressSuggestions remoteAddress input =
    Api.sendRequest
        (remoteAddress ++ "?elm=AddressSuggestions")
        decodeAddressResponse
        AddressSuggestionsError
        AddressSuggestionsResponse
        (encodeAddressRequest input)


requestPositionSuggestions : String -> Cmd Msg
requestPositionSuggestions input =
    Task.perform PositionSuggestionsResponse (Task.succeed (parsePosition input))



-- LOCAL STORAGE


encodeSuggestion : Suggestion -> Encode.Value
encodeSuggestion suggestion =
    case suggestion of
        StationSuggestion station ->
            Encode.object
                [ "type" => Encode.string "Station"
                , "station" => encodeStation station
                ]

        AddressSuggestion address ->
            Encode.object
                [ "type" => Encode.string "Address"
                , "address" => encodeAddress address
                ]

        PositionSuggestion pos ->
            Encode.object
                [ "type" => Encode.string "Position"
                , "position" => encodePosition pos
                ]


decodeSuggestion : Decode.Decoder Suggestion
decodeSuggestion =
    let
        suggestion : String -> Decode.Decoder Suggestion
        suggestion type_ =
            case type_ of
                "Station" ->
                    decode StationSuggestion
                        |> JDP.required "station" decodeStation

                "Address" ->
                    decode AddressSuggestion
                        |> JDP.required "address" decodeAddress

                "Position" ->
                    decode PositionSuggestion
                        |> JDP.required "position" decodePosition

                _ ->
                    Decode.fail "unknown suggestion type"
    in
    Decode.field "type" Decode.string |> Decode.andThen suggestion


saveSelection : Model -> String
saveSelection model =
    model.selectedSuggestion
        |> Maybe.map encodeSuggestion
        |> Maybe.map (Encode.encode 0)
        |> Maybe.withDefault ""


restoreSelection : String -> Maybe Suggestion
restoreSelection str =
    Decode.decodeString decodeSuggestion str
        |> Result.toMaybe



-- UTIL


parsePosition : String -> Maybe Position
parsePosition input =
    let
        split =
            input
                |> String.split ";"
                |> List.map String.toFloat
    in
    case split of
        [ Ok lat, Ok lng ] ->
            Just { lat = lat, lng = lng }

        _ ->
            Nothing
