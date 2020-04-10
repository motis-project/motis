module Widgets.ModePicker exposing
    ( Model
    , Msg
    , PprProfileMode(..)
    , getModes
    , getSearchProfile
    , init
    , saveSelections
    , update
    , view
    )

import Data.Intermodal.Types as Intermodal
import Data.PPR.Decode exposing (decodeSearchOptions)
import Data.PPR.Request exposing (encodeSearchOptions)
import Data.PPR.Types exposing (SearchOptions)
import Html exposing (..)
import Html.Attributes as Attr exposing (..)
import Html.Events exposing (onClick, onInput)
import Html.Lazy exposing (..)
import Json.Decode as Decode
import Json.Decode.Pipeline as JDP exposing (decode, hardcoded, optionalAt, requiredAt)
import Json.Encode as Encode
import Localization.Base exposing (..)
import Maybe.Extra
import Util.Core exposing ((=>))



-- MODEL


type PprProfileMode
    = OnlyDefaultPprProfile
    | PremadePprProfiles


type alias Model =
    { pprMode : PprProfileMode
    , walkEnabled : Bool
    , bikeEnabled : Bool
    , carEnabled : Bool
    , pprSearchOptions : SearchOptions
    , bikeMaxDuration : Int
    , carMaxDuration : Int
    , useCarParking : Bool
    , editorVisible : Bool
    }


type alias SearchProfileEntry =
    ( String, SearchProfileNames -> String )


defaultProfileId : String
defaultProfileId =
    "default"


defaultSearchProfile : SearchOptions
defaultSearchProfile =
    { profile = defaultProfileId, duration_limit = 15 * 60 }


defaultProfileEntry : SearchProfileEntry
defaultProfileEntry =
    ( defaultProfileId, \l -> l.default )


searchProfiles : List SearchProfileEntry
searchProfiles =
    [ defaultProfileEntry
    , ( "accessibility1", \l -> l.accessibility1 )
    , ( "wheelchair", \l -> l.wheelchair )
    , ( "elevation", \l -> l.elevation )
    ]


defaultCarParkingSearchProfile : SearchOptions
defaultCarParkingSearchProfile =
    { defaultSearchProfile | duration_limit = 5 * 60 }


init : Maybe String -> PprProfileMode -> Model
init storedSelections pprMode =
    let
        defaultModel =
            { pprMode = pprMode
            , walkEnabled = True
            , bikeEnabled = False
            , carEnabled = False
            , pprSearchOptions = defaultSearchProfile
            , bikeMaxDuration = 15
            , carMaxDuration = 15
            , useCarParking = True
            , editorVisible = False
            }

        model =
            storedSelections
                |> Maybe.map (restoreSelections defaultModel)
                |> Maybe.withDefault defaultModel
                |> clampValues
    in
    { model
        | pprMode = pprMode
    }


getModes : Model -> List Intermodal.Mode
getModes model =
    [ getWalkMode model
    , getBikeMode model
    , getCarMode model
    ]
        |> Maybe.Extra.values


getWalkMode : Model -> Maybe Intermodal.Mode
getWalkMode model =
    if model.walkEnabled then
        Just (Intermodal.FootPPR { searchOptions = getSearchProfile model })

    else
        Nothing


getBikeMode : Model -> Maybe Intermodal.Mode
getBikeMode model =
    if model.bikeEnabled then
        Just (Intermodal.Bike { maxDuration = model.bikeMaxDuration * 60 })

    else
        Nothing


getCarMode : Model -> Maybe Intermodal.Mode
getCarMode model =
    if model.carEnabled then
        if model.useCarParking then
            Just
                (Intermodal.CarParking
                    { maxCarDuration = model.carMaxDuration * 60
                    , pprSearchOptions = defaultCarParkingSearchProfile
                    }
                )

        else
            Just (Intermodal.Car { maxDuration = model.carMaxDuration * 60 })

    else
        Nothing


getSearchProfile : Model -> SearchOptions
getSearchProfile model =
    case model.pprMode of
        OnlyDefaultPprProfile ->
            defaultSearchProfile

        PremadePprProfiles ->
            model.pprSearchOptions


maxWalkDuration : Model -> Int
maxWalkDuration model =
    30


maxBikeDuration : Model -> Int
maxBikeDuration model =
    30


maxCarDuration : Model -> Int
maxCarDuration model =
    30



-- clamp : number -> number -> number -> number


clamp min max val =
    if val < min then
        min

    else if val > max then
        max

    else
        val


clampWalkDuration : Model -> Int -> Int
clampWalkDuration model val =
    clamp 0 (maxWalkDuration model) val


clampBikeDuration : Model -> Int -> Int
clampBikeDuration model val =
    clamp 0 (maxBikeDuration model) val


clampCarDuration : Model -> Int -> Int
clampCarDuration model val =
    clamp 0 (maxCarDuration model) val



-- UPDATE


type Msg
    = NoOp
    | ToggleEditor
    | ToggleWalk
    | ToggleBike
    | ToggleCar
    | ToggleUseCarParking
    | WalkMaxDurationInput String
    | BikeMaxDurationInput String
    | CarMaxDurationInput String
    | SelectProfile String


update : Msg -> Model -> Model
update msg model =
    case msg of
        NoOp ->
            model

        ToggleEditor ->
            { model | editorVisible = not model.editorVisible }

        ToggleWalk ->
            { model | walkEnabled = not model.walkEnabled }

        ToggleBike ->
            { model | bikeEnabled = not model.bikeEnabled }

        ToggleCar ->
            { model | carEnabled = not model.carEnabled }

        ToggleUseCarParking ->
            { model | useCarParking = not model.useCarParking }

        WalkMaxDurationInput str ->
            case String.toFloat str of
                Ok input ->
                    let
                        newDuration =
                            input
                                |> floor
                                |> clampWalkDuration model
                                |> (\v -> v * 60)
                                |> toFloat

                        pprProfile =
                            model.pprSearchOptions
                    in
                    { model | pprSearchOptions = { pprProfile | duration_limit = newDuration } }

                _ ->
                    model

        BikeMaxDurationInput str ->
            case String.toInt str of
                Ok val ->
                    { model | bikeMaxDuration = clampBikeDuration model val }

                _ ->
                    model

        CarMaxDurationInput str ->
            case String.toInt str of
                Ok val ->
                    { model | carMaxDuration = clampCarDuration model val }

                _ ->
                    model

        SelectProfile id ->
            selectPresetProfile model id


selectPresetProfile : Model -> String -> Model
selectPresetProfile model selectedId =
    let
        profile =
            model.pprSearchOptions

        currentMaxDuration =
            profile.duration_limit
    in
    { model
        | pprSearchOptions = { profile | profile = selectedId, duration_limit = currentMaxDuration }
    }


clampValues : Model -> Model
clampValues model =
    let
        walkDurationLimit =
            model.pprSearchOptions.duration_limit
                |> (\v -> v / 60.0)
                |> floor
                |> clampWalkDuration model
                |> (\v -> v * 60)
                |> toFloat

        pprProfile =
            model.pprSearchOptions

        updatedPprProfile =
            { pprProfile | duration_limit = walkDurationLimit }
    in
    { model
        | bikeMaxDuration = clampBikeDuration model model.bikeMaxDuration
        , carMaxDuration = clampCarDuration model model.carMaxDuration
        , pprSearchOptions = updatedPprProfile
    }



-- VIEW


buttonView : Localization -> Model -> Html Msg
buttonView locale model =
    div [ class "mode-picker-btn", onClick ToggleEditor ]
        [ div
            [ classList
                [ "mode" => True
                , "enabled" => model.walkEnabled
                ]
            ]
            [ i [ class "icon" ] [ text "directions_walk" ] ]
        , div
            [ classList
                [ "mode" => True
                , "enabled" => model.bikeEnabled
                ]
            ]
            [ i [ class "icon" ] [ text "directions_bike" ] ]
        , div
            [ classList
                [ "mode" => True
                , "enabled" => model.carEnabled
                ]
            ]
            [ i [ class "icon" ] [ text "directions_car" ] ]
        ]


walkView : Localization -> Model -> Html Msg
walkView locale model =
    let
        profile =
            model.pprSearchOptions

        lt =
            locale.t.search

        profilesVisible =
            model.pprMode /= OnlyDefaultPprProfile

        profilesView =
            if profilesVisible then
                div [ class "option" ]
                    [ div [ class "label" ] [ text lt.searchProfile.profile ]
                    , searchProfilePickerView locale model
                    ]

            else
                div [] []

        maxDuration =
            toFloat (maxWalkDuration model)
    in
    fieldset
        [ classList
            [ "mode" => True
            , "walk" => True
            , "disabled" => not model.walkEnabled
            ]
        ]
        [ legend [ class "mode-header" ]
            [ label []
                [ input
                    [ type_ "checkbox"
                    , checked model.walkEnabled
                    , onClick ToggleWalk
                    ]
                    []
                , text locale.t.connections.walk
                ]
            ]
        , profilesView
        , div [ class "option" ]
            [ div [ class "label" ] [ text lt.maxDuration ]
            , numericSliderView (profile.duration_limit / 60) 0 maxDuration 1 WalkMaxDurationInput
            ]
        ]


bikeView : Localization -> Model -> Html Msg
bikeView locale model =
    fieldset
        [ classList
            [ "mode" => True
            , "bike" => True
            , "disabled" => not model.bikeEnabled
            ]
        ]
        [ legend [ class "mode-header" ]
            [ label []
                [ input
                    [ type_ "checkbox"
                    , checked model.bikeEnabled
                    , onClick ToggleBike
                    ]
                    []
                , text locale.t.connections.bike
                ]
            ]
        , div [ class "option" ]
            [ div [ class "label" ] [ text locale.t.search.maxDuration ]
            , numericSliderView model.bikeMaxDuration 0 (maxBikeDuration model) 1 BikeMaxDurationInput
            ]
        ]


carView : Localization -> Model -> Html Msg
carView locale model =
    fieldset
        [ classList
            [ "mode" => True
            , "car" => True
            , "disabled" => not model.bikeEnabled
            ]
        ]
        [ legend [ class "mode-header" ]
            [ label []
                [ input
                    [ type_ "checkbox"
                    , checked model.carEnabled
                    , onClick ToggleCar
                    ]
                    []
                , text locale.t.connections.car
                ]
            ]
        , div [ class "option" ]
            [ div [ class "label" ] [ text locale.t.search.maxDuration ]
            , numericSliderView model.carMaxDuration 0 (maxCarDuration model) 1 CarMaxDurationInput
            ]
        , div [ class "option" ]
            [ label []
                [ input
                    [ type_ "checkbox"
                    , checked model.useCarParking
                    , onClick ToggleUseCarParking
                    ]
                    []
                , text locale.t.search.useParking
                ]
            ]
        ]


searchProfilePickerView : Localization -> Model -> Html Msg
searchProfilePickerView locale model =
    let
        createOption ( id, title ) =
            option
                [ value id
                , selected (id == model.pprSearchOptions.profile)
                ]
                [ text (title locale.t.searchProfiles) ]

        includeProfile ( id, title ) =
            case model.pprMode of
                OnlyDefaultPprProfile ->
                    id == defaultProfileId

                PremadePprProfiles ->
                    True

        options =
            searchProfiles
                |> List.filter includeProfile
                |> List.map createOption
    in
    div [ class "profile-picker" ]
        [ select [ onInput SelectProfile ]
            options
        ]


numericSliderView : number -> number -> number -> number -> (String -> Msg) -> Html Msg
numericSliderView val minVal maxVal stepVal tag =
    div [ class "numeric slider control" ]
        [ input
            [ type_ "range"
            , value (toString val)
            , Attr.min (toString minVal)
            , Attr.max (toString maxVal)
            , step (toString stepVal)
            , onInput tag
            ]
            []
        , input
            [ type_ "text"
            , value (toString val)
            , onInput tag
            ]
            []
        ]


editorView : Localization -> String -> Model -> Html Msg
editorView locale label model =
    div
        [ classList
            [ "mode-picker-editor" => True
            , "visible" => model.editorVisible
            ]
        ]
        [ div [ class "header" ]
            [ div [ class "sub-overlay-close", onClick ToggleEditor ]
                [ i [ class "icon" ] [ text "close" ] ]
            , div [ class "title" ] [ text label ]
            ]
        , div [ class "content" ]
            [ walkView locale model
            , bikeView locale model
            , carView locale model
            ]
        ]


modePickerView : Localization -> String -> Model -> Html Msg
modePickerView locale label model =
    div []
        [ buttonView locale model
        , editorView locale label model
        ]


view : Localization -> String -> Model -> Html Msg
view locale label model =
    lazy3 modePickerView locale label model



-- LOCAL STORAGE


encodeModel : Model -> Encode.Value
encodeModel model =
    Encode.object
        [ "walk"
            => Encode.object
                [ "enabled" => Encode.bool model.walkEnabled
                , "search_profile" => encodeSearchOptions model.pprSearchOptions
                ]
        , "bike"
            => Encode.object
                [ "enabled" => Encode.bool model.bikeEnabled
                , "max_duration" => Encode.int model.bikeMaxDuration
                ]
        , "car"
            => Encode.object
                [ "enabled" => Encode.bool model.carEnabled
                , "max_duration" => Encode.int model.carMaxDuration
                , "use_parking" => Encode.bool model.useCarParking
                ]
        ]


decodeModel : Decode.Decoder Model
decodeModel =
    decode Model
        |> hardcoded PremadePprProfiles
        |> requiredAt [ "walk", "enabled" ] Decode.bool
        |> requiredAt [ "bike", "enabled" ] Decode.bool
        |> requiredAt [ "car", "enabled" ] Decode.bool
        |> requiredAt [ "walk", "search_profile" ] decodeSearchOptions
        |> requiredAt [ "bike", "max_duration" ] Decode.int
        |> requiredAt [ "car", "max_duration" ] Decode.int
        |> optionalAt [ "car", "use_parking" ] Decode.bool True
        |> hardcoded False


saveSelections : Model -> String
saveSelections model =
    model
        |> encodeModel
        |> Encode.encode 0


restoreSelections : Model -> String -> Model
restoreSelections base str =
    Decode.decodeString decodeModel str
        |> Result.withDefault base
