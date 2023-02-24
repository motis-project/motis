module Widgets.TagList exposing
    ( Model
    , Msg
    , Tag(..)
    , getSelectedTags
    , init
    , saveSelections
    , subscriptions
    , update
    , view
    )

import Html exposing (..)
import Html.Attributes exposing (..)
import Html.Events exposing (onClick, onInput)
import Html.Lazy exposing (..)
import Json.Decode as Decode
import Json.Decode.Pipeline as JDP exposing (decode, hardcoded, optional, required, requiredAt)
import Json.Encode as Encode
import List.Extra
import Localization.Base exposing (..)
import Mouse
import Util.Core exposing ((=>))
import Util.List exposing ((!!))
import Util.View exposing (onStopAll, onStopPropagation)



-- MODEL


type alias Model =
    { tags : List Tag
    , selected : List Tag
    , visible : Bool
    , ignoreNextToggle : Bool
    }


type Tag
    = WalkTag TagOptions
    | BikeTag TagOptions
    | CarTag TagOptions
    | OnDemandTag TagOptions


type alias TagOptions =
    { maxDuration : Int }


defaultMaxDuration : Int
defaultMaxDuration =
    900


init : Maybe String -> Model
init storedSelections =
    { tags =
        [ WalkTag { maxDuration = defaultMaxDuration }
        , BikeTag { maxDuration = defaultMaxDuration }
        , CarTag { maxDuration = defaultMaxDuration }
        , OnDemandTag { maxDuration = defaultMaxDuration }
        ]
    , selected =
        storedSelections
            |> Maybe.map restoreSelections
            |> Maybe.withDefault []
    , visible = False
    , ignoreNextToggle = False
    }


getSelectedTags : Model -> List Tag
getSelectedTags model =
    model.selected


getTagOptions : Tag -> TagOptions
getTagOptions tag =
    case tag of
        WalkTag o ->
            o

        BikeTag o ->
            o

        CarTag o ->
            o

        OnDemandTag o ->
            o



updateTagOptions : (TagOptions -> TagOptions) -> Tag -> Tag
updateTagOptions f tag =
    case tag of
        WalkTag o ->
            WalkTag (f o)

        BikeTag o ->
            BikeTag (f o)

        CarTag o ->
            CarTag (f o)

        OndemandTag o ->
            OnDemandTag (f o)


-- UPDATE


type Msg
    = AddTag Tag
    | RemoveTag Tag
    | ToggleVisibility
    | Click
    | TagDurationInput Tag String
    | NoOp


update : Msg -> Model -> Model
update msg model =
    case msg of
        NoOp ->
            model

        AddTag t ->
            { model
                | selected = model.selected ++ [ t ]
                , visible = False
                , ignoreNextToggle = True
            }

        RemoveTag t ->
            { model | selected = List.filter (\s -> s /= t) model.selected }

        ToggleVisibility ->
            if model.ignoreNextToggle then
                { model | ignoreNextToggle = False }

            else
                { model | visible = not model.visible }

        Click ->
            { model | visible = False }

        TagDurationInput t input ->
            let
                duration =
                    String.toInt input
                        |> Result.map (\v -> v * 60)
                        |> Result.withDefault 0

                t_ =
                    updateTagOptions (\o -> { o | maxDuration = duration }) t

                selected_ =
                    List.Extra.replaceIf (\x -> x == t) t_ model.selected
            in
            { model | selected = selected_ }



-- VIEW


tagListView : Localization -> String -> Model -> Html Msg
tagListView locale label model =
    let
        availableTags =
            List.filter (\t -> not (List.member t model.selected)) model.tags
                |> List.map
                    (\t ->
                        a [ class "tag", onClick (AddTag t) ]
                            [ i [ class "icon" ] [ text (tagIcon t) ] ]
                    )
                |> div
                    [ classList
                        [ ( "add-tag-popup", True )
                        , ( "paper", True )
                        , ( "hide", not model.visible )
                        ]
                    , onStopAll "mousedown" NoOp
                    ]

        addButton =
            if List.length model.selected == List.length model.tags then
                []

            else
                [ div [ class "tag outline clickable", onClick ToggleVisibility ]
                    ([ i [ class "icon" ] [ text "add" ] ]
                        ++ [ availableTags ]
                    )
                ]

        selectedTags =
            model.selected
                |> List.map (tagView locale)
    in
    div [ class "tag-list" ]
        [ div [ class "label" ] [ text label ]
        , div [ class "tags" ]
            (selectedTags ++ addButton)
        ]


tagView : Localization -> Tag -> Html Msg
tagView locale tag =
    let
        tagOptions =
            getTagOptions tag

        maxDuration =
            toString (tagOptions.maxDuration // 60)
    in
    div [ class "tag" ]
        [ i [ class "tag-icon icon" ] [ text (tagIcon tag) ]
        , i [ class "remove icon", onClick (RemoveTag tag) ] [ text "cancel" ]
        , div [ class "options" ]
            [ input
                [ type_ "text"
                , attribute "inputmode" "numeric"
                , attribute "pattern" "[0-9]+"
                , value maxDuration
                , title locale.t.search.maxDuration
                , onInput (TagDurationInput tag)
                ]
                []
            ]
        ]


tagIcon : Tag -> String
tagIcon tag =
    case tag of
        WalkTag _ ->
            "directions_walk"

        BikeTag _ ->
            "directions_bike"

        CarTag _ ->
            "directions_car"

        OnDemandTag _ ->
            "directions_ondemand"


view : Localization -> String -> Model -> Html Msg
view locale label model =
    lazy3 tagListView locale label model



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions model =
    if model.visible then
        Mouse.downs (\_ -> Click)

    else
        Sub.none



-- LOCAL STORAGE


encodeTag : Tag -> Encode.Value
encodeTag tag =
    case tag of
        WalkTag o ->
            Encode.object
                [ "type" => Encode.string "Walk"
                , "max_duration" => Encode.int o.maxDuration
                ]

        BikeTag o ->
            Encode.object
                [ "type" => Encode.string "Bike"
                , "max_duration" => Encode.int o.maxDuration
                ]

        CarTag o ->
            Encode.object
                [ "type" => Encode.string "Car"
                , "max_duration" => Encode.int o.maxDuration
                ]

        OnDemandTag o ->
            Encode.object
                [ "type" => Encode.string "OnDemand"
                , "max_duration" => Encode.int o.maxDuration
                ]


decodeTag : Decode.Decoder Tag
decodeTag =
    let
        parseTag : String -> Decode.Decoder Tag
        parseTag type_ =
            case type_ of
                "Walk" ->
                    decodeOptions
                        |> Decode.map WalkTag

                "Bike" ->
                    decodeOptions
                        |> Decode.map BikeTag

                "Car" ->
                    decodeOptions
                        |> Decode.map CarTag

                "OnDemand" ->
                    decodeOptions
                        |> Decode.map OnDemandTag

                _ ->
                    Decode.fail "unknown tag type"

        decodeOptions : Decode.Decoder TagOptions
        decodeOptions =
            decode TagOptions
                |> JDP.required "max_duration" Decode.int
    in
    Decode.field "type" Decode.string |> Decode.andThen parseTag


saveSelections : Model -> String
saveSelections model =
    model.selected
        |> List.map encodeTag
        |> Encode.list
        |> Encode.encode 0


restoreSelections : String -> List Tag
restoreSelections str =
    Decode.decodeString (Decode.list decodeTag) str
        |> Result.withDefault []
