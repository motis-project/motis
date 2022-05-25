module Debounce exposing (Config, State, Msg, init, config, update, debounce, debounce1, debounceCmd)

{-|
This modules allows easy usage of debounced events from the view.

# Story

Say you have an elm app where a button produce a `Clicked` message:

```
type alias Model = { ... }
type Msg = Clicked | ...

update : Msg -> Model -> (Model, Cmd Msg)
update msg model =
    case msg of
        ...
        Clicked -> -- perform update for Clicked --
        ...

view model = ... button [ onClick Clicked ] [ text "click me!" ] ...
```

with this module you will be able to change the view using a `deb : Msg -> Msg`
function that will state that the `Clicked` message should be debounced.

```
view model = ... button [ onClick (deb Clicked) ] [ text "click me!" ] ...
```

You will want to specify the timeout for the debounce.
This is usually constant, hence, it does not belongs to the model or state of the app.

```
cfg : Debounce.Config Model Msg
cfg = ... configuration of the debounce component ...

deb : Msg -> Msg
deb = Debounce.debounce cfg
```

In order to create a Debounce.Config you will need to go trough some boilerplate.

1) Extend the model with `Debounce.State` (and initialize it with `Debounce.init`)

```
type alias Model = { ... , d : Debounce.State , ... }
initialModel = { ... , d = Debounce.init , ... }
-- you can choose any name for `d`
```

2) Extend Msg with a wrapper message that will be routed to Debounce module.

```
type Msg = Clicked | Deb (Debounce.Msg Msg)
-- you can choose any name for `Deb`
```

3) Extend `update`

```
update : Msg -> Model -> (Model, Cmd Msg)
update msg model =
    case msg of
        ...
        Clicked -> -- perform update for Clicked --
        ...
        Deb a ->
            Debounce.update cfg a model
```

4) Configure the debounce component using `Debounce.config`

```
cfg : Debounce.Config Model Msg
cfg =
    Debounce.config
        .d                               -- getState   : Model -> Debounce.State
        (\model s -> { model | d = s })  -- setState   : Debounce.State -> Model -> Debounce.State
        Deb                              -- msgWrapper : Msg a -> Msg
        200                              -- timeout ms : Float
```

5) Enjoy!

## Debouncing messages with values (onInput)

If the message that is wanted to be debounced hold data:

```
type Msg = UserInput String
view model = ... input [ onInput UserInput ] [] ...
```

You will need to use `Debounce.debounce1`

```
view model = ... input [ onInput (deb1 UserInput) ] [] ...

deb1 : (a -> Msg) -> (a -> Msg)
deb1 = Debounce.debounce1 cfg
```

## Debouncing messages from the update

If you want to debounce a message generated from the `update`

```
update msg model =
    case msg of
        ... s ... ->
            ( ... , debCmd <| UserInput s )

        UserInput s ->
            ( ... , Cmd.none )

        Deb a ->
            Debounce.update cfg a model

debCmd =
    Debounce.debounceCmd cfg
```

# Functions to create `deb` `deb1` `debCmd` nice helpers

@docs debounce
@docs debounce1
@docs debounceCmd

# Boilerplate functions

@docs init
@docs config
@docs update

# Opaque structures

@docs Config
@docs State
@docs Msg

-}

import Task
import Process
import Helpers


-- a Config msg is represented by a message wrapper and the desired timeout for the debounce


{-|
  Configuration of a debounce component.
-}
type Config model msg
    = Config (model -> State) (model -> State -> model) (Msg msg -> msg) Float


{-|
  State to be included in model.
-}
type State
    = State Int


{-|
  Messages to be wrapped.
-}
type Msg msg
    = Raw msg
    | Deferred Int msg


{-|
  Initial state for the model
-}
init : State
init =
    State 0


{-|
  Creates a configured debounce component.
-}
config : (model -> State) -> (model -> State -> model) -> (Msg msg -> msg) -> Float -> Config model msg
config getState updateState msg delay =
    Config getState updateState msg delay


{-|
  Handle update messages for the debounce component.
-}
update : Config model msg -> Msg msg -> model -> ( model, Cmd msg )
update (Config getState updateState msg delay) deb_msg model =
    case deb_msg of
        Raw m ->
            let
                oldState =
                    (getState model)

                newIncarnation =
                    (incarnation oldState) + 1
            in
                ( (updateState model (setIncarnation oldState newIncarnation))
                , Helpers.deferredCmd delay (msg <| Deferred newIncarnation m)
                )

        Deferred i m ->
            let
                validIncarnation =
                    (incarnation (getState model)) == i
            in
                ( model
                , if validIncarnation then
                    (performMessage m)
                  else
                    Cmd.none
                )


{-|
  Helper function for Msg without parameters.
-}
debounce : Config model msg -> msg -> msg
debounce (Config _ _ msg delay) =
    (\raw_msg -> msg (Raw raw_msg))


{-|
  Helper function for Msg with 1 parameter.
-}
debounce1 : Config model msg -> (a -> msg) -> (a -> msg)
debounce1 (Config _ _ msg delay) =
    (\raw_msg a -> msg (Raw (raw_msg a)))


{-|
  Helper function for deboucing a Cmd.
-}
debounceCmd : Config model msg -> msg -> Cmd msg
debounceCmd cfg msg =
    performMessage <| debounce cfg msg


incarnation (State i) =
    i


setIncarnation (State _) i =
    State i


performMessage : msg -> Cmd msg
performMessage msg =
    Task.perform (always msg) (Task.succeed never)
