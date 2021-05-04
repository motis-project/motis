# Install Elm 0.18

https://github.com/elm-lang/elm-platform/releases

# Build

    elm-make src/Main.elm --output elm.js

## Run

    elm-make src/Main.elm --output elm.js && static


# Automatic recompilation

## Using [devd](https://github.com/cortesi/devd) + [modd](https://github.com/cortesi/modd) (includes live reload)

    modd

## Using elm-live (includes live reload)

    elm-live src/Main.elm --output elm.js

## Using inotify-tools

    while inotifywait -r -e close_write src; do elm-make src/Main.elm --output elm.js; done

## Using fswatch

    fswatch -0 -or src | xargs -0 -n 1 -I {} elm-make src/Main.elm --output elm.js



# URL parameters

* `?motis=8082`: Connect to MOTIS on <window.location.hostname>:8082
* `?motis=host`: Connect to MOTIS on host:8080
* `?motis=host:8082`: Connect to MOTIS on host:8082
* `?time=1488822529`: Set simulation time to unix timestamp
* `?time=2017-03-06T17:48:49+00:00`: Set simulation time to ISO 8601 timestamp
* `?lang=en`: Set language (de/en)
