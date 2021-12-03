# Web Frontend Rewrite

## Install dependencies
  
Navigate into the `ui/web` directory and install the needed dependencies using

    yarn

## Start the development Server

    yarn dev

# URL parameters

* `?motis=8082`: Connect to MOTIS on <window.location.hostname>:8082
* `?motis=host`: Connect to MOTIS on host:8080
* `?motis=host:8082`: Connect to MOTIS on host:8082
* `?time=1488822529`: Set simulation time to unix timestamp
* `?time=2017-03-06T17:48:49+00:00`: Set simulation time to ISO 8601 timestamp
* `?lang=en`: Set language (de/en)
