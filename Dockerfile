FROM --platform=$BUILDPLATFORM ubuntu:20.04
COPY motis /motis
ENTRYPOINT ["/motis/motis"]
