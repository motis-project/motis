FROM --platform=$BUILDPLATFORM ubuntu:20.04
ARG ARCHIVE
ADD $ARCHIVE /motis
ENTRYPOINT ["/motis/motis"]
