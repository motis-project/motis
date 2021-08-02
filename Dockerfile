FROM ubuntu:20.04
ARG TARGETARCH
ADD motis-linux-$TARGETARCH/motis-linux-$TARGETARCH.tar.bz2 /
RUN useradd --user-group --create-home --shell /bin/bash motis
EXPOSE 8080
VOLUME ["/data"]
WORKDIR /motis
USER motis
CMD ["/motis/motis", "-c", "/data/config.ini"]
