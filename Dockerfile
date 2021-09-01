FROM alpine:3.14
ARG TARGETARCH
ADD motis-linux-$TARGETARCH/motis-linux-$TARGETARCH.tar.bz2 /
RUN addgroup -S motis && adduser -S motis -G motis
EXPOSE 8080
VOLUME ["/data"]
WORKDIR /motis
USER motis
CMD ["/motis/motis", "-c", "/data/config.ini"]
