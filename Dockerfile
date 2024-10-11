FROM alpine:3.20
ARG TARGETARCH
ADD motis-linux-$TARGETARCH/motis-linux-$TARGETARCH.tar.bz2 /
RUN addgroup -S motis && adduser -S motis -G motis && \
    mkdir /data && \
    chown motis:motis /data
EXPOSE 8080
VOLUME ["/data"]
USER motis
CMD ["/motis", "server", "/data"]