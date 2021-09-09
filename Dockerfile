FROM alpine:3.14
ARG TARGETARCH
ADD motis-linux-$TARGETARCH/motis-linux-$TARGETARCH.tar.bz2 /
RUN addgroup -S motis && adduser -S motis -G motis && \
    mkdir /data && \
    chown motis:motis /data && \
    echo -e "\
server.static_path=/motis/web \
\
[import] \
paths=schedule:/input/schedule \
paths=osm:/input/osm.pbf \
data_dir=/data \
\
[tiles] \
profile=/motis/tiles-profiles/background.lua \
\
[osrm] \
profiles=/motis/osrm-profiles/car.lua \
profiles=/motis/osrm-profiles/bike.lua \
profiles=/motis/osrm-profiles/bus.lua \
\
[ppr] \
profile=/motis/ppr-profiles/default.json \
" > /system_config.ini
EXPOSE 8080
VOLUME ["/data"]
VOLUME ["/input"]
WORKDIR /motis
USER motis
CMD ["/motis/motis", "--system_config", "/system_config.ini", "-c", "/input/config.ini"]
