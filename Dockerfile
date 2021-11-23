FROM alpine:3.14
ARG TARGETARCH
ADD motis-linux-$TARGETARCH/motis-linux-$TARGETARCH.tar.bz2 /
RUN addgroup -S motis && adduser -S motis -G motis && \
    mkdir /data && \
    chown motis:motis /data && \
    echo -e "\
server.static_path=/motis/web\n\
\n\
intermodal.router=tripbased\n\
\n\
ris.db=/data/ris.mdb\n\
\n\
[import]\n\
paths=schedule:/input/schedule\n\
paths=osm:/input/osm.pbf\n\
data_dir=/data\n\
\n\
[tiles]\n\
profile=/motis/tiles-profiles/background.lua\n\
\n\
[osrm]\n\
profiles=/motis/osrm-profiles/car.lua\n\
profiles=/motis/osrm-profiles/bike.lua\n\
profiles=/motis/osrm-profiles/bus.lua\n\
\n\
[ppr]\n\
profile=/motis/ppr-profiles/default.json\n\
\n" > /system_config.ini
EXPOSE 8080
VOLUME ["/data"]
VOLUME ["/input"]
WORKDIR /motis
USER motis
CMD ["/motis/motis", "--system_config", "/system_config.ini", "-c", "/input/config.ini"]
