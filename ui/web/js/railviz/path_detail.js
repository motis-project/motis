var RailViz = RailViz || {};
RailViz.Path = RailViz.Path || {};

/**
 * Bunch of trips and walks
 *  - single journey after click on connection result list
 *  - single trip (+ join/through services) (click on train, station table, ...)
 *  - train color by category, walk color black
 */
RailViz.Path.Detail = (function () {
  let map = null;
  let data = null;
  let enabled = true;

  const colors = [
    "#9c27b0",
    "#e91e63",
    "#1a237e",
    "#f44336",
    "#f44336",
    "#4caf50",
    "#3f51b5",
    "#ff9800",
    "#ff9800",
    "#9e9e9e",
  ];

  function init(_map, beforeId) {
    map = _map;

    map.addSource("railviz-detail", {
      type: "geojson",
      data: {
        type: "FeatureCollection",
        features: [],
      },
    });

    if (data !== null) {
      map.getSource("railviz-detail").setData(data);
    }

    map.addLayer(
      {
        id: "railviz-detail-line",
        type: "line",
        source: "railviz-detail",
        layout: {
          "line-join": "round",
          "line-cap": "round",
        },
        paint: {
          "line-color": ["get", "color"],
          "line-width": ["case", ["get", "highlight"], 4.1, 2.6],
        },
        filter: ["==", "$type", "LineString"],
      },
      beforeId
    );

    map.addLayer(
      {
        id: "railviz-detail-stations",
        type: "circle",
        source: "railviz-detail",
        paint: {
          "circle-color": "white",
          "circle-radius": ["case", ["get", "important"], 3.5, 2.25],
          "circle-stroke-color": "#333333",
          "circle-stroke-width": 2,
        },
        filter: ["==", "$type", "Point"],
      },
      beforeId
    );

    setEnabled(enabled);
  }

  function setData(filter, trips) {
    if (!trips) {
      return;
    }

    data = {
      type: "FeatureCollection",
      features: [],
    };


    trips.routes.forEach((route, route_idx) => {
      route.segments.forEach((segment, segment_idx) => {
        const coords = RailViz.Preprocessing.toLatLngs(segment.rawCoordinates);
        if (coords.length == 0) {
          return;
        }
        segment.highlight = false;
        segment.clasz = colors.length - 1;

        trips.trains.forEach((train) => {
          if (
            train.route_index != route_idx ||
            train.segment_index != segment_idx
          ) {
            return;
          }

          segment.clasz = Math.min(segment.clasz, train.clasz);
          segment.highlight |= train.trip.some((this_trp) =>
            filter.trains.some(
              (filter_train) =>
                deepEqual(filter_train.trip, this_trp) &&
                filter_train.sections.some(
                  (filter_sec) =>
                    filter_sec.departureStation.id == segment.from_station_id &&
                    filter_sec.arrivalStation.id == segment.to_station_id
                )
            )
          );
        });

        data.features.push({
          type: "Feature",
          properties: {
            highlight: !!segment.highlight,
            color: colors[segment.clasz],
          },
          geometry: {
            type: "LineString",
            coordinates: coords,
          },
        });
      });
    });

    trips.stations.forEach((station) => {
      data.features.push({
        type: "Feature",
        properties: {
          id: station.id,
          name: station.name,
          important: false,
        },
        geometry: {
          type: "Point",
          coordinates: [station.rawPos.lng, station.rawPos.lat],
        },
      });
    });

    filter.walks.forEach((walk) => {
      if (walk.polyline == null) {
        return;
      }

      const coords = loadAdjustedWalk(walk, trips);
      if (coords.length == 0) {
        return;
      }

      data.features.push({
        type: "Feature",
        properties: {
          highlight: true,
          color: "#222222",
        },
        geometry: {
          type: "LineString",
          coordinates: coords,
        },
      });
    });

    filter.interchangeStations.forEach((station) => {
      data.features.push({
        type: "Feature",
        properties: {
          id: station.id,
          name: station.name,
          important: true,
        },
        geometry: {
          type: "Point",
          coordinates: [station.pos.lng, station.pos.lat],
        },
      });
    });

    if (map !== null) {
      map.getSource("railviz-detail").setData(data);
    }
  }

  function loadAdjustedWalk(walk, trips) {
    const replace = !walk.polyline;
    const from_station_id = walk.departureStation.id;
    const to_station_id = walk.arrivalStation.id;
    const startSegments = trips.routes.reduce(
      (acc, r) =>
        acc.concat(
          r.segments.filter(
            (seg) => seg.to_station_id == from_station_id && seg.highlight
          )
        ),
      []
    );
    const coords = walk.polyline || [];
    if (startSegments.length == 1) {
      const fromCoords = startSegments[0].rawCoordinates;
      const x = fromCoords[fromCoords.length - 2];
      const y = fromCoords[fromCoords.length - 1];
      if (replace) {
        coords[0] = x;
        coords[1] = y;
      } else {
        if (coords[0] != x || coords[1] != y) {
          coords.unshift(x, y);
        }
      }
    }
    const endSegments = trips.routes.reduce(
      (acc, r) =>
        acc.concat(
          r.segments.filter(
            (seg) => seg.from_station_id == to_station_id && seg.highlight
          )
        ),
      []
    );
    if (endSegments.length == 1) {
      const toCoords = endSegments[0].rawCoordinates;
      const x = toCoords[0];
      const y = toCoords[1];
      if (replace) {
        coords[2] = x;
        coords[3] = y;
      } else {
        if (coords[coords.length - 2] != x || coords[coords.length - 1] != y) {
          coords.push(x, y);
        }
      }
    }

    return RailViz.Preprocessing.toLatLngs(coords);
  }

  function setEnabled(b) {
    if (map !== null) {
      ["railviz-detail-line", "railviz-detail-stations"].forEach((l) =>
        map.setLayoutProperty(l, "visibility", b ? "visible" : "none")
      );
    }
    enabled = b;
  }

  return {
    init: init,
    setData: setData,
    setEnabled: setEnabled,
  };
})();
