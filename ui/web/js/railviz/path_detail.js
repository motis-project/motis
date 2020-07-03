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

  // prettier-ignore
  const hslValues = [
    [200, 16, 62], //  0 : AIR   : #90a4ae
    [291, 64, 42], //  1 : ICE   : #9c27b0
    [340, 82, 52], //  2 : IC    : #e91e63
    [88, 50, 60],  //  3 : COACH : #9ccc65
    [235, 66, 30], //  4 : N     : #1a237e
    [4, 90, 58],   //  5 : RE    : #f44336
    [4, 90, 58],   //  6 : RB    : #f44336
    [122, 39, 49], //  7 : S     : #4caf50
    [231, 48, 48], //  8 : U     : #3f51b5
    [36, 100, 50], //  9 : STR   : #ff9800
    [36, 100, 50], // 10 : BUS   : #ff9800
    [187,100, 38], // 11 : SHIP  : #00acc1
    [0, 0, 62],    // 12 : OTHER : #9e9e9e
  ];

  const colors = hslValues.flatMap((c, i) => [
    i,
    `hsl(${c[0]}, ${c[1]}%, ${c[2]}%)`,
  ]);
  const mutedColors = hslValues.flatMap((c, i) => [
    i,
    `hsl(${c[0]}, ${Math.round(c[1] * 0.66)}%, ${Math.round(c[2] * 1.33)}%)`,
  ]);

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
          "line-sort-key": ["case", ["get", "highlight"], 1, 0],
        },
        paint: {
          "line-color": [
            "case",
            ["get", "highlight"],
            ["match", ["get", "clasz"], ...colors, "#222222"],
            ["match", ["get", "clasz"], ...mutedColors, "#222222"],
          ],
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

    trips.trains.forEach((t) => {
      t.highlight = t.trip.some((this_trp) =>
        filter.trains.some(
          (filter_train) =>
            deepEqual(filter_train.trip, this_trp) &&
            filter_train.sections.some(
              (filter_sec) =>
                filter_sec.departureStation.id == t.d_station_id &&
                filter_sec.arrivalStation.id == t.a_station_id
            )
        )
      );

      t.polylines.forEach((p) => {
        p.polyline.clasz = Math.min(p.polyline.clasz || 12, t.clasz);
        p.polyline.highlight |= t.highlight;
      });
    });

    trips.polylines.forEach((p) => {
      if (!p.coordinates) {
        return;
      }
      data.features.push({
        type: "Feature",
        properties: {
          highlight: !!p.highlight,
          clasz: p.clasz,
        },
        geometry: {
          type: "LineString",
          coordinates: flipped(p.coordinates), // from polyline.js
        },
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
    const wp = walk.polyline || [];

    // train_arrival -> walk_departure
    const walk_d_station_id = walk.departureStation.id;
    const trainArrivalCoords = trips.trains.reduce((acc, t) => {
      if (t.a_station_id == walk_d_station_id && t.highlight) {
        acc.push(t.lastCoordinate());
      }
      return acc;
    }, []);
    if (trainArrivalCoords.length == 1) {
      const [lat, lng] = trainArrivalCoords[0];
      if (replace || wp[0] != lat || wp[1] != lng) {
        wp.unshift(lat, lng);
      }
    }

    // walk_arrival -> train_departure
    const walk_a_station_id = walk.arrivalStation.id;
    const trainDepartureCoords = trips.trains.reduce((acc, t) => {
      if (t.d_station_id == walk_a_station_id && t.highlight) {
        acc.push(t.firstCoordinate());
      }
      return acc;
    }, []);
    if (trainDepartureCoords.length == 1) {
      const [lat, lng] = trainDepartureCoords[0];
      if (replace || wp[wp.length - 2] != lat || wp[wp.length - 1] != lng) {
        wp.push(lat, lng);
      }
    }

    return RailViz.Model.doublesToLngLats(wp);
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
