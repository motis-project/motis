var RailViz = RailViz || {};
RailViz.Path = RailViz.Path || {};

RailViz.Path.Trips = (function () {
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

    map.addSource("railviz-path-trips", {
      type: "geojson",
      promoteId: "id",
      data: {
        type: "FeatureCollection",
        features: [],
      },
    });

    if (data !== null) {
      map.getSource("railviz-path-trips").setData(data);
    }

    map.addLayer(
      {
        id: "railviz-path-trips-line",
        type: "line",
        source: "railviz-path-trips",
        layout: {
          "line-join": "round",
          "line-cap": "round",
        },
        paint: {
          "line-color": ["get", "color"],
          "line-width": ["case", ["get", "highlight"], 4.1, 2.6],
        },
      },
      beforeId
    );

    setEnabled(enabled);
  }

  function setData(newData, tripIds) {
    if (!newData) {
      return;
    }

    data = {
      type: "FeatureCollection",
      features: [],
    };

    newData.routes.forEach((route, route_idx) => {
      route.segments.forEach((segment, segment_idx) => {
        const converted = convertCoordinates(segment.rawCoordinates);
        if (converted.length > 0) {
          let highlight = false;
          let clasz = colors.length - 1;

          newData.trains.forEach((train) => {
            if (
              train.route_index != route_idx ||
              train.segment_index != segment_idx
            ) {
              return;
            }

            clasz = Math.min(clasz, train.clasz);
            highlight |= train.trip.some((this_trp) =>
              tripIds.some((other_trp) => deepEqual(this_trp, other_trp))
            );
          });

          data.features.push({
            type: "Feature",
            properties: {
              highlight: !!highlight,
              color: colors[clasz],
            },
            geometry: {
              type: "LineString",
              coordinates: converted,
            },
          });
        }
      });
    });

    if (map !== null) {
      map.getSource("railviz-path-trips").setData(data);
    }
  }

  function convertCoordinates(src) {
    let converted = [];
    for (let i = 0; i < src.length - 1; i += 2) {
      const ll = [src[i + 1], src[i]];
      if (
        converted.length == 0 ||
        ll[0] != converted[converted.length - 1][0] ||
        ll[1] != converted[converted.length - 1][1]
      ) {
        converted.push(ll);
      }
    }
    return converted;
  }

  function setEnabled(b) {
    if (map !== null) {
      if (b) {
        map.setLayoutProperty(
          "railviz-path-trips-line",
          "visibility",
          "visible"
        );
      } else {
        map.setLayoutProperty("railviz-path-trips-line", "visibility", "none");
      }
    }
    enabled = b;
  }

  return {
    init: init,
    setData: setData,
    setEnabled: setEnabled,
  };
})();
