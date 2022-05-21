var RailViz = RailViz || {};
RailViz.Path = RailViz.Path || {};

/**
 * Base Map : Prerendered vector tiles from path module
 */
RailViz.Path.Base = (function () {
  let map = null;
  let enabled = true;

  function init(_map, tilesEndpoint) {
    map = _map;

    map.addSource("railviz-base", {
      type: "vector",
      tiles: [`${tilesEndpoint}path/tiles/{z}/{x}/{y}.mvt`],
      maxzoom: 20,
    });

    map.addLayer({
      id: "railviz-base-line",
      type: "line",
      source: "railviz-base",
      "source-layer": "path",
      layout: {
        "line-join": "round",
        "line-cap": "round",
        "line-sort-key": [
          "*",
          ["-", ["get", "min_class"]],
          ["case", ["get", "stub"], 1, 2],
        ],
      },
      paint: {
        "line-color": RailViz.Style.lineColor(),
        "line-width": RailViz.Style.lineWidth(),
      },
    });

    map.addLayer({
      id: "railviz-base-stations",
      type: "circle",
      source: "railviz-base",
      "source-layer": "station",
      paint: {
        "circle-color": "white",
        "circle-radius": RailViz.Style.stationCircleRadius(),
        "circle-stroke-color": "#666666",
        "circle-stroke-width": RailViz.Style.stationCircleWidth(),
        "circle-pitch-alignment": "map",
      },
    });


    map.loadImage(`${tilesEndpoint}img/bike_marker.png`, (error, image) => {
      if (error) {
        throw error;
      }

      map.addImage("icon-bike", image);
      map.addSource("gbfs-default", {
        type: "vector",
        tiles: [`${tilesEndpoint}gbfs/tiles/default/{z}/{x}/{y}.mvt`],
        maxzoom: 20,
      });
      map.addLayer({
        id: "gbfs-stations",
        type: "symbol",
        source: "gbfs-default",
        minzoom: 14,
        "source-layer": "station",
        layout: {
          "icon-image": "icon-bike",
          "text-field": ['get', 'name'],
          "text-anchor": "top",
          "text-offset": [0, 1],
          "text-font": ["Noto Sans Display Regular"],
          "icon-size": ['interpolate', ['linear'], ['zoom'], 14, 0.7, 16, 1]
        },
      });
    });

    setEnabled(enabled);
  }

  function setEnabled(b) {
    if (map !== null) {
      ["railviz-base-line", "railviz-base-stations", "gbfs-stations"].forEach((l) => {
        if (map.getLayer(l) !== 'undefined') {
          map.setLayoutProperty(l, "visibility", b ? "visible" : "none");
        }
      });
    }
    enabled = b;
  }

  return {
    init: init,
    setEnabled: setEnabled,
  };
})();
