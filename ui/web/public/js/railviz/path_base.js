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

    setEnabled(enabled);
  }

  function setEnabled(b) {
    if (map !== null) {
      ["railviz-base-line", "railviz-base-stations"].forEach((l) => {
        map.setLayoutProperty(l, "visibility", b ? "visible" : "none");
      });
    }
    enabled = b;
  }

  return {
    init: init,
    setEnabled: setEnabled,
  };
})();
