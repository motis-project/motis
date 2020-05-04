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
        "line-color": [
          "case",
          ["get", "stub"],
          "hsl(0, 0, 70%)",
          ["<", ["get", "min_class"], 6],
          "hsl(0, 0, 30%)",
          ["<", ["get", "min_class"], 8],
          "hsl(0, 0, 40%)",
          "hsl(0, 0, 50%)",
        ],
        "line-width": [
          "case",
          ["get", "stub"],
          2,
          ["<", ["get", "min_class"], 6],
          4,
          ["<", ["get", "min_class"], 8],
          3.25,
          2.5,
        ],
      },
    });

    map.addLayer({
      id: "railviz-base-stations",
      type: "circle",
      source: "railviz-base",
      "source-layer": "station",
      paint: {
        "circle-color": "white",
        "circle-radius": 2.25,
        "circle-stroke-color": "#333333",
        "circle-stroke-width": 2,
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
