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
        "line-sort-key": ["case", ["get", "stub"], 0, 1],
      },
      paint: {
        "line-color": ["case", ["get", "stub"], "#777777", "#666666"],
        "line-width": ["case", ["get", "stub"], 2.5, 4],
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
