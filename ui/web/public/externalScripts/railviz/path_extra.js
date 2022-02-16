var RailViz = RailViz || {};
RailViz.Path = RailViz.Path || {};

/**
 * Extra paths which are not in the prepared vector tiles
 *  - cause: additional trains, reroutings
 *  - otherwise the trains "fly" without corresponding line
 */
RailViz.Path.Extra = (function () {
  let map = null;
  let data = null;
  let enabled = true;

  function init(_map, beforeId) {
    map = _map;

    map.addSource("railviz-extra", {
      type: "geojson",
      promoteId: "id",
      data: {
        type: "FeatureCollection",
        features: [],
      },
    });

    if (data !== null) {
      map.getSource("railviz-extra").setData(data);
    }

    map.addLayer(
      {
        id: "railviz-extra-line",
        type: "line",
        source: "railviz-extra",
        layout: {
          "line-join": "round",
          "line-cap": "round",
        },
        paint: {
          "line-color": RailViz.Style.lineColor(),
          "line-width": RailViz.Style.lineWidth(),
        },
      },
      beforeId
    );

    setEnabled(enabled);
  }

  function setData(newData) {
    if (!newData) {
      return;
    }

    data = {
      type: "FeatureCollection",
      features: [],
    };

    newData.extras.forEach((e) => {
      data.features.push({
        type: "Feature",
        properties: {},
        geometry: {
          type: "LineString",
          coordinates: flipped(newData.polylines[e].coordinates), // polyline.js
        },
      });
    });

    if (map !== null) {
      map.getSource("railviz-extra").setData(data);
    }
  }

  function setEnabled(b) {
    if (map !== null) {
      if (b) {
        map.setLayoutProperty("railviz-extra-line", "visibility", "visible");
      } else {
        map.setLayoutProperty("railviz-extra-line", "visibility", "none");
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
