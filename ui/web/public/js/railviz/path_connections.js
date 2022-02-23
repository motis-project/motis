var RailViz = RailViz || {};
RailViz.Path = RailViz.Path || {};

/**
 * Connection set of a search result
 *  - alternative specific colors (walks color by alternative)
 *  - alternative picking on mouse over
 */
RailViz.Path.Connections = (function () {
  // http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
  const colors = [
    "#1f78b4",
    "#33a02c",
    "#e31a1c",
    "#ff7f00",
    "#6a3d9a",
    "#b15928",
    "#a6cee3",
    "#b2df8a",
    "#fb9a99",
    "#fdbf6f",
    "#cab2d6",
    "#ffff99",
  ];

  let map = null;
  let data = null;
  let enabled = true;

  let connections = null;
  let tripData = null;
  let walkData = null;

  function init(_map, beforeId) {
    map = _map;

    map.addSource("railviz-connections", {
      type: "geojson",
      promoteId: "id",
      data: {
        type: "FeatureCollection",
        features: [],
      },
    });

    if (data !== null) {
      map.getSource("railviz-connections").setData(data);
    }

    map.addLayer(
      {
        id: "railviz-connections-line",
        type: "line",
        source: "railviz-connections",
        layout: {
          "line-join": "round",
          "line-cap": "round",
          "line-sort-key": ["get", "sortKey"],
        },
        paint: {
          "line-color": [
            "string",
            ["feature-state", "color"],
            ["get", "color"],
          ],
          "line-width": [
            "case",
            ["boolean", ["feature-state", "highlight"], false],
            10,
            5,
          ],
        },
      },
      beforeId
    );

    setEnabled(enabled);
  }

  function minCIdToColor(minCId) {
    const c = minCId % colors.length;
    return colors[c < 0 ? colors.length + c : c];
  }

  function setData(_connections, _tripData, _walkData) {
    data = {
      type: "FeatureCollection",
      features: [],
    };

    connections = _connections;
    tripData = _tripData;
    walkData = _walkData;

    if (tripData) {
      tripData.polylines.forEach((p, i) => {
        if (!p.trains) {
          return; // not part of any visible connection segment
        }

        data.features.push({
          type: "Feature",
          properties: {
            id: i,
            trip: true,
            sortKey: p.minCId > 0 ? -p.minCId : p.minCId + 1e9,
            color: minCIdToColor(p.minCId),
          },
          geometry: {
            type: "LineString",
            coordinates: flipped(p.coordinates), // from polyline.js
          },
        });
      });
    }

    if (walkData) {
      walkData.forEach((w) => {
        if (w.error || !w.polyline) {
          return;
        }
        const coords = RailViz.Model.doublesToLngLats(w.polyline.coordinates);
        if (coords.length == 0) {
          return;
        }

        data.features.push({
          type: "Feature",
          properties: {
            id: 1e9 + w.id,
            trip: false,
            sortKey: w.minCId > 0 ? -w.minCId : w.minCId + 1e9,
            color: minCIdToColor(w.minCId),
          },
          geometry: {
            type: "LineString",
            coordinates: coords,
          },
        });
      });
    }

    if (map !== null) {
      map.getSource("railviz-connections").setData(data);
    }
  }

  function setEnabled(b) {
    if (map !== null) {
      map.setLayoutProperty(
        "railviz-connections-line",
        "visibility",
        b ? "visible" : "none"
      );
    }
    enabled = b;
  }

  function highlightConnections(connectionIds) {
    if (map === null) {
      return;
    }

    map.removeFeatureState({ source: "railviz-connections" });
    if (!connectionIds || connectionIds.length == 0) {
      return;
    }

    let minCIds = new Map(); // feature id -> minCid
    connectionIds.forEach((cid) => {
      if (connections === null) {
        return; // catch racing condition during initial load
      }
      const c = connections.find((c) => c.id == cid);
      if (c === undefined) {
        return; // catch racing condition on "later" click
      }

      (c.resolved || []).forEach((t) => {
        t.indices.forEach((i) => {
          let curr = minCIds.get(Math.abs(i));
          let next = RailViz.ConnectionManager.cIdCombine(curr, c.id);
          if (curr != next) {
            minCIds.set(Math.abs(i), next);
          }
        });
      });

      (c.walks || []).forEach((w) => {
        let curr = minCIds.get(1e9 + w.id);
        let next = RailViz.ConnectionManager.cIdCombine(curr, c.id);
        if (curr != next) {
          minCIds.set(1e9 + w.id, next);
        }
      });
    });
    minCIds.forEach((minCId, id) => {
      map.setFeatureState(
        { source: "railviz-connections", id: id },
        { highlight: true, color: minCIdToColor(minCId) }
      );
    });
  }

  function getPickedConnections(features) {
    const f = features.find((e) => e.layer.id == "railviz-connections-line");
    if (f === undefined) {
      return null;
    }

    if (f.properties.trip) {
      return tripData.polylines[f.id].trains;
    } else {
      return walkData[f.id - 1e9];
    }
  }

  return {
    init: init,
    setData: setData,
    setEnabled: setEnabled,
    getPickedConnections: getPickedConnections,
    highlightConnections: highlightConnections,
  };
})();
