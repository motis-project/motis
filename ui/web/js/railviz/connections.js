var RailViz = RailViz || {};

RailViz.Connections = (function () {

  // http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
  const colors = [
    "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
    "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"
  ];

  let map = null;
  let data = null;
  let enabled = true;

  let lowestConnId = 0;
  let trainSegments = [];
  let walkSegments = [];

  let connectionToSegments = new Map();
  let segmentToConnections = new Map();

  function getOrCreate(map, k, fn) {
    let v = map.get(k);
    if(v === undefined) {
      v = fn();
      map.set(k, v);
    }
    return v;
  }

  function init(_map, beforeId) {
    map = _map;

    map.addSource('railviz-connections', {
      "type": "geojson",
      "promoteId": "id",
      "data": {
        type: "FeatureCollection",
        features: []
      }
    });

    if(data !== null) {
      map.getSource("railviz-connections").setData(data);
    }

    map.addLayer({
      "id": "railviz-connections-line",
      "type": "line",
      "source": "railviz-connections",
      "layout": {
        "line-join": "round",
        "line-cap": "round"
      },
      "paint": {
        "line-color": ["string", ["feature-state", "color"], ["get", "color"]],
        "line-width": ["case", ["boolean", ["feature-state", "highlight"], false ], 10, 5]
      }
    }, beforeId);

    setEnabled(enabled);
  }

  function setData(newTrainSegments, newWalkSegments, newLowestConnId) {
    data = {
      type: "FeatureCollection",
      features: []
    };

    lowestConnId = newLowestConnId;
    trainSegments = newTrainSegments ? Array.from(newTrainSegments.values()) : [];
    walkSegments = newWalkSegments ? Array.from(newWalkSegments.values()) : [];

    connectionToSegments.clear();
    segmentToConnections.clear();

    let segment_id = 0;
    if(newTrainSegments) {
      trainSegments.forEach(ts => {
          const converted = convertCoordinates(ts.segment.coordinates.coordinates); // wat
          if(converted.length > 0) {
            data.features.push({
              type: "Feature",
              properties: {
                "id": segment_id,
                "color": colors[ts.color % colors.length]
              },
              geometry: {
                type: "LineString",
                coordinates: converted
              }
            });

            ts.trips.forEach(t =>
              t.connectionIds.forEach(c => getOrCreate(connectionToSegments, c, Array).push(segment_id)));
              segmentToConnections.set(segment_id, ts.trips.flatMap(t => t.connectionIds));
          }

          segment_id++;
        });
    }

    if(newWalkSegments) {
      walkSegments.forEach(ws => {
          const converted = convertCoordinates(ws.polyline.coordinates); // wat
          if(converted.length > 0) {
            data.features.push({
              type: "Feature",
              properties: {
                "id": segment_id,
                "color": colors[ws.color % colors.length]
              },
              geometry: {
                type: "LineString",
                coordinates: converted
              }
            });

            ws.connectionIds.forEach(c => getOrCreate(connectionToSegments, c, Array).push(segment_id));
            segmentToConnections.set(segment_id, ws.connectionIds);
          }

          segment_id++;
        });
    }

    if(map !== null) {
      map.getSource("railviz-connections").setData(data);
    }
  }

  function convertCoordinates(src) {
    let converted = [];
    for (let i = 0; i < src.length - 1; i += 2) {
      const ll = [src[i+1], src[i]];
      if (converted.length == 0 ||
        ll[0] != converted[converted.length-1][0] ||
        ll[1] != converted[converted.length-1][1]) {
          converted.push(ll);
      }
    }
    return converted;
  }

  function setEnabled(b) {
    if(map !== null) {
      if(b) {
        map.setLayoutProperty("railviz-connections-line", 'visibility', 'visible');
      } else {
        map.setLayoutProperty("railviz-connections-line", 'visibility', 'none');
      }
    }
    enabled = b;
  }

  function highlightConnections(connectionIds) {
    if(map === null) {
      return;
    }

    map.removeFeatureState({"source": "railviz-connections"});
    if (!connectionIds || connectionIds.length == 0) {
      return;
    }

    let highlightColors = new Map();
    connectionIds.forEach(cid => {
      const sids = connectionToSegments.get(cid)
      if(sids) {
        sids.forEach(sid =>
          getOrCreate(highlightColors, sid, () => {
            const color = segmentToConnections
                            .get(sid).reduce((acc, c) => connectionIds.includes(c) ? Math.min(acc, c) : acc, cid);
            return colors[(color - lowestConnId) % colors.length];
          }));
      }
    });

    highlightColors.forEach((c, sid) => map.setFeatureState({"source": "railviz-connections", "id": sid},
                                                            {"highlight": true, "color": c }));
  }

  function getPickedSegment(features) {
    const f = features.find(e => e.layer.id == 'railviz-connections-line');
    if (f === undefined) {
      return null;
    }

    if (f.id >= 0 && f.id < trainSegments.length) {
      return trainSegments[f.id];
    } else if (f.id >= trainSegments.length && f.id <= trainSegments.length + walkSegments.length) {
      return walkSegments[f.id - trainSegments.length];
    }
  }

  return {
    init: init,
    setData: setData,
    setEnabled: setEnabled,
    getPickedSegment: getPickedSegment,
    highlightConnections: highlightConnections
  }
})();
