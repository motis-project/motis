var RailViz = RailViz || {};
RailViz.Path = RailViz.Path || {};

RailViz.Path.Extra = (function () {
  let map = null;
  let data = null;

  function init(_map, beforeId) {
    map = _map;

    map.addSource('railviz-paths-extra', {
      "type": "geojson",
      "promoteId": "id",
      "data": {
        type: "FeatureCollection",
        features: []
      }
    });

    if(data !== null) {
      map.getSource("railviz-paths-extra").setData(data);
    }

    map.addLayer({
      "id": "railviz-paths-extra-line",
      "type": "line",
      "source": "railviz-paths-extra",
      "layout": {
        "line-join": "round",
        "line-cap": "round"
      },
      "paint": {
        "line-color": "#888888",
        "line-width": 2.5
      }
    }, beforeId);
  }

  function setData(newData) {
    if(!newData) {
      return;
    }

    data = {
      type: "FeatureCollection",
      features: []
    };

    newData.routes.forEach(route => {
      route.segments.forEach(segment => {
        if(segment.extra) {
          const converted = convertCoordinates(segment.rawCoordinates);
          if(converted.length > 0) {
            data.features.push({
              type: "Feature",
              properties: {},
              geometry: {
                type: "LineString",
                coordinates: converted
              }
            });
          }
        }
      });
    });

    if(map !== null) {
      map.getSource("railviz-paths-extra").setData(data);
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

  return {
    init: init,
    setData: setData
  }
})();
