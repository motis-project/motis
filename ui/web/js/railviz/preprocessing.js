var RailViz = RailViz || {};

RailViz.Preprocessing = (function () {

  function preprocess(data) {
    data.stations.forEach(preprocessStation);
    data.routes.forEach(route => route.segments.forEach(preprocessSegment));
    data.trains.forEach(train => preprocessTrain(train, data));
    data.routeVertexCount = data.routes.reduce(
      (acc, route) => acc +
        route.segments.reduce(
          (a, s) => a + s.coordinates.coordinates.length, 0),
      0);
  }

  function preprocessSegment(segment) {
    convertPolyline(segment.coordinates);
    const coords = segment.coordinates.coordinates;
    const n_subsegments = (coords.length / 2) - 1;
    segment.subSegmentLengths = new Array(n_subsegments);
    segment.subSegmentOffsets = new Array(n_subsegments);
    let offset = 0;
    for (let i = 0; i < n_subsegments; i++) {
      const from_base = i * 2;
      const to_base = (i + 1) * 2;
      const x_dist = coords[to_base] - coords[from_base];
      const y_dist = coords[to_base + 1] - coords[from_base + 1];
      const length = Math.sqrt(x_dist * x_dist + y_dist * y_dist);
      segment.subSegmentLengths[i] = length;
      segment.subSegmentOffsets[i] = offset;
      offset += length;
    }
    segment.totalLength = offset;
  }

  function preprocessStation(station) {
    var pos = geoToWorldCoords(station.pos.lat, station.pos.lng);
    station.pos.x = pos.x;
    station.pos.y = pos.y;
  }

  function preprocessTrain(train, data) {
    train.route_index = train.route_index || 0;
    train.segment_index = train.segment_index || 0;
    train.clasz = train.clasz || 0;
    train.trip.forEach(trip => {
      trip.train_nr = trip.train_nr || 0
    });
    const route = data.routes[train.route_index];
    const segment = route.segments[train.segment_index];
    train.departureStation =
      findStationById(segment.from_station_id, data.stations);
    train.arrivalStation =
      findStationById(segment.to_station_id, data.stations);
  }

  function findStationById(id, stations) {
    return stations.find(station => station.id == id);
  }

  function convertPolyline(polyline) {
    const coords = polyline.coordinates;
    let converted = [];
    let j = 0;
    for (let i = 0; i < coords.length - 1; i += 2) {
      const wc = geoToWorldCoords(coords[i], coords[i + 1]);
      if (j == 0 || wc.x != converted[j - 2] || wc.y != converted[j - 1]) {
        converted[j] = wc.x;
        converted[j + 1] = wc.y;
        j += 2;
      }
    }
    polyline.coordinates = converted;
  }

  function prepareWalk(walk) {
    preprocessStation(walk.departureStation);
    preprocessStation(walk.arrivalStation);
    walk.from_station_id = walk.departureStation.id;
    walk.to_station_id = walk.arrivalStation.id;
    if (walk.polyline) {
      walk.coordinates = {
        coordinates: walk.polyline
      };
      convertPolyline(walk.coordinates);
    } else {
      walk.coordinates = {
        coordinates: [
          walk.departureStation.pos.x, walk.departureStation.pos.y,
          walk.arrivalStation.pos.x, walk.arrivalStation.pos.y
        ]
      };
    }
  }

  const initialResolution = 2 * Math.PI * 6378137 / 256;
  const originShift = Math.PI * 6378137;

  function geoToWorldCoords(lat, lng) {
    const mx = lng * originShift / 180;
    const my1 =
      Math.log(Math.tan((90 + lat) * Math.PI / 360)) / (Math.PI / 180);
    const my = my1 * originShift / 180;
    const x = (mx + originShift) / initialResolution;
    const y = 256 - ((my + originShift) / initialResolution);

    return {
      x: x,
      y: y
    };
  }

  return {
    preprocess: preprocess,
    convertPolyline: convertPolyline,
    geoToWorldCoords: geoToWorldCoords,
    prepareWalk: prepareWalk
  };

})();
