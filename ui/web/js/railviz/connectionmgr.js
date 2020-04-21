var RailViz = RailViz || {};

RailViz.ConnectionManager = (function () {

  let connections = [];
  let connectionIdToIdx = new Map();
  let tripToConnections = new Map();
  let trainSegments = new Map();
  let walkSegments = new Map();
  let lastTripsRequest = null;
  let tripsComplete = false;
  let lowestConnId = 0;
  let osrmCache = new Map();
  let pprCache = new Map();

  function setConnections(conns, lowestId) {
    console.log('setConnections:', conns);
    connections = conns;
    lowestConnId = lowestId;
    connectionIdToIdx.clear();
    tripToConnections.clear();
    trainSegments.clear();
    walkSegments.clear();
    tripsComplete = false;
    requestTrips();
    requestWalks();
  }

  function requestTrips() {
    let trips = [];
    connections.forEach((connection, idx) => {
      connectionIdToIdx.set(connection.id, idx);
      connection.trains.forEach(train => {
        if (train.trip) {
          trips.push(train.trip);
          const tripKey = getTripKey(train.trip);
          let c = tripToConnections.get(tripKey);
          if (c === undefined) {
            c = [];
            tripToConnections.set(tripKey, c);
          }
          if (!c.includes(connection.id)) {
            c.push(connection.id);
          }
        }
      });
    });
    const allTripCount = trips.length;
    trips = trips.filter((v, i, a) => i === a.findIndex(e => deepEqual(v, e)));
    const uniqueTripCount = trips.length;
    const request = RailViz.API.makeTripsRequest(trips);

    console.log('sending connection trip request for', uniqueTripCount, '/', allTripCount, 'trips');

    if (uniqueTripCount === 0) {
      tripsComplete = true;
      showConnections();
      return;
    }

    RailViz.Main.setLoading(1);
    lastTripsRequest = RailViz.API.sendRequest(
      RailViz.Main.getApiEndPoint() + '?connectionTrips', request,
      (response, callId, duration) =>
        tripDataReceived(response, callId, duration),
      RailViz.Main.handleApiError);
  }

  function tripDataReceived(response, callId, duration) {
    const data = response.content;
    RailViz.Main.setLoading(-1);
    if (callId != lastTripsRequest) {
      return;
    }
    console.log('Received connection trip data after', duration, 'ms');

    for (let train of data.trains) {
      // preprocess train
      {
        train.route_index = train.route_index || 0;
        train.segment_index = train.segment_index || 0;
        train.clasz = train.clasz || 0;
        train.trip.forEach(trip => { trip.train_nr = trip.train_nr || 0 });
        const route = data.routes[train.route_index];
        const segment = route.segments[train.segment_index];
        train.departureStation = data.stations.find(s => s.id == segment.from_station_id);
        train.arrivalStation = data.stations.find(s => s.id == segment.to_station_id);
      }

      const segmentId = trainSegmentId(train.route_index, train.segment_index);
      const segment = data.routes[train.route_index].segments[train.segment_index];
      for (let trip of train.trip) {
        const conns = connectionsContainingSegment(train, trip, segment);
        if (conns.length > 0) {
          let seg = trainSegments.get(segmentId);
          if (seg === undefined) {
            seg = {
              segment: segment,
              trips: [],
              color: conns[0] - lowestConnId
            };
            trainSegments.set(segmentId, seg);
          }
          seg.trips.push({trip: trip, connectionIds: conns});
        }
      }
    }

    tripsComplete = true;
    showConnections();
  }

  function trainSegmentId(routeIndex, segmentIndex) {
    return routeIndex * 10000 + segmentIndex;
  }

  function connectionsContainingSegment(rvTrain, rvTrip, rvSegment) {
    const tripKey = getTripKey(rvTrip);
    let conns = tripToConnections.get(tripKey);
    if (conns) {
      return conns.filter(connId => {
        const connIdx = connectionIdToIdx.get(connId);
        const conn = connections[connIdx];
        return conn.trains.some(train =>
          deepEqual(train.trip, rvTrip) &&
          train.sections.some(sec =>
            sec.departureStation.id === rvSegment.from_station_id &&
            sec.arrivalStation.id === rvSegment.to_station_id));
      });
    } else {
      return [];
    }
  }

  function requestWalks() {
    connections.forEach(connection => {
      connection.walks.forEach(walk => {
        const walkKey = walkId(walk);
        let w = walkSegments.get(walkKey);
        if (w === undefined) {
          w = {
            walk: walk,
            connectionIds: [],
            polyline: null,
            error: false,
            color: connection.id - lowestConnId
          };
          walkSegments.set(walkKey, w);
        }
        w.connectionIds.push(connection.id);
        if (!w.polyline && walk.polyline) {
          w.polyline = walk.polyline;
        }
      });
    });

    walkSegments.forEach(requestWalkSegment);
    showConnections();
  }

  function requestWalkSegment(w) {
    if (w.polyline) {
      return;
    }
    const walkKey = walkId(w.walk);
    let request = null;
    if (w.walk.mumoType === 'bike' || w.walk.mumoType === 'car') {
      w.routerType = 'osrm';
      const cached = osrmCache.get(walkKey);
      if (cached) {
        w.polyline = cached.polyline;
        w.osrmRoute = cached.osrmRoute;
        walkDataComplete(w);
      } else {
        request = RailViz.API.makeOSRMRequest(w.walk);
      }
    } else {
      w.routerType = 'ppr';
      const cached = pprCache.get(walkKey);
      if (cached) {
        w.polyline = cached.polyline;
        w.pprRoute = cached.pprRoute;
        walkDataComplete(w);
      } else {
        request = RailViz.API.makePPRRequest(w.walk, RailViz.Main.getPPRSearchOptions());
      }
    }
    if (request && w.routerType) {
      console.log('request walkSegment', w);
      RailViz.Main.setLoading(1);
      RailViz.API.sendRequest(
        RailViz.Main.getApiEndPoint() + '?connectionWalk;' + w.routerType, request,
        (response, callId, duration) =>
          walkDataReceived(w, response, callId, duration),
        (response) => {
          w.error = true;
          showConnections();
          RailViz.Main.handleApiError(response)
        });
    }
  }

  function walkDataReceived(w, response, callId, duration) {
    const data = response.content;
    RailViz.Main.setLoading(-1);
    console.log('received walk data from', w.routerType, 'after', duration, 'ms for', w, data);

    switch (w.routerType) {
      case 'ppr':
        handlePPRResponse(w, data);
        break;
      case 'osrm':
        handleOSRMResponse(w, data);
        break;
    }

    showConnections();
  }

  function handlePPRResponse(w, data) {
    if (data.routes.length === 0 || data.routes[0].routes.length === 0) {
      console.log("ppr didn't find routes for:", w);
      w.polyline = [
          w.walk.departureStation.pos.lat,
          w.walk.departureStation.pos.lng,
          w.walk.arrivalStation.pos.lat,
          w.walk.arrivalStation.pos.lng
      ];
      return;
    }
    let routes = data.routes[0].routes;
    routes.forEach(r => {
      r.duration = r.duration || 0;
      r.acceleration = r.accessibility || 0;
    });
    routes.sort((a, b) => {
      const d = Math.abs(a.duration - w.walk.duration) - Math.abs(b.duration - w.walk.duration);
      if (d === 0) {
        return Math.abs(a.accessibility - w.walk.accessibility) - Math.abs(b.accessibility - w.walk.accessibility);
      } else {
        return d;
      }
    });
    const route = routes[0];
    w.polyline = route.path;
    w.pprRoute = route;
    pprCache.set(walkId(w.walk), {
      polyline: w.polyline,
      pprRoute: w.pprRoute
    });
    walkDataComplete(w);
  }

  function handleOSRMResponse(w, data) {
    w.polyline = data.polyline;
    w.osrmRoute = data;
    osrmCache.set(walkId(w.walk), {
      polyline: w.polyline,
      osrmRoute: w.osrmRoute
    });
    walkDataComplete(w);
  }

  function walkDataComplete(w) {
    // TODO: send to elm
  }

  function walkId(walk) {
    return JSON.stringify({
      mumoType: walk.mumoType,
      duration: walk.duration,
      accessibility: walk.accessibility,
      from: walk.departureStation.pos,
      to: walk.arrivalStation.pos
    });
  }

  function getTripKey(trip) {
    return JSON.stringify({
      station_id: trip.station_id,
      train_nr: trip.train_nr,
      time: trip.time,
      target_station_id: trip.target_station_id,
      target_time: trip.target_time,
      line_id: trip.line_id
    });
  }

  function dataIsComplete() {
    if (!tripsComplete) {
      return false;
    }
    for (let w of walkSegments.values()) {
      if (!w.polyline && !w.error) {
        return false;
      }
    }
    return true;
  }

  function showConnections() {
    if (dataIsComplete()) {
      RailViz.Render.setConnections(trainSegments, walkSegments, lowestConnId);
    } else {
    }
  }

  function resetPPRCache() {
    pprCache.clear();
  }

  return {
    setConnections: setConnections,
    resetPPRCache: resetPPRCache
  };
})();
