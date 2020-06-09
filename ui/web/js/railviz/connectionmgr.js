var RailViz = RailViz || {};

RailViz.ConnectionManager = (function () {
  let connections = [];
  let tripData = null;
  let walkData = [];

  let lastTripsRequest = null;
  let tripsComplete = false;
  let osrmCache = new Map();
  let pprCache = new Map();

  function setConnections(conns, _) {
    console.log("setConnections:", conns);
    connections = conns;
    tripData = null;
    walkData = [];
    tripsComplete = false;
    requestTrips();
    requestWalks();
  }

  function requestTrips() {
    let trips = [];
    connections.forEach((c) => {
      c.trains.forEach((t) => {
        trips.push(t.trip);
      });
    });

    if (trips.length === 0) {
      tripsComplete = true;
      showConnections();
      return;
    }

    trips = trips.filter(
      (v, i, a) => i === a.findIndex((e) => deepEqual(v, e))
    );

    RailViz.Main.setLoading(1);
    lastTripsRequest = RailViz.API.sendRequest(
      RailViz.Main.getApiEndPoint() + "?connectionTrips",
      RailViz.API.makeTripsRequest(trips),
      (resp, id, dur) => tripDataReceived(resp, id, dur),
      RailViz.Main.handleApiError
    );
  }

  function tripDataReceived(response, callId, duration) {
    tripData = response.content;
    RailViz.Main.setLoading(-1);
    if (callId != lastTripsRequest) {
      return;
    }
    console.log("Received connection trip data after", duration, "ms");

    RailViz.Model.preprocess(tripData);

    tripData.trains.forEach((t) => {
      t.connections = trainToConnectionIds(t);
      t.connections.forEach((c) => {
        c.resolved = c.resolved || [];
        c.resolved.push(t);
      });

      t.connectionIds = t.connections.map((c) => c.id);

      if (t.connections.length > 0) {
        const minCId = t.connectionIds.reduce(cIdCombine);
        t.polylines.forEach((tp, i) => {
          tp.polyline.trains = tp.polyline.trains || [];
          tp.polyline.trains.push(t);
          tp.polyline.minCId = cIdCombine(tp.polyline.minCId, minCId);
        });
      }
    });

    tripData.trains = tripData.trains.filter((t) => t.connections.length > 0);

    tripsComplete = true;
    showConnections();
  }

  function trainToConnectionIds(t) {
    return connections.reduce((acc, c) => {
      if (
        c.trains.some(
          (ct) =>
            // any train trip id matches connection trip id
            t.trip.some((tt) => deepEqual(tt, ct.trip)) &&
            // train section matches any connection section
            ct.sections.some(
              (cs) =>
                cs.departureStation.id == t.d_station_id &&
                cs.arrivalStation.id == t.a_station_id
            )
        )
      ) {
        acc.push(c);
      }
      return acc;
    }, []);
  }

  function requestWalks() {
    let walkMap = new Map();
    walkData = [];
    connections.forEach((connection) => {
      connection.walks.forEach((walk) => {
        const walkKey = walkId(walk);
        let w = walkMap.get(walkKey);
        if (w === undefined) {
          w = {
            id: walkData.length,
            walk: walk,
            connectionIds: [],
            polyline: null,
            error: false,
          };
          walkMap.set(walkKey, w);
          walkData.push(w);
        }
        walk.id = w.id;
        w.connectionIds.push(connection.id);
        if (!w.polyline && walk.polyline) {
          w.polyline = walk.polyline;
        }
      });
    });

    walkData.forEach((w) => {
      w.minCId = w.connectionIds.reduce(cIdCombine);
      if (!w.polyline) {
        requestWalkSegment(w);
      }
    });

    showConnections();
  }

  function requestWalkSegment(w) {
    const walkKey = walkId(w.walk);
    let request = null;
    if (w.walk.mumoType === "bike" || w.walk.mumoType === "car") {
      w.routerType = "osrm";
      const cached = osrmCache.get(walkKey);
      if (cached) {
        w.polyline = cached.polyline;
        w.osrmRoute = cached.osrmRoute;
        walkDataComplete(w);
      } else {
        request = RailViz.API.makeOSRMRequest(w.walk);
      }
    } else {
      w.routerType = "ppr";
      const cached = pprCache.get(walkKey);
      if (cached) {
        w.polyline = cached.polyline;
        w.pprRoute = cached.pprRoute;
        walkDataComplete(w);
      } else {
        request = RailViz.API.makePPRRequest(
          w.walk,
          RailViz.Main.getPPRSearchOptions()
        );
      }
    }
    if (request && w.routerType) {
      console.log("request walkSegment", w);
      RailViz.Main.setLoading(1);
      RailViz.API.sendRequest(
        RailViz.Main.getApiEndPoint() + "?connectionWalk;" + w.routerType,
        request,
        (response, callId, duration) =>
          walkDataReceived(w, response, callId, duration),
        (response) => {
          w.error = true;
          showConnections();
          RailViz.Main.handleApiError(response);
        }
      );
    }
  }

  function walkDataReceived(w, response, callId, duration) {
    const data = response.content;
    RailViz.Main.setLoading(-1);
    console.log(
      `received walk data from ${w.routerType} after ${duration} ms`,
      w,
      data
    );

    switch (w.routerType) {
      case "ppr":
        handlePPRResponse(w, data);
        break;
      case "osrm":
        handleOSRMResponse(w, data);
        break;
    }

    showConnections();
  }

  function handlePPRResponse(w, data) {
    if (data.routes.length === 0 || data.routes[0].routes.length === 0) {
      console.log("ppr didn't find routes for:", w);
      w.polyline = {
        coordinates: [
          w.walk.departureStation.pos.lat,
          w.walk.departureStation.pos.lng,
          w.walk.arrivalStation.pos.lat,
          w.walk.arrivalStation.pos.lng,
        ],
      };
      return;
    }
    let routes = data.routes[0].routes;
    routes.forEach((r) => {
      r.duration = r.duration || 0;
      r.acceleration = r.accessibility || 0;
    });
    routes.sort((a, b) => {
      const d =
        Math.abs(a.duration - w.walk.duration) -
        Math.abs(b.duration - w.walk.duration);
      if (d === 0) {
        return (
          Math.abs(a.accessibility - w.walk.accessibility) -
          Math.abs(b.accessibility - w.walk.accessibility)
        );
      } else {
        return d;
      }
    });
    const route = routes[0];
    w.polyline = route.path;
    w.pprRoute = route;
    pprCache.set(walkId(w.walk), {
      polyline: w.polyline,
      pprRoute: w.pprRoute,
    });
    walkDataComplete(w);
  }

  function handleOSRMResponse(w, data) {
    w.polyline = data.polyline;
    w.osrmRoute = data;
    osrmCache.set(walkId(w.walk), {
      polyline: w.polyline,
      osrmRoute: w.osrmRoute,
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
      to: walk.arrivalStation.pos,
    });
  }

  function cIdCombine(a, b) {
    if (a < 0 && b < 0) {
      return Math.max(a, b);
    } else if (a === undefined || a < 0) {
      return b;
    } else if (b === undefined || b < 0) {
      return a;
    } else {
      return Math.min(a, b);
    }
  }

  function dataIsComplete() {
    if (!tripsComplete) {
      return false;
    }
    for (let w of walkData.values()) {
      if (!w.polyline && !w.error) {
        return false;
      }
    }
    return true;
  }

  function showConnections() {
    if (dataIsComplete()) {
      RailViz.Path.Connections.setData(connections, tripData, walkData);
    }
  }

  function resetPPRCache() {
    pprCache.clear();
  }

  return {
    setConnections: setConnections,
    resetPPRCache: resetPPRCache,
    cIdCombine: cIdCombine,
  };
})();
