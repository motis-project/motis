var RailViz = RailViz || {};

RailViz.API = (function () {

  var lastCallId = 0;

  function makeTrainsRequest(zoom_level, corner1, corner2, startTime, endTime, maxTrains, lastTrains) {
    return {
      destination: {
        type: 'Module',
        target: '/railviz/get_trains'
      },
      content_type: 'RailVizTrainsRequest',
      content: {
        zoom_bounds: zoom_level,
        zoom_geo: zoom_level + 2,
        corner1: corner1,
        corner2: corner2,
        start_time: Math.floor(startTime),
        end_time: Math.ceil(endTime),
        max_trains: maxTrains,
        last_trains: lastTrains
      }
    };
  }

  function makeTripsRequest(tripIds) {
    return {
      destination: {
        type: 'Module',
        target: '/railviz/get_trips'
      },
      content_type: 'RailVizTripsRequest',
      content: {
        trips: tripIds
      }
    };
  }

  function makePPRRequest(walk, searchOptions) {
    return {
      destination: {
        type: 'Module',
        target: '/ppr/route'
      },
      content_type: 'FootRoutingRequest',
      content: {
        start: walk.departureStation.pos,
        destinations: [walk.arrivalStation.pos],
        search_options: searchOptions,
        include_steps: true,
        include_edges: false,
        include_path: true
      }
    };
  }

  function makeOSRMRequest(walk) {
    return {
      destination: {
        type: 'Module',
        target: '/osrm/via'
      },
      content_type: 'OSRMViaRouteRequest',
      content: {
        profile: getOSRMProfile(walk),
        waypoints: [
          walk.departureStation.pos,
          walk.arrivalStation.pos
        ]
      }
    };
  }

  function getOSRMProfile(walk) {
    switch (walk.mumoType) {
      case 'bike':
        return 'bike';
      case 'car':
        return 'car';
      default:
        return 'foot';
    }
  }

  function sendRequest(apiEndpoint, requestData, onSuccess, onFail) {
    var xhr = new XMLHttpRequest();
    const callId = ++lastCallId;
    const startTime = performance.now();
    xhr.addEventListener('load', function () {
      var response = xhr.responseText;
      try {
        response = JSON.parse(xhr.responseText);
      } catch (ex) {
        console.log('Could not parse JSON response:', ex);
      }
      if (xhr.status == 200) {
        onSuccess(response, callId, performance.now() - startTime);
      } else {
        onFail(response, callId, performance.now() - startTime);
      }
    });
    xhr.addEventListener('error', function () {
      onFail('NetworkError', callId, performance.now() - startTime);
    });
    xhr.addEventListener('timeout', function () {
      onFail('TimeoutError', callId, performance.now() - startTime);
    });
    var url = apiEndpoint;
    if (requestData.destination && requestData.destination.target) {
      if (!url.includes('?')) {
        url += '?target=';
      } else {
        if (!url.endsWith('&')) {
          url += '&';
        }
        url += 'target=';
      }
      url += requestData.destination.target;
    }
    xhr.open('POST', url);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(JSON.stringify(requestData));
    return callId;
  }

  return {
    makeTrainsRequest: makeTrainsRequest,
    makeTripsRequest: makeTripsRequest,
    makePPRRequest: makePPRRequest,
    makeOSRMRequest: makeOSRMRequest,
    sendRequest: sendRequest,
  };
})();
