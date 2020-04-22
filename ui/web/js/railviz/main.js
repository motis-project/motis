var RailViz = RailViz || {};

RailViz.Main = (function () {

  var apiEndpoint;
  var mapInfo;
  var elmPorts;
  var timeOffset = 0;
  var trainUpdateTimeoutId, tripsUpdateTimeoutId;
  var filteredTripIds;
  var connectionFilter = null;
  var fullData, filteredData;
  var showingFilteredData = false;
  var dragEndTime = null;
  var lastTrainsRequest = null;
  var lastTripsRequest = null;
  var useFpsLimiter = true;
  var trainsEnabled = true;
  var pprSearchOptions = {};
  var loadingCount = 0;

  var hoverInfo = {
    x: -1,
    y: -1,
    pickedTrain: null,
    pickedStation: null,
    pickedConnectionSegment: null,
    highlightedConnections: []
  };

  var debouncedSendTrainsRequest = debounce(sendTrainsRequest, 200);

  const FILTERED_MIN_ZOOM = 14;

  function init(endpoint, ports) {
    apiEndpoint = endpoint;
    elmPorts = ports;
    RailViz.Render.init(handleMouseEvent);
  }

  function mapUpdate(newMapInfo) {
    mapInfo = newMapInfo;
    setupFpsLimiter();
    debouncedSendTrainsRequest();
  }

  function setTimeOffset(newTimeOffset) {
    timeOffset = newTimeOffset / 1000;
    RailViz.Render.setTimeOffset(timeOffset);
    debouncedSendTrainsRequest();
  }

  function setTripFilter(tripIds) {
    filteredTripIds = tripIds;
    if (filteredTripIds) {
      sendTripsRequest();
    } else {
      filteredData = null;
      showFullData();
    }
  }

  function setConnectionFilter(filter) {
    filter.walks.forEach(RailViz.Preprocessing.prepareWalk);
    connectionFilter = filter;
    if (showingFilteredData) {
      RailViz.Render.setConnectionFilter(connectionFilter);
    }
  }

  function setConnections(opt) {
    const conns = opt.connections;
    const lowestId = opt.lowestId;
    RailViz.ConnectionManager.setConnections(conns, lowestId);
  }

  function highlightConnections(ids) {
    RailViz.Render.highlightConnections(ids);
  }

  function showTrains(b) {
    trainsEnabled = b;
    RailViz.Render.setTrainsEnabled(trainsEnabled);
  }

  function updateWalks(walks) {
    if (!connectionFilter) {
      return;
    }
    let updated = false;
    walks.forEach(walk => {
      RailViz.Preprocessing.prepareWalk(walk);
      const existingWalk =
        connectionFilter.walks.find(w => isSameWalk(w, walk));
      if (existingWalk) {
        existingWalk.polyline = walk.polyline;
        existingWalk.coordinates = walk.coordinates;
        updated = true;
      }
    });
    if (showingFilteredData) {
      showFilteredData();
    }
  }

  function isSameWalk(a, b) {
    return isSameStation(a.departureStation, b.departureStation) &&
      isSameStation(a.arrivalStation, b.arrivalStation);
  }

  function isSameStation(a, b) {
    return a.id == b.id && a.pos.lat == b.pos.lat && a.pos.lng == b.pos.lng;
  }

  function setUseFpsLimiter(enabled) {
    useFpsLimiter = enabled;
    setupFpsLimiter();
  }

  function setupFpsLimiter() {
    let targetFps = null;
    if (useFpsLimiter && mapInfo) {
      const zoom = mapInfo.zoom;
      if (zoom <= 10) {
        targetFps = 2;
      } else if (zoom <= 12) {
        targetFps = 5;
      } else if (zoom <= 14) {
        targetFps = 10;
      } else if (zoom <= 15) {
        targetFps = 30;
      }
    }
    RailViz.Render.setTargetFps(targetFps);
  }

  function setLoading(change) {
    const visibleBefore = loadingCount > 0;
    loadingCount = loadingCount + change;
    const visibleAfter = loadingCount > 0;
    if (visibleBefore !== visibleAfter) {
      toggleLoadingSpinner(visibleAfter);
    }
  }

  function toggleLoadingSpinner(visible) {
    const spinner = document.getElementById('railviz-loading-spinner');
    if (spinner) {
      spinner.className = visible ? 'visible' : '';
    }
  }

  function makeTrainsRequest() {
    var bounds = mapInfo.railVizBounds;
    return RailViz.API.makeTrainsRequest(
      Math.min(mapInfo.zoom + 2, 18), {
        lat: bounds.north,
        lng: bounds.west
      }, {
        lat: bounds.south,
        lng: bounds.east
      }, timeOffset + (Date.now() / 1000),
      timeOffset + (Date.now() / 1000) + 120, 1000);
  }

  function makeTripsRequest() {
    return filteredTripIds && RailViz.API.makeTripsRequest(filteredTripIds);
  }

  function sendTrainsRequest() {
    if (trainUpdateTimeoutId) {
      clearTimeout(trainUpdateTimeoutId);
    }
    trainUpdateTimeoutId = setTimeout(debouncedSendTrainsRequest, 90000);
    setLoading(1);
    lastTrainsRequest = RailViz.API.sendRequest(
      apiEndpoint, makeTrainsRequest(),
      (response, callId, duration) =>
        dataReceived(response, false, callId, duration),
      handleApiError);
  }

  function sendTripsRequest() {
    if (tripsUpdateTimeoutId) {
      clearTimeout(tripsUpdateTimeoutId);
    }
    if (filteredTripIds) {
      tripsUpdateTimeoutId = setTimeout(sendTripsRequest, 90000);
      setLoading(1);
      lastTripsRequest = RailViz.API.sendRequest(
        apiEndpoint, makeTripsRequest(),
        (response, callId, duration) =>
          dataReceived(response, true, callId, duration),
        handleApiError);
    }
  }

  function handleApiError(response) {
    setLoading(-1);
    elmPorts.handleRailVizError.send(response);
  }

  function dataReceived(response, onlyFilteredTrips, callId, duration) {
    var data = response.content;
    const lastRequest =
      onlyFilteredTrips ? lastTripsRequest : lastTrainsRequest;
    setLoading(-1);
    if (callId != lastRequest) {
      return;
    }
    RailViz.Preprocessing.preprocess(data);
    if (onlyFilteredTrips) {
      filteredData = data;
      if (filteredTripIds) {
        showFilteredData();
      }
    } else {
      fullData = data;
      if (!filteredTripIds) {
        showFullData();
      }
    }
    elmPorts.clearRailVizError.send(null);
  }

  function showFullData() {
    showingFilteredData = false;
    RailViz.Path.Extra.setData(fullData);
    RailViz.Path.Trips.setEnabled(false);
    RailViz.Render.setData(fullData);
    RailViz.Render.setMinZoom(0);
    RailViz.Render.setConnectionsEnabled(true);
    RailViz.Render.setTrainsEnabled(trainsEnabled);
    RailViz.Render.setRoutesEnabled(false);
  }

  function showFilteredData() {
    showingFilteredData = true;
    RailViz.Path.Trips.setData(filteredData, filteredTripIds);
    RailViz.Path.Trips.setEnabled(true);
    RailViz.Render.setData(filteredData);
    RailViz.Render.setMinZoom(FILTERED_MIN_ZOOM);
    RailViz.Render.setConnectionFilter(connectionFilter);
    RailViz.Render.setConnectionsEnabled(false);
    RailViz.Render.setTrainsEnabled(true);
    RailViz.Render.setRoutesEnabled(true);
    RailViz.Main.highlightConnections([]);
  }

  function setPPRSearchOptions(options) {
    if (!deepEqual(options, pprSearchOptions)) {
      RailViz.ConnectionManager.resetPPRCache();
    }
    pprSearchOptions = options;
  }

  function getPPRSearchOptions() {
    return pprSearchOptions;
  }

  function handleMouseEvent(eventType, button, x, y, pickedTrain, pickedStation, pickedConnectionSegment) {
    if (eventType == 'mouseup') {
      if (dragEndTime != null && (Date.now() - dragEndTime < 100)) {
        // ignore mouse up events immediately after drag end
        return;
      }
      if (pickedTrain) {
        if (pickedTrain.trip && pickedTrain.trip.length > 0) {
          elmPorts.showTripDetails.send(pickedTrain.trip[0]);
        }
      } else if (pickedStation) {
        elmPorts.showStationDetails.send("" + pickedStation.id);
      }
      if (button == 0) {
        elmPorts.mapCloseContextMenu.send(null);
      }
    } else {
      if (eventType != 'mouseout') {
        setTooltip(x, y, pickedTrain, pickedStation, pickedConnectionSegment);
      } else {
        setTooltip(-1, -1, null, null, null);
      }
    }
  }

  function dragEnd() {
    dragEndTime = Date.now();
  }

  function setTooltip(x, y, pickedTrain, pickedStation, pickedConnectionSegment) {
    if (hoverInfo.x == x && hoverInfo.y == y &&
      hoverInfo.pickedTrain == pickedTrain &&
      hoverInfo.pickedStation == pickedStation &&
      hoverInfo.pickedConnectionSegment == pickedConnectionSegment) {
      return;
    }

    hoverInfo.x = x;
    hoverInfo.y = y;
    hoverInfo.pickedTrain = pickedTrain;
    hoverInfo.pickedStation = pickedStation;
    hoverInfo.pickedConnectionSegment = pickedConnectionSegment;

    var rvTrain = null;
    if (pickedTrain) {
      rvTrain = {
        names: pickedTrain.names,
        departureTime: pickedTrain.d_time * 1000,
        arrivalTime: pickedTrain.a_time * 1000,
        scheduledDepartureTime: pickedTrain.sched_d_time * 1000,
        scheduledArrivalTime: pickedTrain.sched_a_time * 1000,
        hasDepartureDelayInfo:
          !!(pickedTrain.d_time_reason &&
            pickedTrain.d_time_reason != 'SCHEDULE'),
        hasArrivalDelayInfo:
          !!(pickedTrain.a_time_reason &&
            pickedTrain.a_time_reason != 'SCHEDULE'),
        departureStation: pickedTrain.departureStation.name,
        arrivalStation: pickedTrain.arrivalStation.name
      };
    }
    var rvStation = pickedStation && pickedStation.name;

    let connectionSegment = null;
    let walkSegment = null;

    let highlightedConnections = [];
    if (pickedConnectionSegment) {
      console.log('pickedConnectionSegment:', pickedConnectionSegment);
      if (pickedConnectionSegment.walk) {
        walkSegment = pickedConnectionSegment;
        highlightedConnections = walkSegment.connectionIds;
      } else {
        connectionSegment = pickedConnectionSegment;
        highlightedConnections = Array.from(new Set([].concat.apply([], connectionSegment.trips.map(trip => trip.connectionIds))));
      }
    }
    if (!deepEqual(highlightedConnections, hoverInfo.highlightedConnections)) {
      hoverInfo.highlightedConnections = highlightedConnections;
      RailViz.Main.highlightConnections(highlightedConnections);
    }

    elmPorts.mapSetTooltip.send({
      mouseX: x,
      mouseY: y,
      hoveredTrain: rvTrain,
      hoveredStation: rvStation,
      hoveredConnectionSegment: connectionSegment,
      hoveredWalkSegment: walkSegment
    });
  }

  function debounce(f, t) {
    var timeout;
    return function () {
      var context = this;
      var args = arguments;
      var later = function () {
        timeout = null;
        f.apply(context, args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, t);
    }
  }

  return {
    init: init,
    debounce: debounce,
    mapUpdate: mapUpdate,
    setTimeOffset: setTimeOffset,
    getTimeOffset: function () {
      return timeOffset;
    },
    setTripFilter: setTripFilter,
    setConnectionFilter: setConnectionFilter,
    setConnections: setConnections,
    highlightConnections: highlightConnections,
    updateWalks: updateWalks,
    dragEnd: dragEnd,
    setUseFpsLimiter: setUseFpsLimiter,
    showTrains: showTrains,
    setPPRSearchOptions: setPPRSearchOptions,
    getPPRSearchOptions: getPPRSearchOptions,
    setLoading: setLoading,
    handleApiError: handleApiError,
    getApiEndPoint: function () {
      return apiEndpoint;
    }
  };

})();
