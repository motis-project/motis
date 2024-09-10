var RailViz = RailViz || {};

RailViz.Main = (function () {
  var apiEndpoint;
  var mapInfo;
  var elmPorts;
  var timeOffset = 0;
  var trainUpdateTimeoutId, tripsUpdateTimeoutId;
  var detailFilter = null;
  var fullData, detailTrips;
  var showingDetailData = false;
  var lastTrainsRequest = null;
  var lastTrainsCount = 0;
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
    pickedConnection: null,
    highlightedConnections: [],
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

  function setDetailFilter(filter) {
    detailFilter = filter;
    if (detailFilter == null) {
      detailTrips = null;
      showFullData();
    } else {
      sendTripsRequest();
    }
  }

  function setDetailWalks(walks) {
    if (!detailFilter) {
      return;
    }

    const isSameStation = (a, b) =>
      a.id == b.id && a.pos.lat == b.pos.lat && a.pos.lng == b.pos.lng;

    const isSameWalk = (a, b) =>
      isSameStation(a.departureStation, b.departureStation) &&
      isSameStation(a.arrivalStation, b.arrivalStation);

    let updated = false;
    walks.forEach((walk) => {
      RailViz.Model.prepareWalk(walk);
      const existingWalk = detailFilter.walks.find((w) => isSameWalk(w, walk));
      if (existingWalk) {
        existingWalk.polyline = walk.polyline;
        existingWalk.coordinates = walk.coordinates;
        updated = true;
      }
    });
    if (updated && showingDetailData) {
      showDetailData();
    }
  }

  function setConnections(opt) {
    const conns = opt.connections;
    const lowestId = opt.lowestId;
    RailViz.ConnectionManager.setConnections(conns, lowestId);
  }

  function highlightConnections(ids) {
    RailViz.Path.Connections.highlightConnections(ids);
  }

  function makeTrainsRequest() {
    var bounds = mapInfo.railVizBounds;
    return RailViz.API.makeTrainsRequest(
      Math.min(mapInfo.zoom, 18),
      {
        lat: bounds.north,
        lng: bounds.west,
      },
      {
        lat: bounds.south,
        lng: bounds.east,
      },
      timeOffset + Date.now() / 1000,
      timeOffset + Date.now() / 1000 + 120,
      2500,
      lastTrainsCount
    );
  }

  function makeTripsRequest() {
    return (
      detailFilter &&
      RailViz.API.makeTripsRequest(detailFilter.trains.map((t) => t.trip))
    );
  }

  function sendTrainsRequest() {
    if (trainUpdateTimeoutId) {
      clearTimeout(trainUpdateTimeoutId);
    }
    trainUpdateTimeoutId = setTimeout(debouncedSendTrainsRequest, 90000);
    setLoading(1);
    lastTrainsRequest = RailViz.API.sendRequest(
      apiEndpoint,
      makeTrainsRequest(),
      (response, callId, duration) =>
        dataReceived(response, false, callId, duration),
      handleApiError
    );
  }

  function sendTripsRequest() {
    if (tripsUpdateTimeoutId) {
      clearTimeout(tripsUpdateTimeoutId);
    }
    if (detailFilter) {
      tripsUpdateTimeoutId = setTimeout(sendTripsRequest, 90000);
      setLoading(1);
      lastTripsRequest = RailViz.API.sendRequest(
        apiEndpoint,
        makeTripsRequest(),
        (response, callId, duration) =>
          dataReceived(response, true, callId, duration),
        handleApiError
      );
    }
  }

  function handleApiError(response) {
    setLoading(-1);
    elmPorts.handleRailVizError.send(response);
  }

  function dataReceived(response, onlyFilteredTrips, callId, duration) {
    var data = response.content;
    const lastRequest = onlyFilteredTrips
      ? lastTripsRequest
      : lastTrainsRequest;
    setLoading(-1);
    if (callId != lastRequest) {
      return;
    }
    RailViz.Model.preprocess(data);
    if (onlyFilteredTrips) {
      detailTrips = data;
      if (detailFilter) {
        showDetailData();
      }
    } else {
      lastTrainsCount = (data.trains || []).length
      data.stations = null;
      fullData = data;
      if (!detailFilter) {
        showFullData();
      }
    }
    elmPorts.clearRailVizError.send(null);
  }

  function showFullData() {
    showingDetailData = false;
    //RailViz.Path.Base.setEnabled(trainsEnabled);
    RailViz.Path.Extra.setData(fullData);
    RailViz.Path.Extra.setEnabled(trainsEnabled);
    RailViz.Path.Detail.setEnabled(false);
    RailViz.Path.Connections.setEnabled(true);

    RailViz.Render.setData(fullData);
    RailViz.Render.setMinZoom(0);
    RailViz.Render.setTrainsEnabled(trainsEnabled);
  }

  function showDetailData() {
    showingDetailData = true;
    //RailViz.Path.Base.setEnabled(false);
    RailViz.Path.Extra.setEnabled(false);
    RailViz.Path.Detail.setData(detailFilter, detailTrips);
    RailViz.Path.Detail.setEnabled(true);
    RailViz.Path.Connections.setEnabled(false);

    RailViz.Render.setData(detailTrips);
    RailViz.Render.setMinZoom(FILTERED_MIN_ZOOM);
    RailViz.Render.setTrainsEnabled(true);

    RailViz.Main.highlightConnections([]);
  }

  function showTrains(b) {
    trainsEnabled = b;
    //RailViz.Path.Base.setEnabled(!showingDetailData && trainsEnabled);
    RailViz.Path.Extra.setEnabled(!showingDetailData && trainsEnabled);
    RailViz.Path.Detail.setEnabled(showingDetailData && trainsEnabled);
    RailViz.Render.setTrainsEnabled(trainsEnabled);
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
    const spinner = document.getElementById("railviz-loading-spinner");
    if (spinner) {
      spinner.className = visible ? "visible" : "";
    }
  }

  function handleMouseEvent(
    eventType,
    button,
    x,
    y,
    pickedTrain,
    pickedStation,
    pickedConnection
  ) {
    if (eventType == "click") {
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
      if (eventType != "mouseout") {
        setTooltip(x, y, pickedTrain, pickedStation, pickedConnection);
      } else {
        setTooltip(-1, -1, null, null, null);
      }
    }
  }

  function setTooltip(x, y, pickedTrain, pickedStation, pickedConnection) {
    if (
      hoverInfo.x == x &&
      hoverInfo.y == y &&
      hoverInfo.pickedTrain == pickedTrain &&
      hoverInfo.pickedStation == pickedStation &&
      hoverInfo.pickedConnection == pickedConnection
    ) {
      return;
    }

    hoverInfo.x = x;
    hoverInfo.y = y;
    hoverInfo.pickedTrain = pickedTrain;
    hoverInfo.pickedStation = pickedStation;
    hoverInfo.pickedConnection = pickedConnection;

    var rvTrain = null;
    if (pickedTrain) {
      rvTrain = {
        names: pickedTrain.names,
        departureTime: pickedTrain.d_time * 1000,
        arrivalTime: pickedTrain.a_time * 1000,
        scheduledDepartureTime: pickedTrain.sched_d_time * 1000,
        scheduledArrivalTime: pickedTrain.sched_a_time * 1000,
        hasDepartureDelayInfo: !!(
          pickedTrain.d_time_reason && pickedTrain.d_time_reason != "SCHEDULE"
        ),
        hasArrivalDelayInfo: !!(
          pickedTrain.a_time_reason && pickedTrain.a_time_reason != "SCHEDULE"
        ),
        departureStation: pickedTrain.dStation.name,
        arrivalStation: pickedTrain.aStation.name,
      };
    }
    var rvStation = pickedStation && pickedStation.name;

    let hoveredTripSegments = null;
    let hoveredWalkSegment = null;

    let highlightedConnections = [];

    if (pickedConnection) {
      if (Array.isArray(pickedConnection)) {
        hoveredTripSegments = pickedConnection.map((pc) => {
          return {
            connectionIds: pc.connectionIds,
            trip: pc.trip,
            d_station_id: pc.d_station_id,
            a_station_id: pc.a_station_id,
          };
        });

        let idSet = new Set();
        pickedConnection.forEach((pc) =>
          pc.connectionIds.forEach((id) => idSet.add(id))
        );
        highlightedConnections.push(...idSet);
      } else {
        hoveredWalkSegment = pickedConnection;
        highlightedConnections = pickedConnection.connectionIds;
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
      hoveredTripSegments: hoveredTripSegments,
      hoveredWalkSegment: hoveredWalkSegment,
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
    };
  }

  return {
    init: init,
    debounce: debounce,
    mapUpdate: mapUpdate,
    setTimeOffset: setTimeOffset,
    getTimeOffset: function () {
      return timeOffset;
    },
    setDetailFilter: setDetailFilter,
    setDetailWalks: setDetailWalks,
    setConnections: setConnections,
    highlightConnections: highlightConnections,
    setUseFpsLimiter: setUseFpsLimiter,
    showTrains: showTrains,
    setPPRSearchOptions: setPPRSearchOptions,
    getPPRSearchOptions: getPPRSearchOptions,
    setLoading: setLoading,
    handleApiError: handleApiError,
    getApiEndPoint: function () {
      return apiEndpoint;
    },
  };
})();
