var RailViz = RailViz || {};

RailViz.Render = (function () {

  var data;
  var timeOffset = 0;

  var map = null;
  var gl = null;

  var rafRequest;
  var offscreen = {};
  var mouseHandler;
  var minZoom = 0;
  let pixelRatio = window.devicePixelRatio;
  var lastFrame;
  var targetFrameTime = null;
  var forceDraw = true;

  var routesEnabled = false;
  var trainsEnabled = true;

  function init(mouseEventHandler) {
    setData(null);
    setConnections(null, null);

    mouseHandler = mouseEventHandler || (() => {});
  }

  function setData(newData) {
    data = newData || {
      stations: [],
      routes: [],
      trains: [],
      routeVertexCount: 0,
      footpaths: []
    };

    RailViz.Routes.init(data.routes, data.routeVertexCount, data.footpaths);
    RailViz.Trains.init(data.trains, data.routes);
    forceDraw = true;
  }

  function setConnections(trainSegments, walkSegments, lowestConnId) {
    RailViz.Connections.setData(trainSegments, walkSegments, lowestConnId);
  }

  function highlightConnections(ids) {
    RailViz.Connections.highlightConnections(ids);
  }

  function colorRouteSegments() {
    const categoryColors = RailViz.Trains.categoryColors;
    data.trains.forEach(
      train => RailViz.Routes.colorSegment(
        train.route_index, train.segment_index,
        categoryColors[train.clasz]));
    forceDraw = true;
  }

  function setConnectionFilter(filter) {
    if (!filter) {
      return;
    }
    if (filter.walks && filter.walks.length > 0) {
      data.footpaths = filter.walks;
      data.footpaths.forEach(adjustFootpathCoords);
      RailViz.Routes.init(data.routes, data.routeVertexCount, data.footpaths);
      let additionalStations = false;
      data.footpaths.forEach(footpath => {
        const addedFrom = addAdditionalStation(footpath.departureStation);
        const addedTo = addAdditionalStation(footpath.arrivalStation);
        if (addedFrom || addedTo) {
          additionalStations = true;
        }
      });
      // if (additionalStations) {
      //   RailViz.Stations.init(data.stations);
      // }
    }
    filter.trains.forEach(
      train => train.sections.forEach(
        section => highlightSection(train, section)));

    // XXX
    // filter.interchangeStations.forEach(
    //   RailViz.Stations.highlightInterchangeStation);
    // filter.intermediateStations.forEach(
    //   RailViz.Stations.highlightIntermediateStation);
    forceDraw = true;
  }

  function setTrainsEnabled(b) {
    forceDraw = b != trainsEnabled;
    trainsEnabled = b;
  }

  function setRoutesEnabled(b) {
    forceDraw = b != routesEnabled;
    routesEnabled = b;
  }

  function setConnectionsEnabled(b) {
    RailViz.Connections.setEnabled(b);
  }

  function addAdditionalStation(station) {
    const existingStation = data.stations.some(s => s.id == station.id);
    if (!existingStation) {
      data.stations.push(station);
      forceDraw = true;
      return true;
    }
    return false;
  }

  function adjustFootpathCoords(footpath) {
    const replace = !footpath.polyline;
    const from_station_id = footpath.from_station_id;
    const to_station_id = footpath.to_station_id;
    const startSegments = data.routes.reduce(
      (acc, r) => acc.concat(
        r.segments.filter(seg => seg.to_station_id == from_station_id)),
      []);
    const coords = footpath.coordinates.coordinates;
    if (startSegments.length == 1) {
      const fromCoords = startSegments[0].coordinates.coordinates;
      const x = fromCoords[fromCoords.length - 2];
      const y = fromCoords[fromCoords.length - 1];
      if (replace) {
        coords[0] = x;
        coords[1] = y;
      } else {
        if (coords[0] != x || coords[1] != y) {
          coords.unshift(x, y);
        }
      }
    }
    const endSegments = data.routes.reduce(
      (acc, r) => acc.concat(
        r.segments.filter(seg => seg.from_station_id == to_station_id)),
      []);
    if (endSegments.length == 1) {
      const toCoords = endSegments[0].coordinates.coordinates;
      const x = toCoords[0];
      const y = toCoords[1];
      if (replace) {
        coords[2] = x;
        coords[3] = y;
      } else {
        if (coords[coords.length - 2] != x || coords[coords.length - 1] != y) {
          coords.push(x, y);
        }
      }
    }
    forceDraw = true;
  }

  function highlightSection(train, section) {
    const matchingTrains = data.trains.filter(
      t => t.sched_d_time == section.scheduledDepartureTime / 1000 &&
        t.sched_a_time == section.scheduledArrivalTime / 1000 &&
        data.routes[t.route_index]
          .segments[t.segment_index]
          .from_station_id == section.departureStation.id &&
        data.routes[t.route_index]
          .segments[t.segment_index]
          .to_station_id == section.arrivalStation.id);
    matchingTrains.forEach(
      t => RailViz.Routes.highlightSegment(t.route_index, t.segment_index));
    forceDraw = true;
  }

  function setTimeOffset(newTimeOffset) {
    timeOffset = newTimeOffset;
    forceDraw = true;
  }

  function setMinZoom(newMinZoom) {
    minZoom = newMinZoom;
    forceDraw = true;
  }

  function setTargetFps(targetFps) {
    if (targetFps) {
      targetFrameTime = 1000 / targetFps;
    } else {
      targetFrameTime = null;
    }
  }

  function setup(_map, _gl) {
    map = _map;
    gl = _gl;

    offscreen = {};

    gl.getExtension("OES_element_index_uint");

    RailViz.Routes.setup(gl);
    RailViz.Trains.setup(gl);

    lastFrame = null;
    forceDraw = true;
    rafRequest = requestAnimationFrame(maybe_render);
  }

  function stop() {
    cancelAnimationFrame(rafRequest);
  }

  function maybe_render(timestamp) {
    if (targetFrameTime != null && lastFrame != null && !forceDraw) {
      const frameTime = performance.now() - lastFrame;
      if (frameTime < targetFrameTime) {
        rafRequest = requestAnimationFrame(maybe_render);
        return;
      }
    }
    forceDraw = false;

    if(map != null) {
      map.triggerRepaint();
    }
  }

  function prerender(gl, matrix) {
    var time = timeOffset + (Date.now() / 1000);
    if (trainsEnabled) {
      RailViz.Trains.preRender(gl, time);
    }
  }

  function render(gl, matrix, zoom) {
    createOffscreenBuffer();

    zoom = Math.max(minZoom, zoom);

    for (var i = 0; i <= 1; i++) {
      var isOffscreen = i == 0;

      gl.bindFramebuffer(
        gl.FRAMEBUFFER, isOffscreen ? offscreen.framebuffer : null);
      gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
      gl.enable(gl.BLEND);
      gl.disable(gl.DEPTH_TEST);
      // gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.blendFuncSeparate(
        gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

      if(isOffscreen) {
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT);
      }

      if (routesEnabled) {
        RailViz.Routes.render(gl, matrix, zoom, pixelRatio, isOffscreen);
      }
      if (trainsEnabled) {
        RailViz.Trains.render(gl, matrix, zoom, pixelRatio, isOffscreen);
      }
    }

    lastFrame = performance.now();
    rafRequest = requestAnimationFrame(maybe_render);
  }

  function createOffscreenBuffer() {
    var width = gl.drawingBufferWidth;
    var height = gl.drawingBufferHeight;

    if (offscreen.width === width && offscreen.height === height &&
      offscreen.framebuffer) {
      return;
    }

    offscreen.width = width;
    offscreen.height = height;

    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (offscreen.framebuffer && gl.isFramebuffer(offscreen.framebuffer)) {
      gl.deleteFramebuffer(offscreen.framebuffer);
      offscreen.framebuffer = null;
    }
    if (offscreen.texture && gl.isTexture(offscreen.texture)) {
      gl.deleteTexture(offscreen.texture);
      offscreen.texture = null;
    }

    offscreen.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, offscreen.texture);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    offscreen.framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, offscreen.framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, offscreen.texture,
      0);

    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  function readOffscreenPixel(x, y) {
    if (!offscreen.framebuffer || !gl.isFramebuffer(offscreen.framebuffer)) {
      return null;
    }

    if (x < 0 || y < 0 || x >= gl.drawingBufferWidth ||
      y >= gl.drawingBufferHeight) {
      return null;
    }

    var pixels = new Uint8Array(4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, offscreen.framebuffer);
    gl.readPixels(
      x, gl.drawingBufferHeight - y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    return pixels;
  }

  function handleMouseEvent(eventType, event) {
    if(!map) {
      return;
    }
    const canvas = map.getCanvas();

    const mouseX = (event.pageX - canvas.offsetLeft);
    const mouseY = (event.pageY - canvas.offsetTop);
    const pickingX = mouseX * pixelRatio;
    const pickingY = mouseY * pixelRatio;
    const button = event.button;

    const offscreenPixel = readOffscreenPixel(pickingX, pickingY);
    const pickId = RailViz.Picking.colorToPickId(offscreenPixel);

    const pickedTrainIndex = RailViz.Trains.getPickedTrainIndex(pickId);
    const pickedTrain =
      pickedTrainIndex != null ? data.trains[pickedTrainIndex] : null;

    const features = map.queryRenderedFeatures([mouseX, mouseY]);

    const station = features.find(e => e.sourceLayer == 'station');
    const pickedStation = station !== undefined ? {id: station.id, name: station.properties.name } : null;

    const pickedConnectionSegment = RailViz.Connections.getPickedSegment(features);

    // if (pickId && eventType != 'mouseout') {
    //   canvas.style.cursor = 'pointer';
    // } else {
    //   canvas.style.cursor = 'default';
    // }

    mouseHandler(eventType, button, mouseX, mouseY, pickedTrain, pickedStation, pickedConnectionSegment);
  }

  return {
    init: init,
    setup: setup,
    stop: stop,
    setData: setData,
    setConnections: setConnections,
    highlightConnections: highlightConnections,
    setTimeOffset: setTimeOffset,
    prerender: prerender,
    render: render,
    colorRouteSegments: colorRouteSegments,
    setMinZoom: setMinZoom,
    setConnectionFilter: setConnectionFilter,
    setTargetFps: setTargetFps,
    setRoutesEnabled: setRoutesEnabled,
    setTrainsEnabled: setTrainsEnabled,
    setConnectionsEnabled: setConnectionsEnabled,
    handleMouseEvent: handleMouseEvent
  };

})();
