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

  var trainsEnabled = true;

  function init(mouseEventHandler) {
    setData(null);

    mouseHandler = mouseEventHandler || (() => {});
  }

  function setData(newData) {
    data = newData || {
      stations: [],
      routes: [],
      trains: [],
      routeVertexCount: 0,
      footpaths: [],
    };

    RailViz.Trains.setData(data.trains);
    forceDraw = true;
  }

  function setTrainsEnabled(b) {
    forceDraw = b != trainsEnabled;
    trainsEnabled = b;
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

    if (map != null) {
      map.triggerRepaint();
    }
  }

  function prerender(gl, matrix) {
    var time = timeOffset + Date.now() / 1000;
    if (trainsEnabled) {
      RailViz.Trains.preRender(gl, time);
    }
  }

  function render(gl, matrix, zoom) {
    createOffscreenBuffer();

    let pre_scale = Math.min(1.0, Math.max(minZoom, zoom) * pixelRatio / 10);

    for (var i = 0; i <= 1; i++) {
      var isOffscreen = i == 0;

      gl.bindFramebuffer(
        gl.FRAMEBUFFER,
        isOffscreen ? offscreen.framebuffer : null
      );
      gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
      gl.enable(gl.BLEND);
      gl.disable(gl.DEPTH_TEST);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      // gl.blendFuncSeparate(
      //   gl.SRC_ALPHA,
      //   gl.ONE_MINUS_SRC_ALPHA,
      //   gl.ONE,
      //   gl.ONE_MINUS_SRC_ALPHA
      // );

      if (isOffscreen) {
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT);
      }

      if (trainsEnabled) {
        RailViz.Trains.render(gl, matrix, zoom, pre_scale, isOffscreen);
      }
    }

    lastFrame = performance.now();
    rafRequest = requestAnimationFrame(maybe_render);
  }

  function createOffscreenBuffer() {
    var width = gl.drawingBufferWidth;
    var height = gl.drawingBufferHeight;

    if (
      offscreen.width === width &&
      offscreen.height === height &&
      offscreen.framebuffer
    ) {
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
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      width,
      height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      null
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    offscreen.framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, offscreen.framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      offscreen.texture,
      0
    );

    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  function readOffscreenPixel(x, y) {
    if (!offscreen.framebuffer || !gl.isFramebuffer(offscreen.framebuffer)) {
      return null;
    }

    if (
      x < 0 ||
      y < 0 ||
      x >= gl.drawingBufferWidth ||
      y >= gl.drawingBufferHeight
    ) {
      return null;
    }

    var pixels = new Uint8Array(4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, offscreen.framebuffer);
    gl.readPixels(
      x,
      gl.drawingBufferHeight - y,
      1,
      1,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      pixels
    );

    return pixels;
  }

  function handleMouseEvent(eventType, event) {
    if (!map) {
      return;
    }
    const canvas = map.getCanvas();

    const mouseX = event.pageX - canvas.offsetLeft;
    const mouseY = event.pageY - canvas.offsetTop;
    const pickingX = mouseX * pixelRatio;
    const pickingY = mouseY * pixelRatio;
    const button = event.button;

    const offscreenColor = readOffscreenPixel(pickingX, pickingY);
    const pickedTrainIndex = RailViz.Trains.getPickedTrainIndex(offscreenColor);
    const pickedTrain =
      pickedTrainIndex != null ? data.trains[pickedTrainIndex] : null;

    const features = map.queryRenderedFeatures([mouseX, mouseY]);

    const station = features.find((e) => e.layer.id.endsWith("-stations"));
    const pickedStation =
      station !== undefined
        ? {
            id: station.properties.id,
            name: station.properties.name,
          }
        : null;

    const pickedConnections = RailViz.Path.Connections.getPickedConnections(
      features
    );

    if ((pickedTrain || pickedStation) && eventType != "mouseout") {
      canvas.style.cursor = "pointer";
    } else {
      canvas.style.cursor = "default";
    }

    mouseHandler(
      eventType,
      button,
      mouseX,
      mouseY,
      pickedTrain,
      pickedStation,
      pickedConnections
    );
  }

  return {
    init: init,
    setup: setup,
    stop: stop,
    setData: setData,
    setTimeOffset: setTimeOffset,
    prerender: prerender,
    render: render,
    setMinZoom: setMinZoom,
    setTargetFps: setTargetFps,
    setTrainsEnabled: setTrainsEnabled,
    handleMouseEvent: handleMouseEvent,
  };
})();
