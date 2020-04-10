var RailViz = RailViz || {};

RailViz.Trains = (function() {
  const vertexShader = `
        attribute vec4 a_startPos;
        attribute vec4 a_endPos;
        attribute float a_angle;
        attribute float a_progress;
        attribute vec4 a_delayColor;
        attribute vec4 a_categoryColor;
        attribute vec4 a_pickColor;
        
        uniform mat4 u_perspective;
        uniform float u_zoom;
        uniform bool u_useCategoryColor;
        
        varying vec4 v_color;
        varying vec4 v_pickColor;
        varying mat3 v_texTransform;

        void main() {
            if (a_progress < 0.0) {
                gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
            } else {
                vec4 startPrj = u_perspective * a_startPos;
                vec4 endPrj = u_perspective * a_endPos;
                gl_Position = mix(startPrj, endPrj, a_progress);
            }

            gl_PointSize = u_zoom * 4.0;
            v_color = u_useCategoryColor ? a_categoryColor : a_delayColor;
            v_pickColor = a_pickColor;

            float c = cos(a_angle);
            float s = sin(a_angle);

            v_texTransform = mat3(
              c, s, 0,
              -s, c, 0,
              -0.5 * c + 0.5 * s + 0.5, -0.5 * s - 0.5 * c + 0.5, 1
            );
        }
    `;

  const fragmentShader = `
        precision mediump float;
        
        uniform bool u_offscreen;
        uniform sampler2D u_texture;
        
        varying vec4 v_color;
        varying vec4 v_pickColor;
        varying mat3 v_texTransform;

        const vec4 transparent = vec4(0.0, 0.0, 0.0, 0.0);
        
        void main() {
            vec2 rotated = (v_texTransform * vec3(gl_PointCoord, 1.0)).xy;
            vec4 tex = texture2D(u_texture, rotated);
            if (u_offscreen) {
                gl_FragColor = tex.a == 0.0 ? transparent : v_pickColor;
            } else {
                gl_FragColor = v_color * tex;
            }
        }
    `;

  var trains = [];
  var routes = [];
  var useCategoryColor = true;
  var positionBuffer = null;
  var progressBuffer = null;
  var colorBuffer = null;
  var elementArrayBuffer = null;
  var positionData = null;
  var positionBufferInitialized = false;
  var positionBufferUpdated = false;
  var progressBufferInitialized = false;
  var colorBufferInitialized = false;
  var texture = null;
  var filteredIndices = null;
  var isFiltered = false;
  var filterBufferInitialized = false;
  var totalFrames = 0;
  var updatedBufferFrames = 0;
  var program;
  var a_startPos;
  var a_endPos;
  var a_angle;
  var a_progress;
  var a_delayColor;
  var a_categoryColor;
  var a_pickColor;
  var u_perspective;
  var u_zoom;
  var u_useCategoryColor;
  var u_offscreen;
  var u_texture;

  const PICKING_BASE = 0;

  const categoryColors = [
    [0x9c, 0x27, 0xb0], [0xe9, 0x1e, 0x63], [0x1a, 0x23, 0x7e],
    [0xf4, 0x43, 0x36], [0xf4, 0x43, 0x36], [0x4c, 0xaf, 0x50],
    [0x3f, 0x51, 0xb5], [0xff, 0x98, 0x00], [0xff, 0x98, 0x00],
    [0x9e, 0x9e, 0x9e]
  ];

  function init(newTrains, newRoutes) {
    trains = newTrains || [];
    routes = newRoutes || [];
    positionData = null;
    positionBufferInitialized = false;
    progressBufferInitialized = false;
    colorBufferInitialized = false;
    isFiltered = false;
    filterBufferInitialized = false;
    filteredIndices = null;
  }

  function setUseCategoryColor(useCategory) {
    useCategoryColor = useCategory;
  }

  function setup(gl) {
    const vshader = WebGL.Util.createShader(gl, gl.VERTEX_SHADER, vertexShader);
    const fshader =
        WebGL.Util.createShader(gl, gl.FRAGMENT_SHADER, fragmentShader);
    program = WebGL.Util.createProgram(gl, vshader, fshader);
    gl.deleteShader(vshader);
    gl.deleteShader(fshader);

    a_startPos = gl.getAttribLocation(program, 'a_startPos');
    a_endPos = gl.getAttribLocation(program, 'a_endPos');
    a_angle = gl.getAttribLocation(program, 'a_angle');
    a_progress = gl.getAttribLocation(program, 'a_progress');
    a_delayColor = gl.getAttribLocation(program, 'a_delayColor');
    a_categoryColor = gl.getAttribLocation(program, 'a_categoryColor');
    a_pickColor = gl.getAttribLocation(program, 'a_pickColor');
    u_perspective = gl.getUniformLocation(program, 'u_perspective');
    u_zoom = gl.getUniformLocation(program, 'u_zoom');
    u_useCategoryColor = gl.getUniformLocation(program, 'u_useCategoryColor');
    u_offscreen = gl.getUniformLocation(program, 'u_offscreen');
    u_texture = gl.getUniformLocation(program, 'u_texture');

    positionBuffer = gl.createBuffer();
    progressBuffer = gl.createBuffer();
    colorBuffer = gl.createBuffer();

    texture =
        WebGL.Util.createTextureFromCanvas(gl, RailViz.Textures.createTrain());

    positionBufferInitialized = false;
    progressBufferInitialized = false;
    colorBufferInitialized = false;
    filterBufferInitialized = false;
  }

  function preRender(gl, time) {
    if (!positionData) {
      fillPositionBuffer();
    }

    positionBufferUpdated = false;
    trains.forEach(
        (train, trainIndex) =>
            updateCurrentSubSegment(train, trainIndex, time));

    gl.useProgram(program);
    totalFrames++;
    if (!positionBufferInitialized) {
      initAndUploadPositionBuffer(gl);
      updatedBufferFrames++;
    } else if (positionBufferUpdated) {
      uploadUpdatedPositionBuffer(gl);
      updatedBufferFrames++;
    }
    fillProgressBuffer(gl);
    if (!colorBufferInitialized) {
      fillColorBuffer(gl);
    }

    // if (totalFrames % 300 == 0) {
    //   console.log(
    //       'position buffer uploaded:', updatedBufferFrames, '/', totalFrames,
    //       '=', (updatedBufferFrames / totalFrames * 100), '% of all frames');
    // }

    if (isFiltered && !filterBufferInitialized) {
      setupElementArrayBuffer(gl);
    }
  }

  function render(gl, perspective, zoom, pixelRatio, isOffscreen) {
    var trainCount = isFiltered ? filteredIndices.length : trains.length;
    if (trainCount == 0) {
      return;
    }

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(a_startPos);
    gl.vertexAttribPointer(a_startPos, 2, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(a_endPos);
    gl.vertexAttribPointer(a_endPos, 2, gl.FLOAT, false, 20, 8);
    gl.enableVertexAttribArray(a_angle);
    gl.vertexAttribPointer(a_angle, 1, gl.FLOAT, false, 20, 16);

    gl.bindBuffer(gl.ARRAY_BUFFER, progressBuffer);
    gl.enableVertexAttribArray(a_progress);
    gl.vertexAttribPointer(a_progress, 1, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.enableVertexAttribArray(a_delayColor);
    gl.vertexAttribPointer(a_delayColor, 3, gl.UNSIGNED_BYTE, true, 12, 0);
    gl.enableVertexAttribArray(a_categoryColor);
    gl.vertexAttribPointer(a_categoryColor, 3, gl.UNSIGNED_BYTE, true, 12, 4);
    gl.enableVertexAttribArray(a_pickColor);
    gl.vertexAttribPointer(a_pickColor, 3, gl.UNSIGNED_BYTE, true, 12, 8);

    gl.uniformMatrix4fv(u_perspective, false, perspective);
    gl.uniform1f(u_zoom, zoom * pixelRatio);
    gl.uniform1i(u_offscreen, isOffscreen);
    gl.uniform1i(u_useCategoryColor, useCategoryColor);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(u_texture, 0);

    if (isFiltered) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, elementArrayBuffer);
      gl.drawElements(gl.POINTS, trainCount, gl.UNSIGNED_SHORT, 0);
    } else {
      gl.drawArrays(gl.POINTS, 0, trainCount);
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    gl.disableVertexAttribArray(a_pickColor);
    gl.disableVertexAttribArray(a_categoryColor);
    gl.disableVertexAttribArray(a_delayColor);
    gl.disableVertexAttribArray(a_angle);
    gl.disableVertexAttribArray(a_endPos);
    gl.disableVertexAttribArray(a_startPos);
  }

  function initAndUploadPositionBuffer(gl) {
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positionData, gl.DYNAMIC_DRAW);
    positionBufferInitialized = true;
  }

  function uploadUpdatedPositionBuffer(gl) {
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, positionData);
  }

  function fillPositionBuffer() {
    positionData = new Float32Array(trains.length * 5);
    for (var i = 0; i < trains.length; i++) {
      updatePositionBuffer(i);
    }
    positionBufferUpdated = true;
  }

  function updatePositionBuffer(trainIndex) {
    const train = trains[trainIndex];
    const subSegmentIndex = train.currentSubSegmentIndex;
    const base = trainIndex * 5;
    if (subSegmentIndex != null) {
      const segment = routes[train.route_index].segments[train.segment_index];
      const polyline = segment.coordinates.coordinates;
      const polyOffset = subSegmentIndex * 2;

      const x0 = polyline[polyOffset], y0 = polyline[polyOffset + 1],
            x1 = polyline[polyOffset + 2], y1 = polyline[polyOffset + 3];
      const angle = -Math.atan2(y1 - y0, x1 - x0);

      // a_startPos
      positionData[base] = x0;
      positionData[base + 1] = y0;
      // a_endPos
      positionData[base + 2] = x1;
      positionData[base + 3] = y1;
      // a_angle
      positionData[base + 4] = angle;
    } else {
      // a_startPos
      positionData[base] = -100;
      positionData[base + 1] = -100;
      // a_endPos
      positionData[base + 2] = -100;
      positionData[base + 3] = -100;
      // a_angle
      positionData[base + 4] = 0;
    }
    positionBufferUpdated = true;
  }

  function fillProgressBuffer(gl) {
    var data = new Float32Array(trains.length);
    for (var i = 0; i < trains.length; i++) {
      const train = trains[i];
      if (train.currentSubSegmentProgress != null) {
        data[i] = train.currentSubSegmentProgress;
      } else {
        data[i] = -1.0;
      }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, progressBuffer);
    if (progressBufferInitialized) {
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    } else {
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      progressBufferInitialized = true;
    }
  }

  function updateCurrentSubSegment(train, trainIndex, time) {
    let updated = false;
    if (time < train.d_time || time > train.a_time) {
      updated = (train.currentSubSegmentIndex != null);
      train.currentSubSegmentIndex = null;
      train.currentSubSegmentProgress = null;
    } else {
      const progress = (time - train.d_time) / (train.a_time - train.d_time);
      const segment = routes[train.route_index].segments[train.segment_index];
      const totalPosition = progress * segment.totalLength;
      if (train.currentSubSegmentIndex != null) {
        const subOffset =
            segment.subSegmentOffsets[train.currentSubSegmentIndex];
        const subLen = segment.subSegmentLengths[train.currentSubSegmentIndex];
        if (totalPosition >= subOffset &&
            totalPosition <= (subOffset + subLen)) {
          train.currentSubSegmentProgress =
              (totalPosition - subOffset) / subLen;
          return;
        }
      }
      for (let i = train.currentSubSegmentIndex || 0;
           i < segment.subSegmentOffsets.length; i++) {
        const subOffset = segment.subSegmentOffsets[i];
        const subLen = segment.subSegmentLengths[i];
        if (totalPosition >= subOffset &&
            totalPosition <= (subOffset + subLen)) {
          updated = (train.currentSubSegmentIndex !== i);
          train.currentSubSegmentIndex = i;
          train.currentSubSegmentProgress =
              (totalPosition - subOffset) / subLen;
          break;
        }
      }
    }
    if (updated) {
      updatePositionBuffer(trainIndex);
    }
  }

  function fillColorBuffer(gl) {
    var data = new Uint8Array(trains.length * 12);
    for (var i = 0; i < trains.length; i++) {
      const train = trains[i];
      const base = i * 12;
      const pickColor = RailViz.Picking.pickIdToColor(PICKING_BASE + i);

      // a_delayColor
      setDelayColor(train, data, base);

      // a_categoryColor
      const categoryColor = categoryColors[train.clasz];
      data[base + 4] = categoryColor[0];
      data[base + 5] = categoryColor[1];
      data[base + 6] = categoryColor[2];

      // a_pickColor
      data[base + 8] = pickColor[0];
      data[base + 9] = pickColor[1];
      data[base + 10] = pickColor[2];
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    colorBufferInitialized = true;
  }

  const colDelay3Min = [69, 209, 74];
  const colDelay5Min = [255, 237, 0];
  const colDelay10Min = [255, 102, 0];
  const colDelay15Min = [255, 48, 71];
  const colDelayMax = [163, 0, 10];

  function setDelayColor(train, data, offset) {
    const delay = (train.d_time - train.sched_d_time) / 60;
    let color;
    if (delay <= 3) {
      color = colDelay3Min;
    } else if (delay <= 5) {
      color = colDelay5Min;
    } else if (delay <= 10) {
      color = colDelay10Min;
    } else if (delay <= 15) {
      color = colDelay15Min;
    } else {
      color = colDelayMax;
    }
    data[offset] = color[0];
    data[offset + 1] = color[1];
    data[offset + 2] = color[2];
  }

  function setFilteredIndices(indices) {
    filteredIndices = indices;
    isFiltered = indices != null;
    filterBufferInitialized = false;
  }

  function setupElementArrayBuffer(gl) {
    if (!elementArrayBuffer || !gl.isBuffer(elementArrayBuffer)) {
      elementArrayBuffer = gl.createBuffer();
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, elementArrayBuffer);
    gl.bufferData(
        gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(filteredIndices),
        gl.STATIC_DRAW);
    filterBufferInitialized = true;
  }

  function getPickedTrainIndex(pickId) {
    if (pickId != null) {
      const index = pickId - PICKING_BASE;
      if (index >= 0 && index < trains.length) {
        return index;
      }
    }
    return null;
  }

  return {
    init: init, setup: setup, render: render, preRender: preRender,
        setFilteredIndices: setFilteredIndices,
        getPickedTrainIndex: getPickedTrainIndex,
        setUseCategoryColor: setUseCategoryColor, categoryColors: categoryColors
  }
})();
