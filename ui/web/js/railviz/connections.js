var RailViz = RailViz || {};

RailViz.Connections = (function () {
  const vertexShader = `
        attribute vec4 a_pos;
        attribute vec2 a_normal;
        attribute vec4 a_color;
        attribute vec4 a_pickColor;
        attribute vec4 a_highlightColor;
        attribute float a_highlightFlags;
        
        uniform vec2 u_resolution;
        uniform mat4 u_perspective;
        uniform float u_width;

        varying vec4 v_color;
        varying vec4 v_pickColor;

        void main() {
            float flags = a_highlightFlags;
            float width = u_width;
            
            if (flags >= 2.0) {
              flags -= 2.0;
              width *= 2.0;
            }
            
            vec4 base = u_perspective * a_pos;
            vec2 offset = width * a_normal / u_resolution;
            gl_Position = base + vec4(offset, 0.0, 0.0);
            
            if (flags >= 1.0) {
              v_color = a_highlightColor;
            } else {
              v_color = a_color;
            }
            v_pickColor = a_pickColor;
        }
    `;

  const fragmentShader = `
        precision mediump float;
        
        uniform bool u_offscreen;

        varying vec4 v_color;
        varying vec4 v_pickColor;
        
        void main() {
            if (u_offscreen) {
              gl_FragColor = v_pickColor;
            } else {
              gl_FragColor = v_color;
            }
        }
    `;


  // http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
  const colors = [
    [0x1f, 0x78, 0xb4], [0x33, 0xa0, 0x2c], [0xe3, 0x1a, 0x1c],
    [0xff, 0x7f, 0x00], [0x6a, 0x3d, 0x9a], [0xb1, 0x59, 0x28],
    [0xa6, 0xce, 0xe3], [0xb2, 0xdf, 0x8a], [0xfb, 0x9a, 0x99],
    [0xfd, 0xbf, 0x6f], [0xca, 0xb2, 0xd6], [0xff, 0xff, 0x99]
  ];

  const PICKING_BASE = 200000;
  const VERTEX_SIZE = 6;  // in 32-bit floats

  let trainSegments;
  let walkSegments;
  let lowestConnId = 0;
  let vertexCount = 0;
  let mesh;
  let highlightData = null;
  let mainBuffer = null;
  let highlightBuffer = null;
  let elementArrayBuffer = null;
  let mainBufferValid = false;
  let highlightBufferValid = false;
  let highlightedConnectionIds = null;
  let program;
  let a_pos;
  let a_normal;
  let a_color;
  let a_pickColor;
  let a_highlightColor;
  let a_highlightFlags;
  let u_resolution;
  let u_perspective;
  let u_width;
  let u_offscreen;

  function init(newTrainSegments, newWalkSegments, newLowestConnId) {
    if (newTrainSegments) {
      trainSegments = Array.from(newTrainSegments.values());
    } else {
      trainSegments = [];
    }
    if (newWalkSegments) {
      walkSegments = Array.from(newWalkSegments.values());
    } else {
      walkSegments = [];
    }
    lowestConnId = newLowestConnId;
    let trainVertexCount = trainSegments.reduce((acc, seg) => acc + seg.segment.coordinates.coordinates.length, 0);
    let walkVertexCount = walkSegments.reduce((acc, w) => acc + w.polyline.length, 0);
    vertexCount = trainVertexCount + walkVertexCount;
    highlightedConnectionIds = null;
    mesh = null;
    highlightData = null;
    mainBuffer = mainBuffer || null;
    highlightBuffer = highlightBuffer || null;
    mainBufferValid = false;
    highlightBufferValid = false;
  }

  function setup(gl) {
    const vshader = WebGL.Util.createShader(gl, gl.VERTEX_SHADER, vertexShader);
    const fshader =
      WebGL.Util.createShader(gl, gl.FRAGMENT_SHADER, fragmentShader);
    program = WebGL.Util.createProgram(gl, vshader, fshader);
    gl.deleteShader(vshader);
    gl.deleteShader(fshader);

    a_pos = gl.getAttribLocation(program, 'a_pos');
    a_normal = gl.getAttribLocation(program, 'a_normal');
    a_color = gl.getAttribLocation(program, 'a_color');
    a_pickColor = gl.getAttribLocation(program, 'a_pickColor');
    a_highlightColor = gl.getAttribLocation(program, 'a_highlightColor');
    a_highlightFlags = gl.getAttribLocation(program, 'a_highlightFlags');
    u_resolution = gl.getUniformLocation(program, 'u_resolution');
    u_perspective = gl.getUniformLocation(program, 'u_perspective');
    u_width = gl.getUniformLocation(program, 'u_width');
    u_offscreen = gl.getUniformLocation(program, 'u_offscreen');

    mainBuffer = gl.createBuffer();
    highlightBuffer = gl.createBuffer();
    elementArrayBuffer = gl.createBuffer();
    mainBufferValid = false;
    highlightBufferValid = false;
  }

  function render(gl, perspective, zoom, pixelRatio, isOffscreen) {
    if (vertexCount == 0) {
      return;
    }

    gl.useProgram(program);

    if (!mesh) {
      mesh = createMesh(gl);
    }

    if (!highlightData || !highlightBufferValid) {
      fillHighlightBuffer();
    }

    if (!mainBufferValid) {
      uploadMainBuffer(gl);
    }

    if (!highlightBufferValid) {
      uploadHighlightBuffer(gl);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, mainBuffer);
    gl.enableVertexAttribArray(a_pos);
    gl.vertexAttribPointer(a_pos, 2, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(a_normal);
    gl.vertexAttribPointer(a_normal, 2, gl.FLOAT, false, 24, 8);
    gl.enableVertexAttribArray(a_color);
    gl.vertexAttribPointer(a_color, 3, gl.UNSIGNED_BYTE, true, 24, 16);
    gl.enableVertexAttribArray(a_pickColor);
    gl.vertexAttribPointer(a_pickColor, 3, gl.UNSIGNED_BYTE, true, 24, 20);

    gl.bindBuffer(gl.ARRAY_BUFFER, highlightBuffer);
    gl.enableVertexAttribArray(a_highlightColor);
    gl.vertexAttribPointer(a_highlightColor, 3, gl.UNSIGNED_BYTE, true, 4, 0);
    gl.enableVertexAttribArray(a_highlightFlags);
    gl.vertexAttribPointer(a_highlightFlags, 1, gl.UNSIGNED_BYTE, false, 4, 3);

    // -height because y axis is flipped in webgl (-1 is at the bottom)
    gl.uniform2f(u_resolution, gl.canvas.width, -gl.canvas.height);
    gl.uniformMatrix4fv(u_perspective, false, perspective);
    gl.uniform1f(u_width, 4.0 * pixelRatio * 2048.0); // XXX
    gl.uniform1i(u_offscreen, isOffscreen);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, elementArrayBuffer);
    gl.drawElements(
      gl.TRIANGLES, mesh.elementArray.length, mesh.elementArrayType, 0);

    gl.disableVertexAttribArray(a_highlightFlags);
    gl.disableVertexAttribArray(a_highlightColor);
    gl.disableVertexAttribArray(a_pickColor);
    gl.disableVertexAttribArray(a_color);
    gl.disableVertexAttribArray(a_normal);
    gl.disableVertexAttribArray(a_pos);
  }

  function uploadMainBuffer(gl) {
    gl.bindBuffer(gl.ARRAY_BUFFER, mainBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.vertexArray, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, elementArrayBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.elementArray, gl.STATIC_DRAW);

    mainBufferValid = true;
  }

  function uploadHighlightBuffer(gl) {
    gl.bindBuffer(gl.ARRAY_BUFFER, highlightBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, highlightData, gl.DYNAMIC_DRAW);
    highlightBufferValid = true;
  }

  function fillHighlightBuffer() {
    if (highlightData) {
      highlightData.fill(0);
    } else {
      highlightData = new Float32Array(vertexCount * 2);
    }
    if (!highlightedConnectionIds) {
      return;
    }
    const byteView = new Uint8Array(highlightData.buffer);

    const color = colors[(highlightedConnectionIds[0] - lowestConnId) % colors.length];

    let highlightedFlags = 2; // highlighted
    if (highlightedConnectionIds.length === 1) {
      highlightedFlags = highlightedFlags | 1; // use highlight color
    }

    const highlightSeg = function (seg) {
      for (let vertexIndex = seg.firstVertexIndex; vertexIndex <= seg.lastVertexIndex; vertexIndex++) {
        const offset = vertexIndex * 4;
        byteView[offset] = color[0];
        byteView[offset + 1] = color[1];
        byteView[offset + 2] = color[2];
        byteView[offset + 3] = highlightedFlags;
      }
    };

    trainSegments.forEach(seg => {
      const highlighted = trainSegmentContainedInConnections(seg, highlightedConnectionIds);
      if (highlighted) {
        highlightSeg(seg);
      }
    });

    walkSegments.forEach(seg => {
      const highlighted = walkSegmentContainedInConnections(seg, highlightedConnectionIds);
      if (highlighted) {
        highlightSeg(seg);
      }
    });
  }

  function trainSegmentContainedInConnections(seg, connectionIds) {
    return seg.trips.some(t => t.connectionIds.some(id => connectionIds.indexOf(id) !== -1));
  }

  function walkSegmentContainedInConnections(walk, connectionIds) {
    return walk.connectionIds.some(id => connectionIds.indexOf(id) !== -1);
  }

  function createMesh(gl) {
    if (vertexCount == 0) {
      return {
        vertexArray: new Float32Array(0),
        elementArray: new Uint16Array(0),
        elementArrayType: gl.UNSIGNED_SHORT
      };
    }

    // miter joins only, no splits:
    // vertexCount = 2 * #pointsOnTheLine
    // triangleCount = (#pointsOnTheLine - 1) * 2 = vertexCount - 2
    // * 2 because splits / bevel joins may be necessary
    const triangleCount = (vertexCount * 2) - 2;
    const elementCount = triangleCount * 3;  // 3 vertices per triangle

    let vertexArray = new Float32Array(vertexCount * 2 * VERTEX_SIZE);
    const byteView = new Uint8Array(vertexArray.buffer);
    const uintExt = gl.getExtension('OES_element_index_uint');
    const elementArrayType = (/*elementCount > 65535 &&*/ uintExt) ?
      gl.UNSIGNED_INT :
      gl.UNSIGNED_SHORT;
    let elementArray = elementArrayType == gl.UNSIGNED_INT ?
      new Uint32Array(elementCount) :
      new Uint16Array(elementCount);

    let vertexIndex = 0;
    let elementIndex = 0;

    trainSegments.forEach((seg, segIdx) => {
      const r = createSegmentMesh(seg, segIdx, seg.segment.coordinates.coordinates, seg.color, vertexArray, vertexIndex, elementArray, elementIndex, byteView);
      vertexIndex = r.nextVertexIndex;
      elementIndex = r.nextElementIndex;
    });

    const walkPickOffset = trainSegments.length;
    walkSegments.forEach((walk, walkIdx) => {
      const r = createSegmentMesh(walk, walkPickOffset + walkIdx, walk.polyline, walk.color, vertexArray, vertexIndex, elementArray, elementIndex, byteView);
      vertexIndex = r.nextVertexIndex;
      elementIndex = r.nextElementIndex;
    });

    // set correct array sizes
    const vertexSize = vertexIndex;
    const elementSize = elementIndex;

    if (vertexSize < vertexArray.length) {
      vertexArray = vertexArray.subarray(0, vertexSize);
    }
    if (elementSize < elementArray.length) {
      elementArray = elementArray.subarray(0, elementSize);
    }

    return {
      vertexArray: vertexArray,
      elementArray: elementArray,
      elementArrayType: elementArrayType
    };
  }

  function createSegmentMesh(seg, pickIdx, coords, colorId, vertexArray, firstVertexIndex, elementArray, firstElementIndex, byteView) {
    const color = colors[colorId % colors.length];
    const pointCount = coords.length / 2;
    const subsegmentCount = pointCount - 1;
    let vertexIndex = firstVertexIndex;
    let elementIndex = firstElementIndex;

    if (pointCount < 2) {
      return {nextVertexIndex: vertexIndex, nextElementIndex: elementIndex};
    }

    // calculate unit normals for each segment
    const normals = new Array(subsegmentCount);
    for (let i = 0; i < subsegmentCount; i++) {
      const base = i * 2;
      const x0 = coords[base], y0 = coords[base + 1], x1 = coords[base + 2],
        y1 = coords[base + 3];
      // direction vector
      const direction = vec2.fromValues(x1 - x0, y1 - y0);
      vec2.normalize(direction, direction);
      // normal
      vec2.set(direction, -direction[1], direction[0]);
      normals[i] = direction;
    }

    const setFlags =
      function () {
        const colorBase = vertexIndex * 4 + 16;
        // color
        byteView[colorBase] = color[0];
        byteView[colorBase + 1] = color[1];
        byteView[colorBase + 2] = color[2];
        byteView[colorBase + 3] = 0;
        // pick color
        const pickColor = RailViz.Picking.pickIdToColor(PICKING_BASE + pickIdx);
        byteView[colorBase + 4] = pickColor[0];
        byteView[colorBase + 5] = pickColor[1];
        byteView[colorBase + 6] = pickColor[2];
        byteView[colorBase + 7] = 0;
      };

    // points -> vertices (2 vertices per point)
    for (let i = 0; i < pointCount; i++) {
      const prevSubsegment = i - 1;
      const nextSubsegment = i;

      let normal;
      let miterLen = 1.0;
      let split = false;

      if (prevSubsegment == -1) {
        // first point of the polyline
        normal = normals[nextSubsegment];
      } else if (nextSubsegment == subsegmentCount) {
        // last point of the polyline
        normal = normals[prevSubsegment];
      } else {
        const pn = normals[prevSubsegment];
        const nn = normals[nextSubsegment];
        // average normals
        normal = vec2.create();
        vec2.add(normal, pn, nn);
        if (vec2.length(normal) > 0) {
          vec2.normalize(normal, normal);
          miterLen = 1 / vec2.dot(normal, nn);
          if (miterLen > 2.0 || miterLen < 0.5) {
            split = true;
          }
        } else {
          split = true;
        }
        if (split) {
          vec2.copy(normal, pn);
          miterLen = 1.0;
        }
      }

      const x = coords[i * 2], y = coords[i * 2 + 1];

      const thisVertexIndex = vertexIndex;

      const addVertices = function () {
        vertexArray[vertexIndex] = x;
        vertexArray[vertexIndex + 1] = y;
        vertexArray[vertexIndex + 2] = normal[0] * miterLen;
        vertexArray[vertexIndex + 3] = normal[1] * miterLen;
        setFlags();
        vertexIndex += VERTEX_SIZE;

        vertexArray[vertexIndex] = x;
        vertexArray[vertexIndex + 1] = y;
        vertexArray[vertexIndex + 2] = normal[0] * -miterLen;
        vertexArray[vertexIndex + 3] = normal[1] * -miterLen;
        setFlags();
        vertexIndex += VERTEX_SIZE;
      };

      addVertices();

      if (i != 0) {
        const thisElementIndex = thisVertexIndex / VERTEX_SIZE;
        const prevElementIndex = thisElementIndex - 2;
        // 1st triangle
        elementArray[elementIndex++] = prevElementIndex;
        elementArray[elementIndex++] = prevElementIndex + 1;
        elementArray[elementIndex++] = thisElementIndex + 1;
        // 2nd triangle
        elementArray[elementIndex++] = prevElementIndex;
        elementArray[elementIndex++] = thisElementIndex + 1;
        elementArray[elementIndex++] = thisElementIndex;
      }

      if (split) {
        const prevElementIndex = thisVertexIndex / VERTEX_SIZE;

        // mid point for bevel join (the original point)
        const midElementIndex = vertexIndex / VERTEX_SIZE;
        vertexArray[vertexIndex] = x;
        vertexArray[vertexIndex + 1] = y;
        vertexArray[vertexIndex + 2] = 0;
        vertexArray[vertexIndex + 3] = 0;
        setFlags();
        vertexIndex += VERTEX_SIZE;

        const nextElementIndex = vertexIndex / VERTEX_SIZE;
        vec2.copy(normal, normals[nextSubsegment]);
        addVertices();

        // bevel join
        elementArray[elementIndex++] = prevElementIndex;
        elementArray[elementIndex++] = midElementIndex;
        elementArray[elementIndex++] = nextElementIndex;
      }
    }

    seg.firstVertexIndex = firstVertexIndex / VERTEX_SIZE;
    seg.lastVertexIndex = (vertexIndex - VERTEX_SIZE) / VERTEX_SIZE;

    return {nextVertexIndex: vertexIndex, nextElementIndex: elementIndex};
  }

  function getPickedSegment(pickId) {
    if (pickId != null) {
      const index = pickId - PICKING_BASE;
      if (index >= 0 && index < trainSegments.length) {
        return trainSegments[index];
      } else if (index >= trainSegments.length && index <= trainSegments.length + walkSegments.length) {
        return walkSegments[index - trainSegments.length];
      }
    }
    return null;
  }

  function highlightConnections(ids) {
    if (ids && ids.length > 0) {
      highlightedConnectionIds = ids;
    } else {
      highlightedConnectionIds = null;
    }
    highlightBufferValid = false;
  }

  return {
    init: init,
    setup: setup,
    render: render,
    getPickedSegment: getPickedSegment,
    highlightConnections: highlightConnections
  }
})();
