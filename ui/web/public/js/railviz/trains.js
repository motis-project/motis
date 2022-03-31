var RailViz = RailViz || {};

glMatrix.setMatrixArrayType(Float64Array);

RailViz.Trains = (function () {
  const vertexShader = `
        attribute float a_vertex;

        attribute highp vec4 a_startPos;
        attribute highp vec4 a_endPos;
        attribute highp float a_angle;
        attribute highp float a_progress;

        attribute vec4 a_delayColor;
        attribute vec4 a_categoryColor;
        attribute vec4 a_pickColor;

        uniform highp mat4 u_perspective;
        uniform highp float u_radius;

        uniform bool u_useCategoryColor;
        uniform bool u_offscreen;

        varying vec2 v_texCoord;
        varying vec4 v_color;

        void main() {
          // interpolate coordinates of current subsegment
          vec4 pc = mix(a_startPos, a_endPos, a_progress);

          // move vertices from center to corners and assign texture coordinates
          vec4 pv = pc;
          if(a_vertex == 0.0 || a_vertex == 2.0) {
            pv.x += u_radius;
            v_texCoord.s = 1.0;
          } else {
            pv.x -= u_radius;
          }
          if(a_vertex == 0.0 || a_vertex == 1.0) {
            pv.y += u_radius;
            v_texCoord.t = 1.0;
          } else {
            pv.y -= u_radius;
          }

          // compute rotation matrix
          float c = cos(-a_angle);
          float s = sin(-a_angle);
          mat4 rotate = mat4(
             c, s, 0, 0,
             -s, c, 0, 0,
             0, 0, 1, 0,
            pc.x - pc.x * c + pc.y * s, pc.y - pc.x * s - pc.y * c, 0, 1
          );

          // project: mercator -> rotate -> perspective -> screen
          gl_Position =  u_perspective * rotate * pv;

          // determine color
          if (u_offscreen) {
            v_color = a_pickColor;
          } else if(u_useCategoryColor) {
            v_color = a_categoryColor;
          } else {
            v_color = a_delayColor;
          }
        }
    `;

  const fragmentShader = `
        precision mediump float;

        uniform bool u_offscreen;
        uniform highp sampler2D u_texture;

        varying vec2 v_texCoord;
        varying vec4 v_color;

        void main() {
          gl_FragColor = texture2D(u_texture, v_texCoord);
          if(u_offscreen) {
            gl_FragColor = gl_FragColor.a == 0.0 ? vec4(0, 0, 0, 0) : v_color;
          } else {
            gl_FragColor = v_color * gl_FragColor;
          }
        }
    `;

  class VertexAttributes {
    constructor(gl, program) {
      this.a_vertex = gl.getAttribLocation(program, "a_vertex");
      this.buffer = gl.createBuffer();

      this.trainCount = null;
      this.bufferDirty = false;
    }

    setData(trains) {
      this.trainCount = trains.length;
      this.bufferDirty = true;
    }

    update(gl) {
      if (!this.bufferDirty || !this.trainCount) {
        return;
      }

      let data = new Uint8Array(trains.length * 6);
      for (let i = 0; i < trains.length; ++i) {
        data[i * 6 + 0] = 0; // south east
        data[i * 6 + 1] = 1; // north east
        data[i * 6 + 2] = 2; // south west
        data[i * 6 + 3] = 1; // north east
        data[i * 6 + 4] = 2; // south west
        data[i * 6 + 5] = 3; // north west
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      this.bufferDirty = false;
    }

    enable(gl, ext) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.enableVertexAttribArray(this.a_vertex);
      gl.vertexAttribPointer(this.a_vertex, 1, gl.UNSIGNED_BYTE, false, 0, 0);
      ext.vertexAttribDivisorANGLE(this.a_vertex, 0);
    }

    disable(gl) {
      gl.disableVertexAttribArray(this.a_vertex);
    }
  }

  class PositionAttributes {
    constructor(gl, program) {
      this.a_startPos = gl.getAttribLocation(program, "a_startPos");
      this.a_endPos = gl.getAttribLocation(program, "a_endPos");
      this.a_angle = gl.getAttribLocation(program, "a_angle");
      this.a_progress = gl.getAttribLocation(program, "a_progress");

      this.positionBuffer = gl.createBuffer();
      this.progressBuffer = gl.createBuffer();

      this.positionData = null;
      this.progressData = null;
      this.anchor = null;
    }

    setData(trains) {
      this.trains = trains;

      this.positionData = null;
      this.progressData = null;
      this.anchor = null;
    }

    update(gl, time) {
      if (this.anchor == null) {
        // first update after setData() was called
        this.anchor = this.computeAnchor();

        this.positionData = new Float32Array(this.trains.length * 5);
        this.progressData = new Float32Array(this.trains.length);
        for (var i = 0; i < this.trains.length; i++) {
          this.updateTrain(i, time);
          this.writePositionToBuffer(i);
          this.writeProgressToBuffer(i);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.positionData, gl.DYNAMIC_DRAW);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.progressBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.progressData, gl.DYNAMIC_DRAW);
      } else {
        // simulation continues: only the time has changed
        let uploadPositions = false;
        for (var i = 0; i < this.trains.length; i++) {
          if (this.updateTrain(i, time)) {
            this.writePositionToBuffer(i);
            uploadPositions = true;
          }
          this.writeProgressToBuffer(i);
        }

        if (uploadPositions) {
          gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
          gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.positionData);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, this.progressBuffer);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.progressData);
      }
    }

    computeAnchor() {
      let x = 0;
      let y = 0;
      this.trains.forEach((t) => {
        const first = t.firstMercatorCoordinate();
        const last = t.lastMercatorCoordinate();
        x += first.x + last.x;
        y += first.y + last.y;
      });

      return {
        x: x / (2 * this.trains.length),
        y: y / (2 * this.trains.length),
      };
    }

    getAnchor() {
      return vec3.fromValues(this.anchor.x, this.anchor.y, 0);
    }

    updateTrain(trainIndex, time) {
      return this.trains[trainIndex].updatePosition(time);
    }

    writePositionToBuffer(trainIndex) {
      const mercLine = this.trains[trainIndex].getMercatorLine();

      const offset = trainIndex * 5;
      if (mercLine != null) {
        const [x0, y0, x1, y1] = mercLine;
        const angle = -Math.atan2(y1 - y0, x1 - x0);

        // Move coordinates to the anchor: This seems to be enough to fix
        // noticable precision problems on higher zoom levels.
        this.positionData[offset + 0] = x0 - this.anchor.x; // a_startPos
        this.positionData[offset + 1] = y0 - this.anchor.y; // a_startPos
        this.positionData[offset + 2] = x1 - this.anchor.x; // a_endPos
        this.positionData[offset + 3] = y1 - this.anchor.y; // a_endPos
        this.positionData[offset + 4] = angle; // a_angle
      } else {
        this.positionData[offset + 0] = -100; // a_startPos
        this.positionData[offset + 1] = -100; // a_startPos
        this.positionData[offset + 2] = -100; // a_endPos
        this.positionData[offset + 3] = -100; // a_endPos
        this.positionData[offset + 4] = 0; // a_angle
      }
    }

    writeProgressToBuffer(trainIndex) {
      const train = trains[trainIndex];
      if (train.currProgress != null) {
        this.progressData[trainIndex] = train.currProgress;
      } else {
        this.progressData[trainIndex] = -1.0;
      }
    }

    enable(gl, ext) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
      gl.enableVertexAttribArray(this.a_startPos);
      gl.vertexAttribPointer(this.a_startPos, 2, gl.FLOAT, false, 20, 0);
      ext.vertexAttribDivisorANGLE(this.a_startPos, 1);
      gl.enableVertexAttribArray(this.a_endPos);
      gl.vertexAttribPointer(this.a_endPos, 2, gl.FLOAT, false, 20, 8);
      ext.vertexAttribDivisorANGLE(this.a_endPos, 1);
      gl.enableVertexAttribArray(this.a_angle);
      gl.vertexAttribPointer(this.a_angle, 1, gl.FLOAT, false, 20, 16);
      ext.vertexAttribDivisorANGLE(this.a_angle, 1);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.progressBuffer);
      gl.enableVertexAttribArray(this.a_progress);
      gl.vertexAttribPointer(this.a_progress, 1, gl.FLOAT, false, 0, 0);
      ext.vertexAttribDivisorANGLE(this.a_progress, 1);
    }

    disable(gl) {
      gl.disableVertexAttribArray(this.a_progress);

      gl.disableVertexAttribArray(this.a_angle);
      gl.disableVertexAttribArray(this.a_endPos);
      gl.disableVertexAttribArray(this.a_startPos);
    }
  }

  class ColorAttributes {
    categoryColors = [
      [0x90, 0xa4, 0xae], //  0 : AIR
      [0x9c, 0x27, 0xb0], //  1 : ICE
      [0xe9, 0x1e, 0x63], //  2 : IC
      [0x9c, 0xcc, 0x65], //  3 : COACH
      [0x1a, 0x23, 0x7e], //  4 : N
      [0xf4, 0x43, 0x36], //  5 : RE
      [0xf4, 0x43, 0x36], //  6 : RB
      [0x4c, 0xaf, 0x50], //  7 : S
      [0x3f, 0x51, 0xb5], //  8 : U
      [0xf5, 0x7c, 0x00], //  9 : long-distance busses + str
      [0xff, 0x98, 0x00], // 10 : short-distance busses + str
      [0x00, 0xac, 0xc1], // 11 : SHIP
      [0x9e, 0x9e, 0x9e], // 12 : OTHER
    ];

    constructor(gl, program) {
      this.a_delayColor = gl.getAttribLocation(program, "a_delayColor");
      this.a_categoryColor = gl.getAttribLocation(program, "a_categoryColor");
      this.a_pickColor = gl.getAttribLocation(program, "a_pickColor");

      this.buffer = gl.createBuffer();

      this.trains = null;
      this.bufferDirty = false;
    }

    setData(trains) {
      this.trains = trains;
      this.bufferDirty = true;
    }

    update(gl) {
      if (!this.bufferDirty || !this.trains) {
        return;
      }

      let data = new Uint8Array(this.trains.length * 12);
      for (let i = 0; i < this.trains.length; ++i) {
        const train = this.trains[i];
        const offset = i * 12;

        const delayColor = this.getDelayColor(train);
        data[offset + 0] = delayColor[0]; // a_delayColor
        data[offset + 1] = delayColor[1]; // a_delayColor
        data[offset + 2] = delayColor[2]; // a_delayColor

        const categoryColor = this.getCategoryColor(train);
        data[offset + 4] = categoryColor[0]; // a_categoryColor
        data[offset + 5] = categoryColor[1]; // a_categoryColor
        data[offset + 6] = categoryColor[2]; // a_categoryColor

        const pickColor = this.trainIndexToPickColor(i);
        data[offset + 8] = pickColor[0]; // a_pickColor
        data[offset + 9] = pickColor[1]; // a_pickColor
        data[offset + 10] = pickColor[2]; // a_pickColor
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      this.bufferDirty = false;
    }

    getCategoryColor(train) {
      if (train.clasz == 9 || train.clasz == 10) {
        if (train.route_distance > 10_000) {
          return this.categoryColors[9];
        } else {
          return this.categoryColors[10];
        }
      } else {
        return this.categoryColors[train.clasz];
      }
    }

    getDelayColor(train) {
      const delay = (train.d_time - train.sched_d_time) / 60;
      if (delay <= 3) {
        return [69, 209, 74];
      } else if (delay <= 5) {
        return [255, 237, 0];
      } else if (delay <= 10) {
        return [255, 102, 0];
      } else if (delay <= 15) {
        return [255, 48, 71];
      } else {
        return [163, 0, 10];
      }
    }

    trainIndexToPickColor(i) {
      return [i & 255, (i >>> 8) & 255, (i >>> 16) & 255];
    }

    pickColorToTrainIndex(color) {
      if (!color || color[3] === 0) {
        return null;
      }
      const index = color[0] + (color[1] << 8) + (color[2] << 16);
      if (index >= 0 && index < this.trains.length) {
        return index;
      }
      return null;
    }

    // prettier-ignore
    enable(gl, ext) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.enableVertexAttribArray(this.a_delayColor);
      gl.vertexAttribPointer(this.a_delayColor, 3, gl.UNSIGNED_BYTE, true, 12, 0);
      ext.vertexAttribDivisorANGLE(this.a_delayColor, 1);
      gl.enableVertexAttribArray(this.a_categoryColor);
      gl.vertexAttribPointer(this.a_categoryColor, 3, gl.UNSIGNED_BYTE, true, 12, 4);
      ext.vertexAttribDivisorANGLE(this.a_categoryColor, 1);
      gl.enableVertexAttribArray(this.a_pickColor);
      gl.vertexAttribPointer(this.a_pickColor, 3, gl.UNSIGNED_BYTE, true, 12, 8);
      ext.vertexAttribDivisorANGLE(this.a_pickColor, 1);
    }

    disable(gl) {
      gl.disableVertexAttribArray(this.a_pickColor);
      gl.disableVertexAttribArray(this.a_categoryColor);
      gl.disableVertexAttribArray(this.a_delayColor);
    }
  }

  let trains = [];

  let ext = null;
  let texture = null;

  let initialized = false;
  let vertexAttributes = null;
  let positionAttributes = null;
  let colorAttributes = null;

  let useCategoryColor = true;

  let u_perspective = null;
  let u_radius = null;
  let u_useCategoryColor = null;
  let u_offscreen = null;
  let u_texture = null;

  function setup(gl) {
    const vshader = WebGL.Util.createShader(gl, gl.VERTEX_SHADER, vertexShader);
    const fshader = WebGL.Util.createShader(
      gl,
      gl.FRAGMENT_SHADER,
      fragmentShader
    );
    program = WebGL.Util.createProgram(gl, vshader, fshader);
    gl.deleteShader(vshader);
    gl.deleteShader(fshader);

    ext = gl.getExtension("ANGLE_instanced_arrays");

    initialized = false;
    vertexAttributes = new VertexAttributes(gl, program);
    positionAttributes = new PositionAttributes(gl, program);
    colorAttributes = new ColorAttributes(gl, program);

    u_perspective = gl.getUniformLocation(program, "u_perspective");
    u_radius = gl.getUniformLocation(program, "u_radius");
    u_useCategoryColor = gl.getUniformLocation(program, "u_useCategoryColor");
    u_offscreen = gl.getUniformLocation(program, "u_offscreen");
    u_texture = gl.getUniformLocation(program, "u_texture");

    // prettier-ignore
    texture = WebGL.Util.createTextureFromCanvas(
      gl,RailViz.Textures.createTrain());
  }

  function setData(newTrains) {
    trains = newTrains || [];
    initialized = false;
  }

  function preRender(gl, time) {
    if (!initialized) {
      initialized = true;
      vertexAttributes.setData(trains);
      positionAttributes.setData(trains);
      colorAttributes.setData(trains);
    }

    gl.useProgram(program);

    vertexAttributes.update(gl);
    positionAttributes.update(gl, time);
    colorAttributes.update(gl);
  }

  function render(gl, perspective, zoom, pre_scale, isOffscreen) {
    if (trains.length == 0) {
      return;
    }

    gl.useProgram(program);

    vertexAttributes.enable(gl, ext);
    positionAttributes.enable(gl, ext);
    colorAttributes.enable(gl, ext);

    let p2 = mat4.create();
    mat4.fromTranslation(p2, positionAttributes.getAnchor());
    mat4.multiply(p2, perspective, p2);
    gl.uniformMatrix4fv(u_perspective, false, p2);

    // size of a pixel in mapboxgl mercator units
    const px_size = 1 / (512 * Math.pow(2, zoom)); // magic number YAY
    gl.uniform1f(
      u_radius,
      32.0 * pre_scale * px_size * RailViz.Style.interpolateTrainScale(zoom)
    );

    gl.uniform1i(u_offscreen, isOffscreen);
    gl.uniform1i(u_useCategoryColor, useCategoryColor);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(u_texture, 0);

    ext.drawArraysInstancedANGLE(gl.TRIANGLES, 0, 6, trains.length);

    colorAttributes.disable(gl);
    positionAttributes.disable(gl);
    vertexAttributes.disable(gl);
  }

  function setUseCategoryColor(useCategory) {
    useCategoryColor = useCategory;
  }

  function getPickedTrainIndex(color) {
    return colorAttributes.pickColorToTrainIndex(color);
  }

  return {
    setup: setup,
    setData: setData,
    preRender: preRender,
    render: render,
    getPickedTrainIndex: getPickedTrainIndex,
    setUseCategoryColor: setUseCategoryColor,
  };
})();
