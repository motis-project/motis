var Isochrone = Isochrone || {};

Isochrone.Render = (function () {
    const vertexShader = `
        uniform mat4 u_matrix;
        attribute vec2 a_pos;
        void main() {
            gl_Position = u_matrix * vec4(a_pos, 0.0, 1.0);
        }
    `;

    const fragmentShader = `
        precision mediump float;
        uniform vec4 fColor;
        void main() {
            gl_FragColor = fColor;   
        }
    `;

    var data;
    var timeOffset = 0;

    var map = null;
    var gl = null;


    var offscreenIso;
    var mouseHandler;
    var forceDraw;
    var lastFrame;
    var rafRequest;
    var targetFrameTime = null;
    var minZoom = 0;
    let pixelRatio = window.devicePixelRatio;
    let vertex_count = 256
    let isonum = 8;
    let circlesPerIso = new Array(isonum);
    let color = [
        [1.0, 0.0, 0.0, 0.5],   //red
        [1.0, 0.25, 0.0, 0.5],  //orange
        [1.0, 0.5, 0.0, 0.5],   //yellow
        [0.0, 1.0, 0.0, 0.5],   //green
        [0.0, 0.25, 1.0, 0.5], //cyan
        [0.0, 0.0, 1.0, 0.5],   //blue
        [0.75, 0.0, 0.75, 0.5], //purple
        [0.5, 0.0, 0.5, 0.5]    //darker purple
    ];

    let initialized = false;

    function init(mouseEventHandler) {
        setData(null);

        mouseHandler = mouseEventHandler || (() => {});
    }

    function setData(newData) {
        data = newData || {
            stations: [],
            times: []
        };
        forceDraw = true;
        initialized = false;
    }

    function setup(_map, _gl) {
        map = _map;
        gl = _gl;

        offscreenIso = {};



        const vshader = WebGL.Util.createShader(gl, gl.VERTEX_SHADER, vertexShader);
        const fshader = WebGL.Util.createShader(
            gl,
            gl.FRAGMENT_SHADER,
            fragmentShader
        );

        prog = WebGL.Util.createProgram(gl, vshader, fshader);
        gl.deleteShader(vshader);
        gl.deleteShader(fshader);

        gl.getExtension("OES_element_index_uint");

        this.a_pos = gl.getAttribLocation(prog, "a_pos");
        this.fColorLocation = gl.getUniformLocation(prog, "fColor");

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
        if (initialized) {
            return;
        }
        initialized = true;
        gl.useProgram(prog);

        const t0 = performance.now();
        if(data != undefined && data.stations.length!= 0 ) {

            let arr = new Float32Array(vertex_count * 2 * data.stations.length * isonum);
            let isodata = {stations: data.stations, times: data.times};
            for(let iso = 0; iso < isonum; iso++) {
                if(iso != 0) {
                    isodata.times = isodata.times.map(x => x - 900).filter(x => x >= 0);
                    isodata.stations = isodata.stations.slice(0, isodata.times.length);
                }
                circlesPerIso[iso] = isodata.times.length;
                for (let s = 0; s < isodata.stations.length; ++s) {
                    let coord = isodata.stations[s].pos;
                    let m = mapboxgl.MercatorCoordinate.fromLngLat(coord);
                    let t = isodata.times[s];

                    for (let i = 0; i < vertex_count; i++) {
                        let offsetInMeters = t * 1.5;
                        let deg = 2 * Math.PI * i / (vertex_count - 1);
                        let xOffset = Math.cos(deg) * offsetInMeters * m.meterInMercatorCoordinateUnits();
                        let yOffset = Math.sin(deg) * offsetInMeters * m.meterInMercatorCoordinateUnits();
                        let outerCoord = new mapboxgl.MercatorCoordinate(m.x + xOffset, m.y + yOffset, m.z);
                        arr[iso * data.stations.length * vertex_count * 2 + s * vertex_count * 2 + 2 * i] = outerCoord.x;
                        arr[iso * data.stations.length * vertex_count * 2 + s * vertex_count * 2 + 2 * i + 1] = outerCoord.y;
                    }
                }
            }
            const t1 = performance.now();

            console.log(`isodata took ${t1 - t0} milliseconds.`);

            this.buffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
            gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);

        }
        //TODO:prepare data
    }

    function render(gl, matrix, zoom) {
        if(data == undefined || data.stations.length == 0) {
            return;
        }
        //createoffscreenIsoBuffer();

        let pre_scale = Math.min(1.0, Math.max(minZoom, zoom) * pixelRatio / 10);

        for (var i = 0; i <= 1; i++) {
            var isoffscreenIso = i == 0;

            gl.bindFramebuffer(
                gl.FRAMEBUFFER,
                isoffscreenIso ? offscreenIso.framebuffer : null
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

            if (isoffscreenIso) {
                gl.clearColor(0, 0, 0, 0);
                gl.clear(gl.COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT);
            }
        }
        gl.useProgram(prog);

        gl.uniformMatrix4fv(
            gl.getUniformLocation(prog, 'u_matrix'),
            false,
            matrix
        );


        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
        gl.enableVertexAttribArray(this.a_pos);
        gl.vertexAttribPointer(this.a_pos, 2, gl.FLOAT, false, 0, 0);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ZERO);
        for(let i = 0; i < isonum; ++i) {
            gl.uniform4fv(this.fColorLocation, color[i]);
            for (let j = 0; j < circlesPerIso[i]; ++j) {
                gl.drawArrays(gl.TRIANGLE_FAN, j * vertex_count + data.stations.length * vertex_count * i, vertex_count);
            }
        }

        //TODO: Run Program

        lastFrame = performance.now();
        rafRequest = requestAnimationFrame(maybe_render);
    }

    function createoffscreenIsoBuffer() {
        var width = gl.drawingBufferWidth;
        var height = gl.drawingBufferHeight;

        if (
            offscreenIso.width === width &&
            offscreenIso.height === height &&
            offscreenIso.framebuffer
        ) {
            return;
        }

        offscreenIso.width = width;
        offscreenIso.height = height;

        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.bindRenderbuffer(gl.RENDERBUFFER, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        if (offscreenIso.framebuffer && gl.isFramebuffer(offscreenIso.framebuffer)) {
            gl.deleteFramebuffer(offscreenIso.framebuffer);
            offscreenIso.framebuffer = null;
        }
        if (offscreenIso.texture && gl.isTexture(offscreenIso.texture)) {
            gl.deleteTexture(offscreenIso.texture);
            offscreenIso.texture = null;
        }

        offscreenIso.texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, offscreenIso.texture);
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

        offscreenIso.framebuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, offscreenIso.framebuffer);
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER,
            gl.COLOR_ATTACHMENT0,
            gl.TEXTURE_2D,
            offscreenIso.texture,
            0
        );

        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    function generateIsochrones(newData) {
        setData(newData);
    }

    return {
        init: init,
        setup: setup,
        stop: stop,
        setData: setData,
        prerender: prerender,
        render: render,
        generateIsochrones: generateIsochrones
    };

})();