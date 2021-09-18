var Isochrone = Isochrone || {};

Isochrone.Render = (function () {
    const vertexShader = `
        attribute vec4 a_pos;
        attribute vec2 a_texco;
        
        uniform mat4 u_matrix;

        varying vec2 v_texco;

        void main() {
          gl_Position = u_matrix * a_pos;
          v_texco = a_texco;
        }
    `;

    const canvasFragmentShader = `
        precision mediump float;

        varying vec2 v_texco;

        uniform sampler2D u_tex;

        void main() {
            gl_FragColor = texture2D(u_tex, v_texco);
            //gl_FragColor = vec4(1.0, 0.0, 0.0, 0.5);
        }
    `;

    const textureFragmentShader = `
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
    var boundingBox;
    let pixelRatio = window.devicePixelRatio;
    let vertex_count = 1024;
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
    let walking_speed = 1.5;

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
        const tfshader = WebGL.Util.createShader(
            gl,
            gl.FRAGMENT_SHADER,
            textureFragmentShader
        );
        const cfshader = WebGL.Util.createShader(
            gl,
            gl.FRAGMENT_SHADER,
            canvasFragmentShader
        );

        tprog = WebGL.Util.createProgram(gl, vshader, tfshader);
        cprog = WebGL.Util.createProgram(gl, vshader, cfshader);
        gl.deleteShader(vshader);
        gl.deleteShader(tfshader);
        gl.deleteShader(cfshader);

        this.a_pos = gl.getAttribLocation(tprog, "a_pos");
        this.a_tpos = gl.getAttribLocation(cprog, "a_pos");
        this.texcoordLocation = gl.getAttribLocation(tprog, "a_texco");
        this.fColorLocation = gl.getUniformLocation(tprog, "fColor");

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

    function prepare() {

        //sort
        let list = [];
        for (let j = 0; j < data.stations.length; j++)
            list.push({'station': data.stations[j], 'time': data.times[j]});
        list.sort(function(a, b) {
            return ((a.time > b.time) ? -1 : ((a.time === b.time) ? 0 : 1));
        });
        for (let k = 0; k < list.length; k++) {
            data.stations[k] = list[k].station;
            data.times[k] = list[k].time;
        }

        //filter
        for(let i = 0; i < data.stations.length; i++) {
            let res = data.stations[i].pos;
            for(let j = i+1; j < data.stations.length; j++) {
                if(JSON.stringify(res)==JSON.stringify(data.stations[j].pos)) {
                    data.stations.splice(j,1);
                    data.times.splice(j,1);
                    j--;
                }
            }
        }

        //generate bounding box for texture
        //mercator coords go from 0,0 in NW corner to 1,1 in SE corner
        let minX = 1;
        let minY = 1;
        let maxX = 0;
        let maxY = 0;
        let m, r, north, east, south, west;
        for(let i = 0; i < data.stations.length; i++) {
            m = mapboxgl.MercatorCoordinate.fromLngLat(data.stations[i].pos);
            r = data.times[i] * walking_speed * m.meterInMercatorCoordinateUnits();
            north = new mapboxgl.MercatorCoordinate(m.x, m.y - r, m.z);
            east = new mapboxgl.MercatorCoordinate(m.x + r, m.y, m.z);
            south = new mapboxgl.MercatorCoordinate(m.x, m.y + r, m.z);
            west = new mapboxgl.MercatorCoordinate(m.x - r, m.y, m.z);
            if(north.y < minY) {
                minY = north.y;
            }
            if(east.x > maxX) {
                maxX = east.x;
            }
            if(south.y > maxY) {
                maxY = south.y;
            }
            if(west.x < minX) {
                minX = west.x;
            }
        }
        let center = {
            x: (minX+maxX)/2,
            y: (minY+maxY)/2
        }
        boundingBox = {
            minX : minX,
            minY : minY,
            maxX : maxX,
            maxY : maxY,
            center : center
        }
    }

    function generateProjectionMatrix(bb) {
        let p = mat4.create();

        let scaleX = 2/(bb.maxX-bb.minX);
        let scaleY = 2/(bb.maxY-bb.minY);
        let transMat = mat4.translate([],mat4.create(),[-bb.center.x,-bb.center.y,0]);
        let scaleMat = mat4.scale([],mat4.create(),[scaleX,scaleY,0]);
        let flipMat = mat4.create();
        flipMat[5]=-1;

        mat4.multiply(p,transMat,p);  //translate

        mat4.multiply(p,scaleMat,p);  //scale
        mat4.multiply(p, flipMat,p ); //flip

        return p;
    }

    function prerender(gl, matrix) {
        var time = timeOffset + Date.now() / 1000;
        if (initialized) {
            return;
        }
        initialized = true;


        const t0 = performance.now();
        if(data === undefined || data.stations.length === 0 ) {
            return;
        }

        prepare(); //prepare data and generate bounding box

        let vertices_per_circle = (vertex_count+2) * 2;
        let arr = new Float32Array(vertices_per_circle * data.stations.length * isonum);
        let isodata = {stations: data.stations, times: data.times};
        for(let iso = 0; iso < isonum; iso++) {
            if(iso !== 0) {
                isodata.times = isodata.times.map(x => x - 900).filter(x => x > 0);
                isodata.stations = isodata.stations.slice(0, isodata.times.length);
            } else {
                isodata.times = isodata.times.filter(x => x > 0);
                isodata.stations = isodata.stations.slice(0, isodata.times.length);
            }
            circlesPerIso[iso] = isodata.times.length;

            for (let s = 0; s < isodata.stations.length; ++s) {
                let coord = isodata.stations[s].pos;
                let m = mapboxgl.MercatorCoordinate.fromLngLat(coord);
                let t = isodata.times[s];
                let r = t * walking_speed * m.meterInMercatorCoordinateUnits();

                for (let i = 0; i <= vertex_count/2; i++) {
                    let offsetInMeters = t * walking_speed;
                    let deg = 2 * Math.PI * i / vertex_count;
                    let xOffset = Math.cos(deg) * r;
                    let yOffset = Math.sin(deg) * r;
                    let outerCoordUp = new mapboxgl.MercatorCoordinate(m.x + xOffset, m.y + yOffset, m.z);
                    let outerCoordDown = new mapboxgl.MercatorCoordinate(m.x + xOffset, m.y - yOffset, m.z);
                    arr[iso * data.stations.length * vertices_per_circle + s * vertices_per_circle + 4 * i + 0] = outerCoordUp.x;
                    arr[iso * data.stations.length * vertices_per_circle + s * vertices_per_circle + 4 * i + 1] = outerCoordUp.y;
                    arr[iso * data.stations.length * vertices_per_circle + s * vertices_per_circle + 4 * i + 2] = outerCoordDown.x;
                    arr[iso * data.stations.length * vertices_per_circle + s * vertices_per_circle + 4 * i + 3] = outerCoordDown.y;
                }
            }
        }
        const t1 = performance.now();

        console.log(`isodata took ${t1 - t0} milliseconds.`);


        // draw on texture offscreen
        this.texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.texture);
        let texWidth = 4096;
        let texHeight = 4096;
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA,
            texWidth, texHeight, 0,
            gl.RGBA, gl.UNSIGNED_BYTE, null);

        this.fb = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
        gl.viewport(0, 0, texWidth, texHeight);
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture, 0);

        let buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);


        gl.useProgram(tprog);

        gl.enableVertexAttribArray(this.a_pos);
        gl.vertexAttribPointer(
            this.a_pos, 2, gl.FLOAT, false, 0, 0);

        let projectionMatrix = generateProjectionMatrix(boundingBox);
        gl.uniformMatrix4fv(
            gl.getUniformLocation(tprog, 'u_matrix'),
            false,
            projectionMatrix

        );

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ZERO);

        for(let i = 0; i < isonum; ++i) {
            gl.uniform4fv(this.fColorLocation, color[i]);
            gl.drawArrays(gl.TRIANGLE_STRIP, data.stations.length * (vertex_count+2) * i, (vertex_count+2) * circlesPerIso[i]);

        }
        // anti-aliasing when zooming out
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST);
        //gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        //gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        //gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        //prepare data for drawing texture on map


        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        this.buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
        let mapQuad = new Float32Array([
            boundingBox.minX, boundingBox.maxY,
            boundingBox.minX, boundingBox.minY,
            boundingBox.maxX, boundingBox.maxY,

            boundingBox.minX, boundingBox.minY,
            boundingBox.maxX, boundingBox.minY ,
            boundingBox.maxX, boundingBox.maxY
        ]);
        gl.bufferData(
            gl.ARRAY_BUFFER,
            mapQuad,
            gl.STATIC_DRAW
        );

        this.texcoordBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.texcoordBuffer);
        gl.bufferData(
            gl.ARRAY_BUFFER,
            new Float32Array([
                0, 0,
                0, 1,
                1, 0,
                0, 1,
                1, 1,
                1, 0
            ]),
            gl.STATIC_DRAW
        );
        //TODO:prepare data
    }

    function render(gl, matrix, zoom) {
        if(data === undefined || data.stations.length === 0) {
            return;
        }

        let pre_scale = Math.min(1.0, Math.max(minZoom, zoom) * pixelRatio / 10);

        gl.useProgram(cprog);

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        this.buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
        let mapQuad = new Float32Array([
            boundingBox.minX, boundingBox.maxY,
            boundingBox.minX, boundingBox.minY,
            boundingBox.maxX, boundingBox.maxY,

            boundingBox.minX, boundingBox.minY,
            boundingBox.maxX, boundingBox.minY ,
            boundingBox.maxX, boundingBox.maxY
        ]);
        gl.bufferData(
            gl.ARRAY_BUFFER,
            mapQuad,
            gl.STATIC_DRAW
        );

        this.texcoordBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.texcoordBuffer);
        gl.bufferData(
            gl.ARRAY_BUFFER,
            new Float32Array([
                0, 0,
                0, 1,
                1, 0,
                0, 1,
                1, 1,
                1, 0
            ]),
            gl.STATIC_DRAW
        );
        gl.uniformMatrix4fv(
            gl.getUniformLocation(cprog, 'u_matrix'),
            false,
            matrix
        );
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);

        gl.enableVertexAttribArray(this.a_tpos);
        gl.vertexAttribPointer(this.a_tpos, 2, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.texcoordBuffer);

        gl.enableVertexAttribArray(this.texcoordLocation);
        gl.vertexAttribPointer(this.texcoordLocation, 2, gl.FLOAT, false, 0, 0);

        const texLoc = gl.getUniformLocation(cprog, "u_tex");
        gl.activeTexture(gl.TEXTURE0);
        gl.uniform1i(texLoc, 0);
        gl.bindTexture(gl.TEXTURE_2D, this.texture);
        gl.drawArrays(gl.TRIANGLES, 0, 6);


        //TODO: Run Program

        lastFrame = performance.now();
        rafRequest = requestAnimationFrame(maybe_render);
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