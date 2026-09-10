import maplibregl from 'maplibre-gl';
import type { CustomRenderMethodInput, Map as MapLibreMap } from 'maplibre-gl';

type GLContext = WebGLRenderingContext | WebGL2RenderingContext;

const parseColor = (color: string): Float32Array => {
	const value = Number.parseInt(color.replace('#', '').slice(0, 6), 16);
	return new Float32Array([
		((value >> 16) & 0xff) / 0xff,
		((value >> 8) & 0xff) / 0xff,
		(value & 0xff) / 0xff
	]);
};

export type IsochronesCircle = {
	lng: number;
	lat: number;
	radiusMeters: number;
	// Budget left over at the center of the disc and at its rim. Walking outwards
	// burns exactly one second per second, so the value in between is linear.
	centerSeconds: number;
	edgeSeconds: number;
};

// Two passes, both on the GPU:
//
//  1. prerender(): every reachable place is drawn as an instanced quad into an
//     offscreen texture, cut to a disc in the fragment shader. Each disc writes
//     its remaining travel budget, falling off linearly towards the rim.
//     Blending uses MAX, so every pixel keeps the largest remaining budget any
//     place could give it - i.e. the shortest travel time. That single blend
//     mode replaces both the polygon union and the per-stop time lookup that
//     used to be computed on the CPU.
//  2. render(): the field is composited over the map, mapping travel time
//     through the plasma colormap.
const UNION_VERTEX_SHADER = `#version 300 es
precision highp float;
in vec2 a_corner;
in vec3 a_circle; // xy: center relative to the data origin, z: radius (mercator units)
in vec2 a_time; // normalized remaining budget at the center (x) and at the rim (y)
uniform mat4 u_matrix;
out vec2 v_corner;
out vec2 v_time;
void main() {
	v_corner = a_corner;
	v_time = a_time;
	gl_Position = u_matrix * vec4(a_circle.xy + a_corner * a_circle.z, 0.0, 1.0);
}
`;

const UNION_FRAGMENT_SHADER = `#version 300 es
precision mediump float;
in vec2 v_corner;
in vec2 v_time;
out vec4 fragColor;
void main() {
	float d = length(v_corner);
	// One pixel wide edge fade. Clamped so that discs smaller than a pixel
	// stay visible instead of fading away completely.
	float aa = clamp(fwidth(d), 0.0, 0.5);
	float coverage = 1.0 - smoothstep(1.0 - aa, 1.0, d);
	if (coverage <= 0.0) {
		discard;
	}
	fragColor = vec4(mix(v_time.x, v_time.y, d), coverage, 0.0, 1.0);
}
`;

const COMPOSITE_VERTEX_SHADER = `#version 300 es
precision highp float;
in vec2 a_pos;
in vec2 a_uv;
out vec2 v_uv;
void main() {
	v_uv = a_uv;
	gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

// The plasma colormap from viridis / matplotlib: perceptually uniform and
// monotone in lightness, so equal travel time steps look equally far apart.
export const PLASMA = [
	'#0d0887',
	'#46039f',
	'#7201a8',
	'#9c179e',
	'#bd3786',
	'#d8576b',
	'#ed7953',
	'#fb9f3a',
	'#fdca26',
	'#f0f921'
];

const glslVec3 = (hex: string) =>
	`vec3(${Array.from(parseColor(hex), (v) => v.toFixed(4)).join(',')})`;

const COMPOSITE_FRAGMENT_SHADER = `#version 300 es
precision mediump float;
uniform sampler2D u_texture;
uniform float u_opacity;
in vec2 v_uv;
out vec4 fragColor;

const vec3 PLASMA[${PLASMA.length}] = vec3[${PLASMA.length}](${PLASMA.map(glslVec3).join(',')});

vec3 plasma(float t) {
	float x = clamp(t, 0.0, 1.0) * ${(PLASMA.length - 1).toFixed(1)};
	int i = min(int(x), ${PLASMA.length - 2});
	return mix(PLASMA[i], PLASMA[i + 1], x - float(i));
}

void main() {
	vec2 field = texture(u_texture, v_uv).rg;
	float a = field.g * u_opacity;
	if (a <= 0.0) {
		discard;
	}
	vec3 color = plasma(1.0 - field.r); // field.r is the budget left, so 1 - it is the travel time
	fragColor = vec4(color * a, a); // premultiplied, like MapLibre's own layers
}
`;

// Unit quad as triangle strip: clip space position + texture coordinate.
const QUAD_VERTICES = new Float32Array([-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1]);
const CORNERS = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);

// Per instance: center x, center y, radius, center budget, rim budget.
const CIRCLE_COMPONENTS = 5;
const CIRCLE_STRIDE_BYTES = CIRCLE_COMPONENTS * Float32Array.BYTES_PER_ELEMENT;

const compileShader = (gl: WebGL2RenderingContext, type: number, source: string) => {
	const shader = gl.createShader(type);
	if (!shader) {
		return null;
	}
	gl.shaderSource(shader, source);
	gl.compileShader(shader);
	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		console.error('[IsochronesLayer] failed to compile shader', gl.getShaderInfoLog(shader));
		gl.deleteShader(shader);
		return null;
	}
	return shader;
};

const createProgram = (gl: WebGL2RenderingContext, vertexSrc: string, fragmentSrc: string) => {
	const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSrc);
	const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSrc);
	if (!vertexShader || !fragmentShader) {
		return null;
	}
	const program = gl.createProgram();
	if (!program) {
		return null;
	}
	gl.attachShader(program, vertexShader);
	gl.attachShader(program, fragmentShader);
	gl.linkProgram(program);
	gl.deleteShader(vertexShader);
	gl.deleteShader(fragmentShader);
	if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
		console.error('[IsochronesLayer] failed to link program', gl.getProgramInfoLog(program));
		gl.deleteProgram(program);
		return null;
	}
	return program;
};

export class IsochronesLayer implements maplibregl.CustomLayerInterface {
	readonly id: string;
	readonly type = 'custom' as const;
	readonly renderingMode = '2d' as const;

	private map: MapLibreMap | null = null;
	private gl: WebGL2RenderingContext | null = null;

	private unionProgram: WebGLProgram | null = null;
	private compositeProgram: WebGLProgram | null = null;
	private cornerBuffer: WebGLBuffer | null = null;
	private circleBuffer: WebGLBuffer | null = null;
	private quadBuffer: WebGLBuffer | null = null;
	private unionVao: WebGLVertexArrayObject | null = null;
	private compositeVao: WebGLVertexArrayObject | null = null;
	private framebuffer: WebGLFramebuffer | null = null;
	private texture: WebGLTexture | null = null;
	private textureWidth = 0;
	private textureHeight = 0;

	private matrixLocation: WebGLUniformLocation | null = null;
	private textureLocation: WebGLUniformLocation | null = null;
	private opacityLocation: WebGLUniformLocation | null = null;

	// Circle centers are stored relative to this origin so that they keep full
	// precision as 32 bit floats even at high zoom levels. The offset is folded
	// back into the projection matrix, which is computed with doubles.
	private origin: [number, number] = [0, 0];
	private matrix = new Float32Array(16);
	private circles = new Float32Array(0);
	private circleCount = 0;
	private circlesDirty = false;

	private opacity = 1;
	private active = true;

	constructor(id: string) {
		this.id = id;
	}

	// maxSeconds is the full travel time budget of the query: it maps the
	// remaining budget of every disc onto the 0..1 range the shader works with.
	setCircles(circles: IsochronesCircle[], maxSeconds: number) {
		const scale = maxSeconds > 0 ? 1 / maxSeconds : 0;
		let originX = Number.POSITIVE_INFINITY;
		let originY = Number.POSITIVE_INFINITY;
		const merc = circles.map((c) => {
			const p = maplibregl.MercatorCoordinate.fromLngLat({ lng: c.lng, lat: c.lat });
			originX = Math.min(originX, p.x);
			originY = Math.min(originY, p.y);
			return { x: p.x, y: p.y, r: c.radiusMeters * p.meterInMercatorCoordinateUnits() };
		});

		this.origin = Number.isFinite(originX) ? [originX, originY] : [0, 0];
		this.circles = new Float32Array(merc.length * CIRCLE_COMPONENTS);
		for (let i = 0; i < merc.length; ++i) {
			const base = i * CIRCLE_COMPONENTS;
			this.circles[base] = merc[i].x - this.origin[0];
			this.circles[base + 1] = merc[i].y - this.origin[1];
			this.circles[base + 2] = merc[i].r;
			this.circles[base + 3] = Math.min(1, Math.max(0, circles[i].centerSeconds * scale));
			this.circles[base + 4] = Math.min(1, Math.max(0, circles[i].edgeSeconds * scale));
		}
		this.circleCount = merc.length;
		this.circlesDirty = true;
		this.map?.triggerRepaint();
	}

	setOpacity(opacity: number) {
		this.opacity = Math.min(1, Math.max(0, opacity));
		this.map?.triggerRepaint();
	}

	setActive(active: boolean) {
		if (this.active === active) {
			return;
		}
		this.active = active;
		this.map?.triggerRepaint();
	}

	onAdd(map: MapLibreMap, gl: GLContext) {
		this.map = map;
		if (typeof WebGL2RenderingContext === 'undefined' || !(gl instanceof WebGL2RenderingContext)) {
			console.error('[IsochronesLayer] WebGL2 is required to render isochrones');
			return;
		}
		this.gl = gl;
		this.initialize(gl);
	}

	onRemove(_map: MapLibreMap, gl: GLContext) {
		if (gl instanceof WebGL2RenderingContext) {
			this.release(gl);
		}
		this.map = null;
		this.gl = null;
	}

	// Called when the layer is dropped without MapLibre removing it (e.g. the
	// component is destroyed after the map is already gone).
	cleanup() {
		if (this.gl) {
			this.release(this.gl);
		}
		this.map = null;
		this.gl = null;
	}

	prerender(gl: GLContext, options: CustomRenderMethodInput) {
		if (!this.canRender(gl) || !this.unionProgram || !this.unionVao) {
			return;
		}
		const width = gl.drawingBufferWidth;
		const height = gl.drawingBufferHeight;
		if (width === 0 || height === 0) {
			return;
		}
		this.uploadCircles(gl);
		this.ensureRenderTarget(gl, width, height);
		if (!this.framebuffer) {
			return;
		}

		const previousFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
		const previousViewport = gl.getParameter(gl.VIEWPORT) as Int32Array;
		const depthTest = gl.isEnabled(gl.DEPTH_TEST);
		const stencilTest = gl.isEnabled(gl.STENCIL_TEST);
		const cullFace = gl.isEnabled(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.STENCIL_TEST);
		gl.disable(gl.CULL_FACE);

		gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
		gl.viewport(0, 0, width, height);
		gl.clearColor(0, 0, 0, 0);
		gl.clear(gl.COLOR_BUFFER_BIT);

		// Union instead of alpha accumulation: every fragment keeps the highest
		// coverage any disc wrote to it.
		gl.enable(gl.BLEND);
		gl.blendEquation(gl.MAX);

		gl.useProgram(this.unionProgram);
		gl.uniformMatrix4fv(this.matrixLocation, false, this.projectionMatrix(options));
		gl.bindVertexArray(this.unionVao);
		gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.circleCount);
		gl.bindVertexArray(null);

		gl.blendEquation(gl.FUNC_ADD);
		gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);
		gl.viewport(previousViewport[0], previousViewport[1], previousViewport[2], previousViewport[3]);
		if (depthTest) {
			gl.enable(gl.DEPTH_TEST);
		}
		if (stencilTest) {
			gl.enable(gl.STENCIL_TEST);
		}
		if (cullFace) {
			gl.enable(gl.CULL_FACE);
		}
	}

	render(gl: GLContext, _options: CustomRenderMethodInput) {
		if (!this.canRender(gl) || !this.compositeProgram || !this.compositeVao || !this.texture) {
			return;
		}

		gl.enable(gl.BLEND);
		gl.blendEquation(gl.FUNC_ADD);
		gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

		gl.useProgram(this.compositeProgram);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this.texture);
		gl.uniform1i(this.textureLocation, 0);
		gl.uniform1f(this.opacityLocation, this.opacity);

		gl.bindVertexArray(this.compositeVao);
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
		gl.bindVertexArray(null);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}

	private canRender(gl: GLContext): gl is WebGL2RenderingContext {
		return this.active && this.circleCount > 0 && this.gl !== null && gl === this.gl;
	}

	private initialize(gl: WebGL2RenderingContext) {
		this.unionProgram = createProgram(gl, UNION_VERTEX_SHADER, UNION_FRAGMENT_SHADER);
		this.compositeProgram = createProgram(gl, COMPOSITE_VERTEX_SHADER, COMPOSITE_FRAGMENT_SHADER);
		if (!this.unionProgram || !this.compositeProgram) {
			return;
		}

		this.matrixLocation = gl.getUniformLocation(this.unionProgram, 'u_matrix');
		this.textureLocation = gl.getUniformLocation(this.compositeProgram, 'u_texture');
		this.opacityLocation = gl.getUniformLocation(this.compositeProgram, 'u_opacity');

		this.cornerBuffer = gl.createBuffer();
		this.circleBuffer = gl.createBuffer();
		this.quadBuffer = gl.createBuffer();
		this.unionVao = gl.createVertexArray();
		this.compositeVao = gl.createVertexArray();
		if (
			!this.cornerBuffer ||
			!this.circleBuffer ||
			!this.quadBuffer ||
			!this.unionVao ||
			!this.compositeVao
		) {
			console.error('[IsochronesLayer] failed to allocate GL resources');
			return;
		}

		gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, CORNERS, gl.STATIC_DRAW);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, QUAD_VERTICES, gl.STATIC_DRAW);

		const cornerLocation = gl.getAttribLocation(this.unionProgram, 'a_corner');
		const circleLocation = gl.getAttribLocation(this.unionProgram, 'a_circle');
		const timeLocation = gl.getAttribLocation(this.unionProgram, 'a_time');
		gl.bindVertexArray(this.unionVao);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuffer);
		gl.enableVertexAttribArray(cornerLocation);
		gl.vertexAttribPointer(cornerLocation, 2, gl.FLOAT, false, 0, 0);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.circleBuffer);
		gl.enableVertexAttribArray(circleLocation);
		gl.vertexAttribPointer(circleLocation, 3, gl.FLOAT, false, CIRCLE_STRIDE_BYTES, 0);
		gl.vertexAttribDivisor(circleLocation, 1);
		gl.enableVertexAttribArray(timeLocation);
		gl.vertexAttribPointer(timeLocation, 2, gl.FLOAT, false, CIRCLE_STRIDE_BYTES, 12);
		gl.vertexAttribDivisor(timeLocation, 1);

		const posLocation = gl.getAttribLocation(this.compositeProgram, 'a_pos');
		const uvLocation = gl.getAttribLocation(this.compositeProgram, 'a_uv');
		gl.bindVertexArray(this.compositeVao);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
		gl.enableVertexAttribArray(posLocation);
		gl.vertexAttribPointer(posLocation, 2, gl.FLOAT, false, 16, 0);
		gl.enableVertexAttribArray(uvLocation);
		gl.vertexAttribPointer(uvLocation, 2, gl.FLOAT, false, 16, 8);

		gl.bindVertexArray(null);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

		this.circlesDirty = true;
	}

	private uploadCircles(gl: WebGL2RenderingContext) {
		if (!this.circlesDirty || !this.circleBuffer) {
			return;
		}
		gl.bindBuffer(gl.ARRAY_BUFFER, this.circleBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, this.circles, gl.DYNAMIC_DRAW);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);
		this.circlesDirty = false;
	}

	private ensureRenderTarget(gl: WebGL2RenderingContext, width: number, height: number) {
		if (this.framebuffer && this.textureWidth === width && this.textureHeight === height) {
			return;
		}
		this.framebuffer ??= gl.createFramebuffer();
		this.texture ??= gl.createTexture();
		if (!this.framebuffer || !this.texture) {
			return;
		}

		gl.bindTexture(gl.TEXTURE_2D, this.texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		// R: remaining travel budget (the isochrone field), G: coverage.
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG8, width, height, 0, gl.RG, gl.UNSIGNED_BYTE, null);

		const previousFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture, 0);
		const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
		if (status !== gl.FRAMEBUFFER_COMPLETE) {
			console.error('[IsochronesLayer] incomplete framebuffer', status);
		}
		gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);
		gl.bindTexture(gl.TEXTURE_2D, null);

		this.textureWidth = width;
		this.textureHeight = height;
	}

	// mainMatrix maps mercator coordinates to clip space. Multiplying it with a
	// translation to the data origin keeps the vertex data small: only the last
	// column changes, and it is computed here in double precision.
	private projectionMatrix(options: CustomRenderMethodInput): Float32Array {
		const m = options.defaultProjectionData.mainMatrix;
		const [ox, oy] = this.origin;
		const out = this.matrix;
		for (let i = 0; i < 12; ++i) {
			out[i] = m[i];
		}
		for (let i = 0; i < 4; ++i) {
			out[12 + i] = m[i] * ox + m[4 + i] * oy + m[12 + i];
		}
		return out;
	}

	private release(gl: WebGL2RenderingContext) {
		if (this.unionProgram) {
			gl.deleteProgram(this.unionProgram);
			this.unionProgram = null;
		}
		if (this.compositeProgram) {
			gl.deleteProgram(this.compositeProgram);
			this.compositeProgram = null;
		}
		for (const buffer of [this.cornerBuffer, this.circleBuffer, this.quadBuffer]) {
			if (buffer) {
				gl.deleteBuffer(buffer);
			}
		}
		this.cornerBuffer = null;
		this.circleBuffer = null;
		this.quadBuffer = null;
		if (this.unionVao) {
			gl.deleteVertexArray(this.unionVao);
			this.unionVao = null;
		}
		if (this.compositeVao) {
			gl.deleteVertexArray(this.compositeVao);
			this.compositeVao = null;
		}
		if (this.framebuffer) {
			gl.deleteFramebuffer(this.framebuffer);
			this.framebuffer = null;
		}
		if (this.texture) {
			gl.deleteTexture(this.texture);
			this.texture = null;
		}
		this.textureWidth = 0;
		this.textureHeight = 0;
		this.circlesDirty = true;
	}
}
