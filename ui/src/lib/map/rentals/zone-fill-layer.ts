import earcut from 'earcut';
import { flatten } from 'earcut';
import maplibregl from 'maplibre-gl';
import { type CustomRenderMethodInput, type Map as MapLibreMap, type PointLike } from 'maplibre-gl';
import type { Position } from 'geojson';

import type { RentalZoneFeature, RentalZoneFeatureProperties } from './zone-types';

type GLContext = WebGLRenderingContext | WebGL2RenderingContext;

const DEFAULT_OPACITY = 0.4;
const STRIPE_WIDTH_PX = 6.0;
const STRIPE_OPACITY_VARIATION = 0.1;
const POSITION_COMPONENTS = 2;
const POSITION_STRIDE_BYTES = POSITION_COMPONENTS * Float32Array.BYTES_PER_ELEMENT;
const COLOR_COMPONENTS = 4;
const COLOR_STRIDE_BYTES = COLOR_COMPONENTS * Float32Array.BYTES_PER_ELEMENT;

const FILL_VERTEX_SHADER_SOURCE = `#version 300 es
precision highp float;
in vec2 a_pos;
in vec4 a_color;
out vec4 v_color;
uniform vec4 u_zone_base;
uniform vec4 u_zone_scale_x;
uniform vec4 u_zone_scale_y;
void main() {
	v_color = a_color;
	gl_Position = u_zone_base + a_pos.x * u_zone_scale_x + a_pos.y * u_zone_scale_y;
}
`;

const FILL_FRAGMENT_SHADER_SOURCE = `#version 300 es
precision mediump float;
in vec4 v_color;
out vec4 fragColor;
void main() {
	fragColor = v_color;
}
`;

type ZoneGeometry = {
	vertices: number[]; // mercator [x0, y0, x1, y1, ...]
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
};

type FillProgramState = {
	program: WebGLProgram;
	positionLocation: number;
	colorLocation: number;
	baseLocation: WebGLUniformLocation | null;
	scaleXLocation: WebGLUniformLocation | null;
	scaleYLocation: WebGLUniformLocation | null;
};

type ZoneFillLayerOptions = {
	id: string;
	opacity?: number;
};

const QUAD_VERTICES = new Float32Array([-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1]);

const SCREEN_VERTEX_SHADER_SOURCE = `
attribute vec2 a_pos;
attribute vec2 a_tex_coord;
varying vec2 v_tex_coord;
void main() {
	v_tex_coord = a_tex_coord;
	gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

const SCREEN_FRAGMENT_SHADER_SOURCE = `
precision highp float;
uniform sampler2D u_texture;
uniform float u_opacity_primary;
uniform float u_opacity_secondary;
uniform float u_stripe_width;
varying vec2 v_tex_coord;
void main() {
	vec4 color = texture2D(u_texture, v_tex_coord);
	if (color.a == 0.0) {
		discard;
	}
	float diagonal = gl_FragCoord.x + gl_FragCoord.y;
	float stripeIndex = mod(floor(diagonal / u_stripe_width), 2.0);
	float opacity = mix(u_opacity_primary, u_opacity_secondary, stripeIndex);
	gl_FragColor = vec4(color.rgb, color.a * opacity);
}
`;

const ZONE_COLOR_ALLOWED = new Float32Array([0.13333333, 0.77254902, 0.36862745, 1]); // #22c55e (green)
const ZONE_COLOR_FORBIDDEN = new Float32Array([0.9372549, 0.26666667, 0.26666667, 1]); // #ef4444 (red)
const ZONE_COLOR_RESTRICTED = new Float32Array([1, 0.84313725, 0, 1]); // #ffd700 (yellow)
const ZONE_COLOR_STATION = new Float32Array([0.25882354, 0.52156866, 0.95686275, 1]); // #4287f5 (blue)

export const getZoneColor = (properties: RentalZoneFeatureProperties) => {
	if (properties.stationArea) {
		return ZONE_COLOR_STATION;
	}
	if (properties.rideEndAllowed) {
		return ZONE_COLOR_ALLOWED;
	}
	if (!properties.rideThroughAllowed) {
		return ZONE_COLOR_FORBIDDEN;
	}
	return ZONE_COLOR_RESTRICTED;
};

const encodePickingColor = (i: number): Float32Array => {
	return new Float32Array([
		((i >> 16) & 0xff) / 0xff,
		((i >> 8) & 0xff) / 0xff,
		(i & 0xff) / 0xff,
		1
	]);
};

const decodePickingColor = (px: Uint8Array): number => {
	return (px[0] << 16) | (px[1] << 8) | px[2];
};

type ZoneClipFrame = {
	base: Float32Array;
	scaleX: Float32Array;
	scaleY: Float32Array;
};

const toClipCoordinates = (
	map: MapLibreMap,
	width: number,
	height: number,
	pixelRatioX: number,
	pixelRatioY: number,
	lngLat: maplibregl.LngLat
): Float32Array => {
	const point = map.project(lngLat);
	const px = point.x * pixelRatioX;
	const py = point.y * pixelRatioY;
	const clipX = (px / width) * 2 - 1;
	const clipY = 1 - (py / height) * 2;
	return new Float32Array([clipX, clipY, 0, 1]);
};

const computeZoneClipFrame = (
	map: MapLibreMap,
	width: number,
	height: number,
	pixelRatioX: number,
	pixelRatioY: number,
	origin: Float32Array,
	extent: Float32Array
): ZoneClipFrame => {
	const originLngLat = new maplibregl.MercatorCoordinate(origin[0], origin[1]).toLngLat();
	const maxLngLat = new maplibregl.MercatorCoordinate(
		origin[0] + extent[0],
		origin[1] + extent[1]
	).toLngLat();

	const base = toClipCoordinates(map, width, height, pixelRatioX, pixelRatioY, originLngLat);
	const maxClip = toClipCoordinates(map, width, height, pixelRatioX, pixelRatioY, maxLngLat);

	const scaleX = new Float32Array([maxClip[0] - base[0], 0, 0, 0]);
	const scaleY = new Float32Array([0, maxClip[1] - base[1], 0, 0]);

	return { base, scaleX, scaleY };
};

const toPoint = (p: PointLike): maplibregl.Point => {
	return Array.isArray(p) ? new maplibregl.Point(p[0], p[1]) : p;
};

export class ZoneFillLayer implements maplibregl.CustomLayerInterface {
	id: string;
	type: 'custom' = 'custom' as const;
	renderingMode: '2d' = '2d' as const;

	private opacity: number;
	private gl: GLContext | null = null;
	private map: MapLibreMap | null = null;
	private screenProgram: WebGLProgram | null = null;
	private framebuffer: WebGLFramebuffer | null = null;
	private texture: WebGLTexture | null = null;
	private pickingFramebuffer: WebGLFramebuffer | null = null;
	private pickingTexture: WebGLTexture | null = null;
	private quadBuffer: WebGLBuffer | null = null;
	private features: RentalZoneFeature[] = [];
	private geometryDirty = true;
	private width = 0;
	private height = 0;
	private pickingWidth = 0;
	private pickingHeight = 0;
	private fillProgram: FillProgramState | null = null;
	private pickingLookup = new Map<number, RentalZoneFeature>();
	private pickingPixel = new Uint8Array(4);
	private positionBuffer: WebGLBuffer | null = null;
	private colorBuffer: WebGLBuffer | null = null;
	private pickingColorBuffer: WebGLBuffer | null = null;
	private vertexCount = 0;
	private globalOrigin = new Float32Array([0, 0]);
	private globalExtent = new Float32Array([1, 1]);

	private screenPositionLocation = -1;
	private screenTexCoordLocation = -1;
	private screenTextureLocation: WebGLUniformLocation | null = null;
	private screenOpacityPrimaryLocation: WebGLUniformLocation | null = null;
	private screenOpacitySecondaryLocation: WebGLUniformLocation | null = null;
	private screenStripeWidthLocation: WebGLUniformLocation | null = null;

	constructor(options: ZoneFillLayerOptions) {
		this.id = options.id;
		this.opacity = options.opacity ?? DEFAULT_OPACITY;
	}

	setOpacity(opacity: number) {
		if (this.opacity === opacity) {
			return;
		}
		this.opacity = opacity;
		this.map?.triggerRepaint();
	}

	setFeatures(features: RentalZoneFeature[]) {
		this.features = features;
		this.geometryDirty = true;
		this.updateGeometry();
		this.map?.triggerRepaint();
	}

	onAdd(map: MapLibreMap, gl: GLContext) {
		this.map = map;
		this.gl = gl;
		this.initialize(gl);
		this.updateGeometry();
	}

	onRemove(_map: MapLibreMap, gl: GLContext) {
		this.cleanup(gl);
	}

	cleanup(gl?: GLContext) {
		if (!gl && this.gl) {
			gl = this.gl;
		}
		if (!gl) {
			return;
		}
		this.deleteGeometryBuffers(gl);
		if (this.framebuffer) {
			gl.deleteFramebuffer(this.framebuffer);
			this.framebuffer = null;
		}
		if (this.texture) {
			gl.deleteTexture(this.texture);
			this.texture = null;
		}
		if (this.pickingFramebuffer) {
			gl.deleteFramebuffer(this.pickingFramebuffer);
			this.pickingFramebuffer = null;
		}
		if (this.pickingTexture) {
			gl.deleteTexture(this.pickingTexture);
			this.pickingTexture = null;
		}
		if (this.quadBuffer) {
			gl.deleteBuffer(this.quadBuffer);
			this.quadBuffer = null;
		}
		if (this.fillProgram) {
			gl.deleteProgram(this.fillProgram.program);
			this.fillProgram = null;
		}
		this.pickingLookup.clear();
		if (this.screenProgram) {
			gl.deleteProgram(this.screenProgram);
			this.screenProgram = null;
		}
		this.geometryDirty = true;
		this.width = 0;
		this.height = 0;
		this.pickingWidth = 0;
		this.pickingHeight = 0;
		this.gl = null;
		this.map = null;
	}

	prerender(gl: GLContext, _options: CustomRenderMethodInput) {
		if (
			!this.map ||
			this.vertexCount === 0 ||
			!this.positionBuffer ||
			!this.colorBuffer ||
			!this.pickingColorBuffer
		) {
			return;
		}

		const map = this.map;
		const width = gl.drawingBufferWidth;
		const height = gl.drawingBufferHeight;
		if (width === 0 || height === 0) {
			return;
		}

		const fillProgram = this.ensureFillProgram(gl);
		if (!fillProgram || fillProgram.positionLocation < 0 || fillProgram.colorLocation < 0) {
			return;
		}

		const canvas = map.getCanvas();
		const rect = canvas.getBoundingClientRect();
		const cssWidth = rect.width;
		const cssHeight = rect.height;
		if (cssWidth === 0 || cssHeight === 0) {
			return;
		}

		const pixelRatioX = width / cssWidth;
		const pixelRatioY = height / cssHeight;
		const clipFrame = computeZoneClipFrame(
			map,
			width,
			height,
			pixelRatioX,
			pixelRatioY,
			this.globalOrigin,
			this.globalExtent
		);

		this.ensureFramebuffer(gl, width, height);
		this.ensurePickingFramebuffer(gl, width, height);
		if (!this.framebuffer || !this.texture || !this.pickingFramebuffer || !this.pickingTexture) {
			return;
		}

		const previousFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
		const previousViewport = gl.getParameter(gl.VIEWPORT) as Int32Array;
		const blendEnabled = gl.isEnabled(gl.BLEND);
		const depthTestEnabled = gl.isEnabled(gl.DEPTH_TEST);
		const stencilTestEnabled = gl.isEnabled(gl.STENCIL_TEST);
		const cullFaceEnabled = gl.isEnabled(gl.CULL_FACE);

		gl.disable(gl.BLEND);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.STENCIL_TEST);
		gl.disable(gl.CULL_FACE);

		const renderToFramebuffer = (fb: WebGLFramebuffer, colorBuffer: WebGLBuffer) => {
			gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
			gl.viewport(0, 0, width, height);
			gl.clearColor(0, 0, 0, 0);
			gl.clear(gl.COLOR_BUFFER_BIT);

			gl.useProgram(fillProgram.program);
			gl.uniform4fv(fillProgram.baseLocation, clipFrame.base);
			gl.uniform4fv(fillProgram.scaleXLocation, clipFrame.scaleX);
			gl.uniform4fv(fillProgram.scaleYLocation, clipFrame.scaleY);

			gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
			gl.enableVertexAttribArray(fillProgram.positionLocation);
			gl.vertexAttribPointer(
				fillProgram.positionLocation,
				POSITION_COMPONENTS,
				gl.FLOAT,
				false,
				POSITION_STRIDE_BYTES,
				0
			);

			gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
			gl.enableVertexAttribArray(fillProgram.colorLocation);
			gl.vertexAttribPointer(
				fillProgram.colorLocation,
				COLOR_COMPONENTS,
				gl.FLOAT,
				false,
				COLOR_STRIDE_BYTES,
				0
			);

			gl.drawArrays(gl.TRIANGLES, 0, this.vertexCount);

			gl.disableVertexAttribArray(fillProgram.positionLocation);
			gl.disableVertexAttribArray(fillProgram.colorLocation);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);
		};

		renderToFramebuffer(this.framebuffer, this.colorBuffer);
		renderToFramebuffer(this.pickingFramebuffer, this.pickingColorBuffer);

		gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);
		gl.viewport(previousViewport[0], previousViewport[1], previousViewport[2], previousViewport[3]);

		if (blendEnabled) {
			gl.enable(gl.BLEND);
		}
		if (depthTestEnabled) {
			gl.enable(gl.DEPTH_TEST);
		}
		if (stencilTestEnabled) {
			gl.enable(gl.STENCIL_TEST);
		}
		if (cullFaceEnabled) {
			gl.enable(gl.CULL_FACE);
		}
	}

	render(gl: GLContext, _options: CustomRenderMethodInput) {
		if (!this.screenProgram || !this.texture || !this.quadBuffer || this.vertexCount === 0) {
			return;
		}

		gl.useProgram(this.screenProgram);

		const prevBlendSrcRGB = gl.getParameter(gl.BLEND_SRC_RGB) as number;
		const prevBlendDstRGB = gl.getParameter(gl.BLEND_DST_RGB) as number;
		const prevBlendSrcAlpha = gl.getParameter(gl.BLEND_SRC_ALPHA) as number;
		const prevBlendDstAlpha = gl.getParameter(gl.BLEND_DST_ALPHA) as number;
		const prevBlendEquationRGB = gl.getParameter(gl.BLEND_EQUATION_RGB) as number;
		const prevBlendEquationAlpha = gl.getParameter(gl.BLEND_EQUATION_ALPHA) as number;

		gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

		gl.enableVertexAttribArray(this.screenPositionLocation);
		gl.enableVertexAttribArray(this.screenTexCoordLocation);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
		gl.vertexAttribPointer(this.screenPositionLocation, 2, gl.FLOAT, false, 16, 0);
		gl.vertexAttribPointer(this.screenTexCoordLocation, 2, gl.FLOAT, false, 16, 8);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this.texture);
		gl.uniform1i(this.screenTextureLocation, 0);
		const minOpacity = Math.max(this.opacity - STRIPE_OPACITY_VARIATION, 0.0);
		const maxOpacity = Math.min(this.opacity + STRIPE_OPACITY_VARIATION, 1.0);
		gl.uniform1f(this.screenOpacityPrimaryLocation, minOpacity);
		gl.uniform1f(this.screenOpacitySecondaryLocation, maxOpacity);
		gl.uniform1f(this.screenStripeWidthLocation, STRIPE_WIDTH_PX);

		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

		gl.bindTexture(gl.TEXTURE_2D, null);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);
		gl.disableVertexAttribArray(this.screenPositionLocation);
		gl.disableVertexAttribArray(this.screenTexCoordLocation);

		gl.blendFuncSeparate(prevBlendSrcRGB, prevBlendDstRGB, prevBlendSrcAlpha, prevBlendDstAlpha);
		gl.blendEquationSeparate(prevBlendEquationRGB, prevBlendEquationAlpha);
	}

	private initialize(gl: GLContext) {
		this.screenProgram = this.createProgram(
			gl,
			SCREEN_VERTEX_SHADER_SOURCE,
			SCREEN_FRAGMENT_SHADER_SOURCE
		);

		if (!this.screenProgram) {
			throw new Error('Failed to initialize zone fill shaders');
		}

		this.screenPositionLocation = gl.getAttribLocation(this.screenProgram, 'a_pos');
		this.screenTexCoordLocation = gl.getAttribLocation(this.screenProgram, 'a_tex_coord');
		this.screenTextureLocation = gl.getUniformLocation(this.screenProgram, 'u_texture');
		this.screenOpacityPrimaryLocation = gl.getUniformLocation(
			this.screenProgram,
			'u_opacity_primary'
		);
		this.screenOpacitySecondaryLocation = gl.getUniformLocation(
			this.screenProgram,
			'u_opacity_secondary'
		);
		this.screenStripeWidthLocation = gl.getUniformLocation(this.screenProgram, 'u_stripe_width');

		this.quadBuffer = gl.createBuffer();
		if (!this.quadBuffer) {
			throw new Error('Failed to allocate quad buffer');
		}
		gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, QUAD_VERTICES, gl.STATIC_DRAW);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);
	}

	private ensureFramebuffer(gl: GLContext, width: number, height: number) {
		if (this.framebuffer && this.texture && this.width === width && this.height === height) {
			return;
		}

		this.width = width;
		this.height = height;

		if (!this.framebuffer) {
			this.framebuffer = gl.createFramebuffer();
		}
		if (!this.texture) {
			this.texture = gl.createTexture();
		}
		if (!this.framebuffer || !this.texture) {
			return;
		}

		gl.bindTexture(gl.TEXTURE_2D, this.texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

		gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture, 0);

		const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
		if (status !== gl.FRAMEBUFFER_COMPLETE) {
			console.error('[ZoneFillLayer] Incomplete framebuffer:', status);
		}

		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}

	private ensurePickingFramebuffer(gl: GLContext, width: number, height: number) {
		if (
			this.pickingFramebuffer &&
			this.pickingTexture &&
			this.pickingWidth === width &&
			this.pickingHeight === height
		) {
			return;
		}

		if (!this.pickingFramebuffer) {
			this.pickingFramebuffer = gl.createFramebuffer();
		}
		if (!this.pickingTexture) {
			this.pickingTexture = gl.createTexture();
		}
		if (!this.pickingFramebuffer || !this.pickingTexture) {
			return;
		}

		gl.bindTexture(gl.TEXTURE_2D, this.pickingTexture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

		gl.bindFramebuffer(gl.FRAMEBUFFER, this.pickingFramebuffer);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT0,
			gl.TEXTURE_2D,
			this.pickingTexture,
			0
		);

		const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
		if (status !== gl.FRAMEBUFFER_COMPLETE) {
			console.error('[ZoneFillLayer] Incomplete picking framebuffer:', status);
		}

		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.bindTexture(gl.TEXTURE_2D, null);

		this.pickingWidth = width;
		this.pickingHeight = height;
	}

	private ensureFillProgram(gl: GLContext): FillProgramState | null {
		if (this.fillProgram) {
			return this.fillProgram;
		}

		const program = this.createProgram(gl, FILL_VERTEX_SHADER_SOURCE, FILL_FRAGMENT_SHADER_SOURCE);
		if (!program) {
			return null;
		}

		const state: FillProgramState = {
			program,
			positionLocation: gl.getAttribLocation(program, 'a_pos'),
			colorLocation: gl.getAttribLocation(program, 'a_color'),
			baseLocation: gl.getUniformLocation(program, 'u_zone_base'),
			scaleXLocation: gl.getUniformLocation(program, 'u_zone_scale_x'),
			scaleYLocation: gl.getUniformLocation(program, 'u_zone_scale_y')
		};

		this.fillProgram = state;
		return state;
	}

	private createProgram(
		gl: GLContext,
		vertexSrc: string,
		fragmentSrc: string
	): WebGLProgram | null {
		const vertexShader = this.compileShader(gl, gl.VERTEX_SHADER, vertexSrc);
		const fragmentShader = this.compileShader(gl, gl.FRAGMENT_SHADER, fragmentSrc);
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
		if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
			console.error('Failed to link shader program', gl.getProgramInfoLog(program));
			gl.deleteProgram(program);
			return null;
		}
		gl.deleteShader(vertexShader);
		gl.deleteShader(fragmentShader);
		return program;
	}

	private compileShader(gl: GLContext, type: number, source: string): WebGLShader | null {
		const shader = gl.createShader(type);
		if (!shader) {
			return null;
		}
		gl.shaderSource(shader, source);
		gl.compileShader(shader);
		if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
			console.error('Failed to compile shader', gl.getShaderInfoLog(shader));
			gl.deleteShader(shader);
			return null;
		}
		return shader;
	}

	private updateGeometry() {
		const gl = this.gl;
		if (!gl || !this.geometryDirty) {
			return;
		}

		this.deleteGeometryBuffers(gl);
		this.pickingLookup.clear();

		const features = [...this.features].sort((a, b) => a.properties.z - b.properties.z);
		const zoneGeometries: ZoneGeometry[] = [];
		const colorValues: number[] = [];
		const pickingValues: number[] = [];
		let totalVertices = 0;
		let globalMinX = Number.POSITIVE_INFINITY;
		let globalMinY = Number.POSITIVE_INFINITY;
		let globalMaxX = Number.NEGATIVE_INFINITY;
		let globalMaxY = Number.NEGATIVE_INFINITY;
		let zoneIdx = 0;

		for (const feature of features) {
			const geometry = this.buildZoneGeometry(feature);
			if (!geometry) {
				continue;
			}
			const vertexCount = geometry.vertices.length / POSITION_COMPONENTS;
			if (vertexCount === 0) {
				continue;
			}

			zoneGeometries.push(geometry);
			totalVertices += vertexCount;
			globalMinX = Math.min(globalMinX, geometry.minX);
			globalMinY = Math.min(globalMinY, geometry.minY);
			globalMaxX = Math.max(globalMaxX, geometry.maxX);
			globalMaxY = Math.max(globalMaxY, geometry.maxY);

			const pickingIdx = zoneIdx + 1;
			zoneIdx += 1;
			const color = getZoneColor(feature.properties);
			const pickingColor = encodePickingColor(pickingIdx);
			this.pickingLookup.set(pickingIdx, feature);

			for (let i = 0; i < vertexCount; ++i) {
				colorValues.push(color[0], color[1], color[2], color[3]);
				pickingValues.push(pickingColor[0], pickingColor[1], pickingColor[2], pickingColor[3]);
			}
		}

		const extentX = globalMaxX - globalMinX;
		const extentY = globalMaxY - globalMinY;

		if (
			totalVertices === 0 ||
			globalMinX === Number.POSITIVE_INFINITY ||
			globalMinY === Number.POSITIVE_INFINITY ||
			globalMaxX === Number.NEGATIVE_INFINITY ||
			globalMaxY === Number.NEGATIVE_INFINITY ||
			extentX === 0 ||
			extentY === 0
		) {
			this.vertexCount = 0;
			this.geometryDirty = false;
			return;
		}

		this.globalOrigin = new Float32Array([globalMinX, globalMinY]);
		this.globalExtent = new Float32Array([extentX, extentY]);

		const positions = new Float32Array(totalVertices * POSITION_COMPONENTS);
		let posOffset = 0;
		for (const geometry of zoneGeometries) {
			for (let i = 0; i < geometry.vertices.length; i += POSITION_COMPONENTS) {
				const x = geometry.vertices[i];
				const y = geometry.vertices[i + 1];
				positions[posOffset++] = (x - globalMinX) / extentX;
				positions[posOffset++] = (y - globalMinY) / extentY;
			}
		}

		const colorArray = new Float32Array(colorValues);
		const pickingArray = new Float32Array(pickingValues);

		this.positionBuffer = gl.createBuffer();
		this.colorBuffer = gl.createBuffer();
		this.pickingColorBuffer = gl.createBuffer();
		if (!this.positionBuffer || !this.colorBuffer || !this.pickingColorBuffer) {
			console.error('[ZoneFillLayer] Failed to allocate geometry buffers');
			this.deleteGeometryBuffers(gl);
			this.geometryDirty = false;
			return;
		}

		gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

		gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, colorArray, gl.STATIC_DRAW);

		gl.bindBuffer(gl.ARRAY_BUFFER, this.pickingColorBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, pickingArray, gl.STATIC_DRAW);

		gl.bindBuffer(gl.ARRAY_BUFFER, null);

		this.vertexCount = totalVertices;
		this.geometryDirty = false;
	}

	private deleteGeometryBuffers(gl: GLContext) {
		if (this.positionBuffer) {
			gl.deleteBuffer(this.positionBuffer);
			this.positionBuffer = null;
		}
		if (this.colorBuffer) {
			gl.deleteBuffer(this.colorBuffer);
			this.colorBuffer = null;
		}
		if (this.pickingColorBuffer) {
			gl.deleteBuffer(this.pickingColorBuffer);
			this.pickingColorBuffer = null;
		}
		this.vertexCount = 0;
		this.globalOrigin = new Float32Array([0, 0]);
		this.globalExtent = new Float32Array([1, 1]);
	}

	pickFeatureAt(pointLike: PointLike): RentalZoneFeature | null {
		if (!this.map || !this.gl || !this.pickingFramebuffer || !this.pickingTexture) {
			return null;
		}

		const pt = toPoint(pointLike);
		const canvas = this.map.getCanvas();
		const rect = canvas.getBoundingClientRect();
		if (rect.width === 0 || rect.height === 0) {
			return null;
		}
		const scaleX = canvas.width / rect.width;
		const scaleY = canvas.height / rect.height;
		const pixelX = Math.floor(pt.x * scaleX);
		const pixelY = Math.floor(canvas.height - pt.y * scaleY - 1);

		const previousFramebuffer = this.gl.getParameter(
			this.gl.FRAMEBUFFER_BINDING
		) as WebGLFramebuffer | null;
		this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.pickingFramebuffer);
		this.gl.readPixels(
			pixelX,
			pixelY,
			1,
			1,
			this.gl.RGBA,
			this.gl.UNSIGNED_BYTE,
			this.pickingPixel
		);
		this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, previousFramebuffer);

		const index = decodePickingColor(this.pickingPixel);
		if (index === 0) {
			return null;
		}

		return this.pickingLookup.get(index) ?? null;
	}

	private buildZoneGeometry(feature: RentalZoneFeature): ZoneGeometry | null {
		const vertices: number[] = [];
		let minX = Number.POSITIVE_INFINITY;
		let minY = Number.POSITIVE_INFINITY;
		let maxX = Number.NEGATIVE_INFINITY;
		let maxY = Number.NEGATIVE_INFINITY;

		const appendPoly = (coords: Position[][]) => {
			if (!coords.length) {
				return;
			}

			const mercatorCoords = coords.map((ring) =>
				ring.map(([lng, lat]) => {
					const merc = maplibregl.MercatorCoordinate.fromLngLat([lng, lat]);
					minX = Math.min(minX, merc.x);
					minY = Math.min(minY, merc.y);
					maxX = Math.max(maxX, merc.x);
					maxY = Math.max(maxY, merc.y);
					return [merc.x, merc.y];
				})
			);
			const data = flatten(mercatorCoords);
			const indices = earcut(data.vertices, data.holes, data.dimensions);
			const stride = data.dimensions;

			for (const index of indices) {
				const base = index * stride;
				vertices.push(data.vertices[base], data.vertices[base + 1]);
			}
		};

		for (const polygon of feature.geometry.coordinates) {
			appendPoly(polygon);
		}

		if (vertices.length === 0) {
			return null;
		}

		return { vertices, minX, minY, maxX, maxY };
	}
}
