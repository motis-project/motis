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

type ZoneBuffer = {
	buffer: WebGLBuffer;
	vertexCount: number;
	color: Float32Array;
	pickingColor: Float32Array;
	pickingIdx: number;
	z: number;
};

type FillProgramState = {
	program: WebGLProgram;
	positionLocation: number;
	colorLocation: WebGLUniformLocation | null;
	projectionMatrixLocation: WebGLUniformLocation | null;
	fallbackMatrixLocation: WebGLUniformLocation | null;
	tileMercatorLocation: WebGLUniformLocation | null;
	clippingPlaneLocation: WebGLUniformLocation | null;
	projectionTransitionLocation: WebGLUniformLocation | null;
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
precision mediump float;
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

export const getZoneColor = (properties: RentalZoneFeatureProperties) => {
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
	private zoneBuffers: ZoneBuffer[] = [];
	private features: RentalZoneFeature[] = [];
	private geometryDirty = true;
	private width = 0;
	private height = 0;
	private pickingWidth = 0;
	private pickingHeight = 0;
	private fillPrograms = new Map<string, FillProgramState>();
	private pickingLookup = new Map<number, RentalZoneFeature>();
	private pickingPixel = new Uint8Array(4);

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
		this.clearZoneBuffers(gl);
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
		for (const program of this.fillPrograms.values()) {
			gl.deleteProgram(program.program);
		}
		this.fillPrograms.clear();
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

	prerender(gl: GLContext, options: CustomRenderMethodInput) {
		if (this.zoneBuffers.length === 0) {
			return;
		}

		const width = gl.drawingBufferWidth;
		const height = gl.drawingBufferHeight;
		if (width === 0 || height === 0) {
			return;
		}

		const fillProgram = this.getFillProgram(gl, options.shaderData);
		if (!fillProgram) {
			return;
		}

		const matrix = new Float32Array(options.defaultProjectionData.mainMatrix);
		const fallbackMatrix = new Float32Array(options.defaultProjectionData.fallbackMatrix);

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

		const renderToFramebuffer = (
			fb: WebGLFramebuffer,
			getZoneColor: (zone: ZoneBuffer) => Float32Array
		) => {
			gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
			gl.viewport(0, 0, width, height);
			gl.clearColor(0, 0, 0, 0);
			gl.clear(gl.COLOR_BUFFER_BIT);

			gl.useProgram(fillProgram.program);
			gl.uniformMatrix4fv(fillProgram.projectionMatrixLocation, false, matrix);
			gl.uniformMatrix4fv(fillProgram.fallbackMatrixLocation, false, fallbackMatrix);
			gl.uniform4fv(
				fillProgram.tileMercatorLocation,
				options.defaultProjectionData.tileMercatorCoords
			);
			gl.uniform4fv(fillProgram.clippingPlaneLocation, options.defaultProjectionData.clippingPlane);
			gl.uniform1f(
				fillProgram.projectionTransitionLocation,
				options.defaultProjectionData.projectionTransition
			);

			gl.enableVertexAttribArray(fillProgram.positionLocation);
			for (const zone of this.zoneBuffers) {
				gl.bindBuffer(gl.ARRAY_BUFFER, zone.buffer);
				gl.vertexAttribPointer(fillProgram.positionLocation, 2, gl.FLOAT, false, 0, 0);
				gl.uniform4fv(fillProgram.colorLocation, getZoneColor(zone));
				gl.drawArrays(gl.TRIANGLES, 0, zone.vertexCount);
			}
			gl.bindBuffer(gl.ARRAY_BUFFER, null);
			gl.disableVertexAttribArray(fillProgram.positionLocation);
		};

		renderToFramebuffer(this.framebuffer, (zone) => zone.color);
		renderToFramebuffer(this.pickingFramebuffer, (zone) => zone.pickingColor);

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
		if (!this.screenProgram || !this.texture || !this.quadBuffer || this.zoneBuffers.length === 0) {
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

	private getFillProgram(
		gl: GLContext,
		shaderData: CustomRenderMethodInput['shaderData']
	): FillProgramState | null {
		const cached = this.fillPrograms.get(shaderData.variantName);
		if (cached) {
			return cached;
		}

		const vertexSource = `#version 300 es
${shaderData.vertexShaderPrelude}
${shaderData.define}
in vec2 a_pos;
void main() {
	gl_Position = projectTile(a_pos);
}
`;

		const fragmentSource = `#version 300 es
precision mediump float;
uniform vec4 u_color;
out vec4 fragColor;
void main() {
	fragColor = u_color;
}
`;

		const program = this.createProgram(gl, vertexSource, fragmentSource);
		if (!program) {
			return null;
		}

		const state: FillProgramState = {
			program,
			positionLocation: gl.getAttribLocation(program, 'a_pos'),
			colorLocation: gl.getUniformLocation(program, 'u_color'),
			projectionMatrixLocation: gl.getUniformLocation(program, 'u_projection_matrix'),
			fallbackMatrixLocation: gl.getUniformLocation(program, 'u_projection_fallback_matrix'),
			tileMercatorLocation: gl.getUniformLocation(program, 'u_projection_tile_mercator_coords'),
			clippingPlaneLocation: gl.getUniformLocation(program, 'u_projection_clipping_plane'),
			projectionTransitionLocation: gl.getUniformLocation(program, 'u_projection_transition')
		};

		this.fillPrograms.set(shaderData.variantName, state);
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
		if (!this.gl || !this.geometryDirty) {
			return;
		}

		this.clearZoneBuffers(this.gl);
		this.pickingLookup.clear();

		const features = [...this.features].sort((a, b) => a.properties.z - b.properties.z);
		for (const feature of features) {
			const triangles = this.buildTriangles(feature);
			if (triangles.length === 0) {
				continue;
			}
			const buffer = this.gl.createBuffer();
			if (!buffer) {
				continue;
			}
			this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
			this.gl.bufferData(this.gl.ARRAY_BUFFER, triangles, this.gl.STATIC_DRAW);
			const pickingIdx = this.zoneBuffers.length + 1;
			const pickingColor = encodePickingColor(pickingIdx);
			this.pickingLookup.set(pickingIdx, feature);
			this.zoneBuffers.push({
				buffer,
				vertexCount: triangles.length / 2,
				color: getZoneColor(feature.properties),
				pickingColor,
				pickingIdx,
				z: feature.properties.z
			});
		}
		this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
		this.geometryDirty = false;
	}

	private clearZoneBuffers(gl: GLContext) {
		for (const zone of this.zoneBuffers) {
			gl.deleteBuffer(zone.buffer);
		}
		this.zoneBuffers = [];
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

	private buildTriangles(feature: RentalZoneFeature): Float32Array {
		const triangles: number[] = [];

		const appendPoly = (coords: Position[][]) => {
			if (!coords.length) {
				return;
			}

			const data = flatten(
				coords.map((ring) =>
					ring.map(([lng, lat]) => {
						const merc = maplibregl.MercatorCoordinate.fromLngLat([lng, lat]);
						return [merc.x, merc.y];
					})
				)
			);
			const indices = earcut(data.vertices, data.holes, data.dimensions);

			for (const index of indices) {
				const base = index * 2;
				triangles.push(data.vertices[base], data.vertices[base + 1]);
			}
		};

		for (const polygon of feature.geometry.coordinates) {
			appendPoly(polygon);
		}
		return new Float32Array(triangles);
	}
}
