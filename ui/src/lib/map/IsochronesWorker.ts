import type { circle } from '@turf/circle';
import type { LngLatBounds } from 'maplibre-gl';
import type { Position } from 'geojson';
import type { ShapeMessage, UpdateMessage } from '$lib/map/IsochronesShapeWorker';
import type { DisplayLevel, Geometry } from '$lib/map/IsochronesShared';
import ShapeWorker from '$lib/map/IsochronesShapeWorker.ts?worker';

export type WorkerMessage = {
	method: 'update-display-level';
	level: DisplayLevel;
	geometry?: Geometry | undefined;
	index: number;
};
type CircleType = ReturnType<typeof circle>;

let canvas: OffscreenCanvas | undefined = undefined;
let dataIndex = 0;
let shapeWorker: Worker | undefined = undefined;
let rects: LngLatBounds[] | undefined = undefined;
let circles: CircleType[] | undefined = undefined;

self.onmessage = async function (event) {
	const method = event.data.method;
	if (method == 'set-canvas') {
		canvas = event.data.canvas;
	} else if (method == 'update-data') {
		const index: number = event.data.index;
		const isochronesData = event.data.data;
		const kilometersPerSecond: number = event.data.kilometersPerSecond;
		const maxSeconds: number = event.data.maxSeconds;
		const circleResolution: number = event.data.circleResolution;
		dataIndex = index;
		rects = undefined;
		circles = undefined;
		const worker = createWorker();
		worker.postMessage({
			method: 'set-data',
			index: dataIndex,
			data: isochronesData,
			kilometersPerSecond,
			maxSeconds,
			circleResolution
		});
	} else if (method == 'set-max-display-level') {
		if (shapeWorker !== undefined) {
			const level: DisplayLevel = event.data.displayLevel;
			shapeWorker.postMessage({ method: 'set-max-level', level: level });
		}
	} else if (method == 'render-canvas') {
		if (!canvas) {
			return;
		}
		const boundingBox: LngLatBounds = event.data.boundingBox;
		const color: string = event.data.color;
		const dimensions: [number, number] = event.data.dimensions;
		const level: DisplayLevel = event.data.level;
		canvas.width = dimensions[0];
		canvas.height = dimensions[1];
		const ctx = canvas.getContext('2d');
		if (!ctx) {
			return;
		}

		const transform = getTransformer(boundingBox, dimensions);

		ctx.fillStyle = color;
		ctx.clearRect(0, 0, dimensions[0], dimensions[1]);

		if (level == 'OVERLAY_RECTS') {
			drawRects(ctx, transform);
		} else if (level == 'OVERLAY_CIRCLES') {
			const isVisible = getIsVisible(boundingBox);
			drawCircles(ctx, transform, isVisible);
		} else {
			console.log(`Cannot render level ${level}`);
		}
	}
};

function getTransformer(boundingBox: LngLatBounds, dimensions: [number, number]) {
	return (pos: Position) => {
		const x = Math.round(
			((pos[0] - boundingBox._sw.lng) / (boundingBox._ne.lng - boundingBox._sw.lng)) * dimensions[0]
		);
		const y = Math.round(
			((boundingBox._ne.lat - pos[1]) / (boundingBox._ne.lat - boundingBox._sw.lat)) * dimensions[1]
		);
		return [x, y];
	};
}

function getIsVisible(boundingBox: LngLatBounds) {
	return (circle: CircleType) => {
		if (!circle.bbox) {
			return false;
		}
		const b = circle.bbox; // [minX, minY, maxX, maxY]
		return (
			boundingBox._sw.lat <= b[3] &&
			b[1] <= boundingBox._ne.lat &&
			boundingBox._sw.lng <= b[2] &&
			b[0] <= boundingBox._ne.lat
		);
	};
}

function drawRects(ctx: OffscreenCanvasRenderingContext2D, transform: (p: Position) => Position) {
	if (rects === undefined) {
		return;
	}
	rects.forEach((rect: LngLatBounds) => {
		ctx.save(); // Store canvas state

		const min = transform([rect._sw.lng, rect._sw.lat]);
		const max = transform([rect._ne.lng, rect._ne.lat]);
		const diffX = max[0] - min[0];
		const diffY = max[1] - min[1];
		ctx.fillRect(min[0], min[1], diffX + 1, diffY + 1);
		// Restore previous state on top
		ctx.restore();
	});
}

function drawCircles(
	ctx: OffscreenCanvasRenderingContext2D,
	transform: (_: Position) => Position,
	isVisible: (_: CircleType) => boolean
) {
	if (circles === undefined) {
		return;
	}
	circles.filter(isVisible).forEach((circle: CircleType) => {
		ctx.save(); // Store canvas state

		const b = circle.bbox!; // Existence checked in filter()
		const min = transform([b[0], b[1]]);
		const max = transform([b[2], b[3]]);
		const diffX = max[0] - min[0];
		const diffY = max[1] - min[1];

		if (diffX < 2 && diffY < 2) {
			// Draw small rect
			ctx.fillRect(min[0], min[1], diffX + 1, diffY + 1);
		} else {
			// Clip circle
			ctx.beginPath();
			const coords = circle.geometry.coordinates[0];
			const start = transform(coords[0]);
			ctx.moveTo(start[0], start[1]);
			for (let i = 0; i < coords.length; ++i) {
				const pos = transform(coords[i]);
				ctx.lineTo(pos[0], pos[1]);
			}
			ctx.clip();

			// Fill bounding box, clipped to circle
			ctx.fillRect(min[0], min[1], diffX + 1, diffY + 1);
		}

		// Restore previous state on top
		ctx.restore();
	});
}

function createWorker() {
	shapeWorker?.terminate();

	shapeWorker = new ShapeWorker();

	shapeWorker.onmessage = (event: { data: ShapeMessage }) => {
		const method = event.data.method;
		switch (method) {
			case 'update-shape':
				{
					const index = event.data.index;
					if (index < dataIndex) {
						console.log(`Got stale index from shape worker (Got ${index}, expected ${dataIndex})`);
						return;
					}
					const msg: UpdateMessage = event.data;
					switch (msg.level) {
						case 'OVERLAY_RECTS':
							rects = msg.data as maplibregl.LngLatBounds[];
							self.postMessage({
								method: 'update-display-level',
								index: dataIndex,
								level: msg.level
							} as WorkerMessage);
							break;
						case 'OVERLAY_CIRCLES':
							circles = msg.data;
							self.postMessage({
								method: 'update-display-level',
								index: dataIndex,
								level: msg.level
							} as WorkerMessage);
							break;
						case 'GEOMETRY_CIRCLES':
							{
								const geometry = msg.data;
								self.postMessage({
									method: 'update-display-level',
									index: dataIndex,
									level: msg.level,
									geometry: geometry
								} as WorkerMessage);
							}
							break;
						default:
							console.log(`Unknown message '${msg}`);
					}
				}
				break;
			default:
				console.log(`Unknown method '${method}'`);
		}
	};
	return shapeWorker;
}
