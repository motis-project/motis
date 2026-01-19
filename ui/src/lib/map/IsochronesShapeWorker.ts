import { circle } from '@turf/circle';
import { destination } from '@turf/destination';
import { featureCollection, point } from '@turf/helpers';
import { union } from '@turf/union';
import {
	isLess,
	type DisplayLevel,
	type Geometry,
	type IsochronesPos
} from '$lib/map/IsochronesShared';
type LngLat = {
	lng: number;
	lat: number;
};
type LngLatBounds = {
	_ne: LngLat;
	_sw: LngLat;
};
export type UpdateMessage =
	| { level: 'OVERLAY_RECTS'; data: LngLatBounds[] }
	| { level: 'OVERLAY_CIRCLES'; data: CircleType[] }
	| { level: 'GEOMETRY_CIRCLES'; data: Geometry | undefined };
export type ShapeMessage = { method: 'update-shape'; index: number } & UpdateMessage;
type RectType = { rect: LngLatBounds; distance: number; data: IsochronesPos };
type CircleType = ReturnType<typeof circle>;

let dataIndex = 0;
let data: IsochronesPos[] | undefined = undefined;
let rects: RectType[] | undefined = undefined;
let circles: CircleType[] | undefined = undefined;
let circleGeometry: Geometry | undefined = undefined;
let highestComputedLevel: DisplayLevel = 'NONE';
let maxLevel: DisplayLevel = 'NONE';
let working = false;
let maxDistance = (_: IsochronesPos) => 0;
let circleResolution: number = 64; // default 'steps' for @turf/circle

self.onmessage = async function (event) {
	const method = event.data.method;
	if (method == 'set-data') {
		resetState(event.data.index);
		data = event.data.data;
		const kilometersPerSecond = event.data.kilometersPerSecond;
		const maxSeconds = event.data.maxSeconds;
		maxDistance = getMaxDistanceFunction(kilometersPerSecond, maxSeconds);
		if (event.data.circleResolution && event.data.circleResolution > 2) {
			circleResolution = event.data.circleResolution;
		}
	} else if (method == 'set-max-level') {
		maxLevel = event.data.level;
		createShapes();
	}
};

function resetState(index: number) {
	dataIndex = index;
	rects = undefined;
	circles = undefined;
	circleGeometry = undefined;
	highestComputedLevel = 'NONE';
	maxLevel = 'NONE';
	maxDistance = (_: IsochronesPos) => 0;
}

function getMaxDistanceFunction(kilometersPerSecond: number, maxSeconds: number) {
	return (pos: IsochronesPos) => Math.min(pos.seconds, maxSeconds) * kilometersPerSecond;
}

async function createShapes() {
	if (working || !isLess(highestComputedLevel, maxLevel)) {
		return;
	}
	working = true;
	const workingIndex = dataIndex;
	const isStale = () => workingIndex != dataIndex;
	switch (highestComputedLevel) {
		case 'NONE':
			await createRects().then(async (allRects: RectType[]) => {
				if (isStale()) {
					console.log('Index got stale while computing rects');
					return;
				}
				rects = allRects;
				highestComputedLevel = 'OVERLAY_RECTS';
				self.postMessage({
					method: 'update-shape',
					index: dataIndex,
					level: 'OVERLAY_RECTS',
					data: rects.map((r) => r.rect)
				} as ShapeMessage);
				await filterNotContainedRects(allRects).then((notContainedRects: RectType[]) => {
					if (isStale()) {
						console.log('Index got stale deleting covered rects');
						return;
					}
					rects = notContainedRects;
					self.postMessage({
						method: 'update-shape',
						index: dataIndex,
						level: 'OVERLAY_RECTS',
						data: rects.map((r) => r.rect)
					} as ShapeMessage);
				});
			});
			break;
		case 'OVERLAY_RECTS':
			await createCircles().then((allCircles: CircleType[]) => {
				if (isStale()) {
					console.log('Index got stale while computing circles');
					return;
				}
				circles = allCircles;
				highestComputedLevel = 'OVERLAY_CIRCLES';
				self.postMessage({
					method: 'update-shape',
					index: dataIndex,
					level: 'OVERLAY_CIRCLES',
					data: circles
				} as ShapeMessage);
			});
			break;
		case 'OVERLAY_CIRCLES':
			await createUnion().then((geometry: Geometry | undefined) => {
				if (isStale()) {
					console.log('Index got stale while computing geometry');
					return;
				}
				circleGeometry = geometry;
				highestComputedLevel = 'GEOMETRY_CIRCLES';
				self.postMessage({
					method: 'update-shape',
					index: dataIndex,
					level: 'GEOMETRY_CIRCLES',
					data: circleGeometry
				} as ShapeMessage);
			});
			break;
		default:
			console.log(`Unexpected level '${highestComputedLevel}'`);
	}
	working = false;
	createShapes();
}

async function createRects() {
	if (data === undefined) {
		return [];
	}
	const promises = data.map(async (pos: IsochronesPos) => {
		const center = point([pos.lng, pos.lat]);
		const r = maxDistance(pos);
		const north = destination(center, r, 0, { units: 'kilometers' });
		const east = destination(center, r, 90, { units: 'kilometers' });
		const south = destination(center, r, 180, { units: 'kilometers' });
		const west = destination(center, r, -90, { units: 'kilometers' });
		return {
			rect: {
				_sw: { lng: west.geometry.coordinates[0], lat: south.geometry.coordinates[1] } as LngLat,
				_ne: { lng: east.geometry.coordinates[0], lat: north.geometry.coordinates[1] } as LngLat
			} as LngLatBounds,
			distance: r,
			data: pos
		};
	});
	return await Promise.all(promises);
}

function contains(larger: RectType, smaller: RectType): boolean {
	const r1 = larger.rect;
	const r2 = smaller.rect;
	return (
		r1._sw.lat <= r2._sw.lat &&
		r1._sw.lng <= r2._sw.lng &&
		r1._ne.lat >= r2._ne.lat &&
		r1._ne.lng >= r2._ne.lng
	);
}

async function filterNotContainedRects(allRects: RectType[]) {
	// Remove all rects, that are completely contained in at least one other
	// Sort by distance, descending
	allRects.sort((a: RectType, b: RectType) => b.distance - a.distance);
	const isCoveredPromises = allRects.map(async (box: RectType, index: number) =>
		allRects.slice(0, index).some((b: RectType) => contains(b, box))
	);
	const isCovered = await Promise.all(isCoveredPromises);
	const visibleBoxes = allRects.filter((_: RectType, index: number) => !isCovered[index]);
	return visibleBoxes;
}

async function createCircles() {
	if (rects === undefined) {
		return [];
	}
	const promises = rects.map(async (rect: RectType) => {
		const c = circle([rect.data.lng, rect.data.lat], rect.distance, {
			steps: circleResolution,
			units: 'kilometers'
		});
		// bbox extent in [minX, minY, maxX, maxY] order
		c.bbox = [rect.rect._sw.lng, rect.rect._sw.lat, rect.rect._ne.lng, rect.rect._ne.lat];
		return c;
	});
	return await Promise.all(promises);
}

// Implementation based on https://stackoverflow.com/a/75982694
// Create union for smaller polygons first
// Using a pipe like approach should place larger polygons at the end,
// reducing the number of expensive computations

async function createUnion() {
	if (circles === undefined) {
		return undefined;
	}
	const queue: Geometry[] = circles.map((c: CircleType) => c);
	while (queue.length > 1) {
		const a: Geometry = queue.shift()!;
		const b: Geometry = queue.shift()!;
		const u: Geometry | null = union(featureCollection([a, b]));
		if (u) {
			queue.push(u);
		}
	}
	return queue.length == 1 ? queue[0] : undefined;
}
