<script lang="ts">
	import bbox from '@turf/bbox';
	import circle from '@turf/circle';
	import maplibregl, { CanvasSource, type LngLatBoundsLike, type Map } from 'maplibre-gl';
	import type { PrePostDirectMode } from '$lib/Modes';

	export interface IsochronesPos {
		lat: number;
		lng: number;
		seconds: number;
	}

	type BoxCoordsType = [[number, number], [number, number], [number, number], [number, number]];
	type CircleType = ReturnType<typeof circle>;

	let {
		map,
		bounds,
		isochronesData,
		streetModes,
		wheelchair,
		maxAllTime,
		active,
		color,
		opacity
	}: {
		map: Map | undefined;
		bounds: LngLatBoundsLike | undefined;
		isochronesData: IsochronesPos[];
		streetModes: PrePostDirectMode[];
		wheelchair: boolean;
		maxAllTime: number;
		active: boolean;
		color: string;
		opacity: number;
	} = $props();

	const name = 'isochrones-data';
	let loaded = false;

	let lastData: IsochronesPos[] | undefined = undefined;
	let lastAllTime: number = maxAllTime;
	let lastSpeed: number | undefined = undefined;

	const kilometersPerSecond = $derived(
		streetModes.includes('BIKE')
			? 0.0038 // 3.8 meters per second
			: wheelchair
				? 0.0008 // 0.8 meters per second
				: 0.0012 // 1.2 meters per second
	);
	const boundingBox = $derived(
		maplibregl.LngLatBounds.convert(
			bounds ?? [
				[0, 0],
				[1, 1]
			]
		)
	);
	const boxCoords: BoxCoordsType = $derived([
		[boundingBox._sw.lng, boundingBox._ne.lat],
		[boundingBox._ne.lng, boundingBox._ne.lat],
		[boundingBox._ne.lng, boundingBox._sw.lat],
		[boundingBox._sw.lng, boundingBox._sw.lat]
	]);
	function reachableKilometers(pos: IsochronesPos) {
		return Math.min(pos.seconds, maxAllTime) * kilometersPerSecond;
	}
	function transform(pos: number[], dimensions: number[]) {
		const x = Math.round(
			((pos[0] - boundingBox._sw.lng) / (boundingBox._ne.lng - boundingBox._sw.lng)) * dimensions[0]
		);
		const y = Math.round(
			((boundingBox._ne.lat - pos[1]) / (boundingBox._ne.lat - boundingBox._sw.lat)) * dimensions[1]
		);
		return [x, y];
	}

	let circles = $state<CircleType[] | undefined>(undefined);
	$effect(() => {
		if (
			!active ||
			(lastData == isochronesData && lastAllTime == maxAllTime && lastSpeed == kilometersPerSecond)
		) {
			return;
		}
		circles = isochronesData.map((data) => {
			const r = reachableKilometers(data);
			let c = circle([data.lng, data.lat], r, {
				// steps: 64,
				units: 'kilometers'
			});
			c.bbox = bbox(c);
			return c;
		});
		lastData = isochronesData;
		lastAllTime = maxAllTime;
		lastSpeed = kilometersPerSecond;
	});

	function is_visible(circle: CircleType) {
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
	}

	$effect(() => {
		if (!map || !circles) {
			return;
		}
		if (!loaded) {
			map.addSource(name, {
				type: 'canvas',
				canvas: 'isochronesCanvas',
				coordinates: boxCoords
			});
			map.addLayer({
				id: name,
				type: 'raster',
				source: name,
				paint: {
					'raster-opacity': opacity / 1000
				}
			});
			loaded = true;
		}

		map.setLayoutProperty(name, 'visibility', active ? 'visible' : 'none');
		if (!active) {
			return;
		}
		map.setPaintProperty(name, 'raster-opacity', opacity / 1000);

		const dimensions = map._containerDimensions();
		const source = map.getSource(name) as CanvasSource;
		source.setCoordinates(boxCoords);

		const canvas = source.canvas;
		canvas.width = dimensions[0];
		canvas.height = dimensions[1];

		const ctx = canvas.getContext('2d');
		if (!ctx) {
			return;
		}
		ctx.fillStyle = color;
		ctx.clearRect(0, 0, dimensions[0], dimensions[1]);

		circles.filter(is_visible).forEach((c) => {
			ctx.save(); // Store canvas state

			const b = c.bbox!; // Existence checked in filter()
			const min = transform([b[0], b[1]], dimensions);
			const max = transform([b[2], b[3]], dimensions);
			const diff_x = max[0] - min[0];
			const diff_y = max[1] - min[1];

			if (diff_x < 2 && diff_y < 2) {
				// Draw small rect
				ctx.fillRect(min[0], min[1], diff_x + 1, diff_y + 1);
			} else {
				// Clip circle
				ctx.beginPath();
				const coords = c.geometry.coordinates[0];
				const start = transform(coords[0], dimensions);
				ctx.moveTo(start[0], start[1]);
				for (let i = 0; i < coords.length; ++i) {
					const pos = transform(coords[i], dimensions);
					ctx.lineTo(pos[0], pos[1]);
				}
				ctx.clip();

				// Fill map, clipped to circle
				ctx.fillRect(0, 0, dimensions[0], dimensions[1]);
			}

			// Restore previous state on top
			ctx.restore();
		});
	});
</script>

<canvas id="isochronesCanvas">Canvas not supported</canvas>
