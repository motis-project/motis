<script lang="ts">
	import circle from '@turf/circle';
	import { bbox } from '@turf/bbox';
	import maplibregl, {
		CanvasSource,
		type LngLatBoundsLike,
		Map,
	} from 'maplibre-gl';

	interface Pos {
		lat: number;
		lng: number;
		seconds: number;
	}

	type boxCoordsType = [[number, number], [number, number], [number, number], [number, number]];
	type CircleType = ReturnType<typeof circle>;

	let {
		map = $bindable(),
		bounds = $bindable(),
		isochronesData,
		active = $bindable(),
		color = $bindable(),
		opacity = $bindable()
	}: {
		map: Map | undefined;
		bounds: LngLatBoundsLike | undefined;
		isochronesData: Pos[];
		active: boolean;
		color: string;
		opacity: number;
	} = $props();

	const name = 'isochrones-data';
	let loaded = false;

	let lastData: Pos[] | undefined = undefined;

	const kilometersPerSecond = 0.001; // 3.6 kilometers per hour
	const box2 = $derived(
		maplibregl.LngLatBounds.convert(
			bounds ?? [
				[0, 0],
				[1, 1]
			]
		)
	);
	const box_coords: boxCoordsType = $derived([
		[box2._sw.lng, box2._ne.lat],
		[box2._ne.lng, box2._ne.lat],
		[box2._ne.lng, box2._sw.lat],
		[box2._sw.lng, box2._sw.lat]
	]);
	function reachable_kilometers(pos: Pos) {
		return pos.seconds * kilometersPerSecond;
	}
	function transform(pos: number[], dimensions: number[]) {
		const x = Math.round(((pos[0] - box2._sw.lng) / (box2._ne.lng - box2._sw.lng)) * dimensions[0]);
		const y = Math.round(((box2._ne.lat - pos[1]) / (box2._ne.lat - box2._sw.lat)) * dimensions[1]);
		return [x, y];
	}

	let circles3 = $state<CircleType[] | undefined>(undefined);
	$effect(() => {
		if (!active || lastData == isochronesData) {
			return;
		}
		circles3 = isochronesData.map((data) => {
			const r = reachable_kilometers(data);
			let c = circle([data.lng, data.lat], r, {
				// steps: 64,
				units: 'kilometers'
			});
			c.bbox = bbox(c);
			return c;
		});
		lastData = isochronesData;
	});

	function is_visible(circle: CircleType) {
		if (!circle.bbox) {
			return false;
		}
		const b = circle.bbox; // [minX, minY, maxX, maxY]
		return (
			box2._sw.lat <= b[3] && b[1] <= box2._ne.lat && box2._sw.lng <= b[2] && b[0] <= box2._ne.lat
		);
	}

	$effect(() => {
		if (!map || !circles3) {
			return;
		}
		if (!loaded) {
			map.addSource(name, {
				type: 'canvas',
				canvas: 'isochronesCanvas',
				coordinates: box_coords
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
		source.setCoordinates(box_coords);

		const canvas = source.canvas;
		canvas.width = dimensions[0];
		canvas.height = dimensions[1];

		const ctx = canvas.getContext('2d');
		if (!ctx) {
			return;
		}
		ctx.fillStyle = color;
		ctx.clearRect(0, 0, dimensions[0], dimensions[1]);

		circles3.filter(is_visible).forEach((c) => {
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