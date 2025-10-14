<script lang="ts">
	import { onMount } from 'svelte';
	import maplibregl from 'maplibre-gl';
	import type { CanvasSource, GeoJSONSource, LngLatBoundsLike, Map } from 'maplibre-gl';
	import type { GeoJSON } from 'geojson';
	import type { PrePostDirectMode } from '$lib/Modes';
	import {
		isCanvasLevel,
		isLess,
		minDisplayLevel,
		type DisplayLevel,
		type IsochronesOptions,
		type IsochronesPos
	} from '$lib/map/IsochronesShared';
	import type { WorkerMessage } from '$lib/map/IsochronesWorker';
	import WebWorker from '$lib/map/IsochronesWorker.ts?worker';

	type BoxCoordsType = [[number, number], [number, number], [number, number], [number, number]];

	let {
		map,
		bounds,
		isochronesData,
		streetModes,
		wheelchair,
		maxAllTime,
		circleResolution,
		active,
		options = $bindable()
	}: {
		map: Map | undefined;
		bounds: LngLatBoundsLike | undefined;
		isochronesData: IsochronesPos[];
		streetModes: PrePostDirectMode[];
		wheelchair: boolean;
		maxAllTime: number;
		circleResolution: number | undefined;
		active: boolean;
		options: IsochronesOptions;
	} = $props();

	const emptyGeometry: GeoJSON = { type: 'LineString', coordinates: [] };
	// Must all exist
	let objects = $state<
		| {
				worker: Worker;
				canvasLayer: 'isochrones-canvas';
				circlesLayer: 'isochrones-circles';
				canvasSource: CanvasSource;
				circlesSource: GeoJSONSource;
		  }
		| undefined
	>(undefined);
	let bestAvailableDisplayLevel = $state<DisplayLevel>('NONE');

	const kilometersPerSecond = $derived(
		// Should match the speed used for routing
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

	let lastData: IsochronesPos[] = [];
	let lastMaxAllTime: number = 0;
	let lastSpeed: number = 0;
	let dataIndex = 0;

	onMount(() => {
		lastMaxAllTime = maxAllTime;
		lastSpeed = kilometersPerSecond;
	});

	// Setup objects
	$effect(() => {
		if (!map || !active || objects !== undefined) {
			return;
		}

		// Create sources, layers and canvases
		const canvasLayer = 'isochrones-canvas';
		const circlesLayer = 'isochrones-circles';

		let canvas = document.createElement('canvas');
		if (canvas === undefined) {
			console.log('Canvas not supported');
			return;
		}
		let offscreenCanvas = canvas.transferControlToOffscreen();

		map.addSource(canvasLayer, {
			type: 'canvas',
			canvas,
			coordinates: boxCoords
		});
		map.addLayer({
			id: canvasLayer,
			type: 'raster',
			source: canvasLayer,
			paint: {
				'raster-opacity': options.opacity / 1000
			}
		});
		const canvasSource = map.getSource(canvasLayer) as CanvasSource;

		map.addSource(circlesLayer, {
			type: 'geojson',
			data: emptyGeometry
		});
		map.addLayer({
			id: circlesLayer,
			type: 'fill',
			source: circlesLayer,
			paint: {
				'fill-color': options.color,
				'fill-opacity': options.opacity / 1000
			}
		});
		const circlesSource = map.getSource(circlesLayer) as GeoJSONSource;

		// Setup worker
		const worker = new WebWorker();

		worker.onmessage = (event: { data: WorkerMessage }) => {
			const method = event.data.method;
			switch (method) {
				case 'update-display-level':
					{
						const index = event.data.index;
						if (index < dataIndex) {
							console.log(`Got stale index from worker (Got ${index}, expected ${dataIndex})`);
							return;
						}
						const level: DisplayLevel = event.data.level;
						if (level == 'GEOMETRY_CIRCLES' && objects) {
							objects.circlesSource.setData(event.data.geometry ?? emptyGeometry);
						}
						if (isLess(bestAvailableDisplayLevel, level)) {
							bestAvailableDisplayLevel = level;
						}
					}
					break;
				default:
					console.log(`Unknown method '${method}'`);
			}
		};

		worker.postMessage(
			{
				method: 'set-canvas',
				canvas: offscreenCanvas
			},
			[offscreenCanvas]
		);

		// Store references
		objects = {
			worker,
			canvasLayer,
			circlesLayer,
			canvasSource,
			circlesSource
		};
	});

	$effect(() => {
		if (!active || options.status == 'FAILED' || objects === undefined) {
			return;
		}

		// isochronesData and lastData might both be empty, but have different references
		if (
			((lastData.length != 0 || isochronesData.length != 0) && lastData != isochronesData) ||
			lastMaxAllTime != maxAllTime ||
			lastSpeed != kilometersPerSecond
		) {
			objects.worker.postMessage({
				method: 'update-data',
				index: ++dataIndex,
				data: $state.snapshot(isochronesData),
				kilometersPerSecond: $state.snapshot(kilometersPerSecond),
				maxSeconds: $state.snapshot(maxAllTime),
				circleResolution
			});

			lastData = isochronesData;
			lastMaxAllTime = maxAllTime;
			lastSpeed = kilometersPerSecond;

			bestAvailableDisplayLevel = 'NONE';
			objects.circlesSource.setData(emptyGeometry);
		}

		objects.worker.postMessage({
			method: 'set-max-display-level',
			displayLevel: options.displayLevel
		});
	});

	$effect(() => {
		if (!map || objects === undefined) {
			return;
		}
		map.setLayoutProperty(
			objects.canvasLayer,
			'visibility',
			active && isCanvasLevel(currentDisplayLevel) ? 'visible' : 'none'
		);
		map.setLayoutProperty(
			objects.circlesLayer,
			'visibility',
			active && currentDisplayLevel == 'GEOMETRY_CIRCLES' ? 'visible' : 'none'
		);
	});

	$effect(() => {
		if (!map || objects === undefined) {
			return;
		}
		map.setPaintProperty(objects.canvasLayer, 'raster-opacity', options.opacity / 1000);
		map.setPaintProperty(objects.circlesLayer, 'fill-opacity', options.opacity / 1000);
	});

	$effect(() => {
		if (!map || objects === undefined) {
			return;
		}
		map.setPaintProperty(objects.circlesLayer, 'fill-color', options.color);
	});

	let currentDisplayLevel = $derived.by<DisplayLevel>(() => {
		if (!map || !active || objects === undefined) {
			return 'NONE';
		}

		const nextLevel = minDisplayLevel(options.displayLevel, bestAvailableDisplayLevel);

		if (isCanvasLevel(nextLevel)) {
			objects.canvasSource.setCoordinates(boxCoords);

			const dimensions = map._containerDimensions();

			objects.worker.postMessage({
				method: 'render-canvas',
				level: nextLevel,
				boundingBox: $state.snapshot(boundingBox),
				dimensions,
				color: options.color
			});
		}

		return nextLevel;
	});
	$effect(() => {
		options.status =
			isochronesData.length == 0
				? 'EMPTY'
				: currentDisplayLevel == 'NONE' || currentDisplayLevel == options.displayLevel
					? 'DONE'
					: 'WORKING';
	});
</script>
