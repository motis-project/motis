<script lang="ts">
	import { lngLatToStr } from '$lib/lngLatToStr';
	import { MapboxOverlay } from '@deck.gl/mapbox';
	import { IconLayer } from '@deck.gl/layers';
	import { createTripIcon } from '$lib/map/createIcon';
	import maplibregl from 'maplibre-gl';
	import { onDestroy, onMount, untrack } from 'svelte';
	import { formatTime } from './toDateTime';
	import { onClickTrip } from './utils';
	import { getDelayColor, rgbToHex } from './Color';
	import type { MetaData } from './types';
	import Control from './map/Control.svelte';
	import { client } from '@motis-project/motis-client';
	import type { PickingInfo } from '@deck.gl/core';
	let {
		map,
		overlay,
		layers,
		bounds,
		zoom,
		colorMode
	}: {
		map: maplibregl.Map | undefined;
		overlay: MapboxOverlay;
		layers: IconLayer[];
		bounds: maplibregl.LngLatBoundsLike | undefined;
		zoom: number;
		colorMode: 'rt' | 'route' | 'mode' | 'none';
	} = $props();

	//QUERY
	let startTime = $state(new Date(Date.now()));
	let endTime = $derived(new Date(startTime.getTime() + 180000));
	let canceled = $derived(colorMode === 'none');
	let query = $derived.by(() => {
		if (!bounds || !zoom) return null;
		const b = maplibregl.LngLatBounds.convert(bounds);
		const max = lngLatToStr(b.getNorthWest());
		const min = lngLatToStr(b.getSouthEast());
		return {
			min,
			max,
			startTime: startTime.toISOString(),
			endTime: endTime.toISOString(),
			zoom
		};
	});

	//TRANSFERABLES
	let isProcessing = false;
	const TRIPS_NUM = 12000;
	const positions = new Float64Array(TRIPS_NUM * 2);
	const angles = new Float32Array(TRIPS_NUM);
	const colors = new Uint8Array(TRIPS_NUM * 3);
	const DATA = {
		length: TRIPS_NUM,
		positions,
		colors,
		angles
	};

	//INTERACTION
	const popup = new maplibregl.Popup({
		closeButton: false,
		closeOnClick: false,
		maxWidth: 'none'
	});

	let activeHoverIndex: number | null = $state(null);

	const onHover = (info: PickingInfo) => {
		activeHoverIndex = info.index;
		if (info.picked && info.index !== -1 && metadata) {
			createPopup(metadata, info.coordinate as maplibregl.LngLatLike);
		} else {
			metadata = undefined;
			popup.remove();
		}
	};
	const onClick = (info: PickingInfo) => {
		if (info.picked && info.index !== -1 && metadata) {
			onClickTrip(metadata.id);
		}
	};
	const createPopup = (trip: MetaData, hoverCoordinate: maplibregl.LngLatLike) => {
		if (!trip || !map) return;

		map.getCanvas().style.cursor = 'pointer';
		const content = trip.realtime
			? `<strong>${trip.displayName}</strong><br>
			   <span style="color: ${rgbToHex(getDelayColor(trip.departureDelay, true))}">${formatTime(new Date(trip.departure), trip.tz)}</span>
			   <span ${trip.departureDelay != 0 ? 'class="line-through"' : ''}>${formatTime(new Date(trip.scheduledDeparture), trip.tz)}</span> ${trip.from}<br>
			   <span style="color: ${rgbToHex(getDelayColor(trip.arrivalDelay, true))}">${formatTime(new Date(trip.arrival), trip.tz)}</span>
			   <span ${trip.arrivalDelay != 0 ? 'class="line-through"' : ''}>${formatTime(new Date(trip.scheduledArrival), trip.tz)}</span> ${trip.to}`
			: `<strong>${trip.displayName}</strong><br>
			   ${formatTime(new Date(trip.departure), trip.tz)} ${trip.from}<br>
			   ${formatTime(new Date(trip.arrival), trip.tz)} ${trip.to}`;

		popup.setLngLat(hoverCoordinate).setHTML(content).addTo(map);
	};

	//ANIMATION
	const TripIcon = createTripIcon(128);
	const IconMapping = {
		marker: {
			x: 0,
			y: 0,
			width: 128,
			height: 128,
			anchorY: 64,
			anchorX: 64,
			mask: true
		}
	};
	const createLayer = () => {
		if (!DATA.positions || DATA.positions.byteLength === 0) return;
		return new IconLayer({
			id: 'trips-layer',
			data: {
				length: DATA.length,
				attributes: {
					getPosition: { value: DATA.positions, size: 2 },
					getAngle: { value: DATA.angles, size: 1 },
					getColor: { value: DATA.colors, size: 3, normalized: true }
				}
			},
			beforeId: 'road-name-text',
			// @ts-expect-error: canvas element seems to work fine
			iconAtlas: TripIcon,
			iconMapping: IconMapping,
			pickable: colorMode != 'none',
			sizeScale: 5,
			getSize: 10,
			getIcon: (_) => 'marker',
			colorFormat: 'RGB',
			visible: colorMode !== 'none',
			useDevicePixels: false,
			parameters: { depthTest: false },
			onClick,
			onHover
		});
	};
	let animationId: number;
	const animate = () => {
		if (!DATA.positions || DATA.positions.length === 0) return;
		worker.postMessage(
			{
				type: 'update',
				colorMode,
				positions: DATA.positions,
				index: activeHoverIndex,
				angles: DATA.angles,
				colors: DATA.colors,
				length: DATA.length
			},
			[DATA.positions.buffer, DATA.angles.buffer, DATA.colors.buffer]
		);
	};

	// UPDATE
	const updateOverlayLayers = (l: IconLayer) => {
		layers[0] = l;
		overlay.setProps({ layers: [...layers] });
	};
	$effect(() => {
		if (!query || isProcessing || canceled) return;
		untrack(() => {
			isProcessing = true;
			worker.postMessage({ type: 'fetch', query });
		});
	});

	setInterval(() => {
		if (query && colorMode !== 'none') {
			startTime = new Date();
		}
	}, 60000);

	//SETUP
	let status = $state();
	let worker: Worker;
	let metadata: MetaData | undefined = $state();

	onMount(() => {
		worker = new Worker(new URL('tripsWorker.ts', import.meta.url), { type: 'module' });
		worker.postMessage({ type: 'init', baseUrl: client.getConfig().baseUrl });
		worker.onmessage = (e) => {
			if (e.data.type == 'fetch-complete') {
				status = e.data.status;
				isProcessing = false;
			} else {
				const { positions, angles, length, colors } = e.data;
				DATA.positions = new Float64Array(positions.buffer);
				DATA.angles = new Float32Array(angles.buffer);
				DATA.colors = new Uint8Array(colors.buffer);
				DATA.length = length;
				metadata = e.data.metadata;
			}
			const layer = createLayer();
			if (layer) {
				updateOverlayLayers(layer);
			}
			if (canceled) {
				cancelAnimationFrame(animationId);
			} else {
				animationId = requestAnimationFrame(animate);
			}
		};
	});
	onDestroy(() => {
		if (animationId) cancelAnimationFrame(animationId);
		worker.terminate();
		popup.remove();
	});
</script>

{#if status && status !== 200}
	<Control position="bottom-left">trips response status: {status}</Control>
{/if}
