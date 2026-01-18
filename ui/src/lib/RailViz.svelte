<script lang="ts">
	import { lngLatToStr } from '$lib/lngLatToStr';
	import { MapboxOverlay } from '@deck.gl/mapbox';
	import { IconLayer } from '@deck.gl/layers';
	import { createTripIcon } from '$lib/map/createTripIcon';
	import maplibregl from 'maplibre-gl';
	import { onDestroy, onMount, untrack } from 'svelte';
	import { formatTime } from './toDateTime';
	import { onClickTrip } from './utils';
	import { getDelayColor, rgbToHex } from './Color';
	import type { MetaData } from './types';
	let {
		map,
		bounds,
		zoom,
		colorMode
	}: {
		map: maplibregl.Map | undefined;
		bounds: maplibregl.LngLatBoundsLike | undefined;
		zoom: number;
		colorMode: 'rt' | 'route' | 'mode' | 'none';
	} = $props();

	//QUERY
	let startTime = $state(new Date(Date.now()));
	let endTime = $derived(new Date(startTime.getTime() + 180000));
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
	const TRIPS_NUM = 6500;
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
	type HoverEvent = {
		index?: number;
		coordinate?: number[];
	};

	type ClickEvent = {
		index: number;
	};
	const onHover = ({ index, coordinate }: HoverEvent) => {
		const trip = coordinate && index && index !== -1 ? metadata[index] : null;

		if (trip && map) {
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
			popup
				.setLngLat(coordinate as maplibregl.LngLatLike)
				.setHTML(content)
				.addTo(map);
		} else if (map) {
			map.getCanvas().style.cursor = '';
			popup.remove();
		}
	};
	const onClick = ({ index }: ClickEvent) => {
		if (index !== -1 && metadata[index]) {
			onClickTrip(metadata[index].id);
		}
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
			getSize: 50,
			getIcon: (_) => 'marker',
			colorFormat: 'RGB',
			visible: colorMode !== 'none',
			useDevicePixels: false,
			parameters: { depthTest: false }
		});
	};
	let animationId: number;
	const animate = () => {
		if (DATA.positions.byteLength === 0) return;
		worker.postMessage(
			{
				type: 'update',
				colorMode,
				positions: DATA.positions,
				angles: DATA.angles,
				colors: DATA.colors,
				length: DATA.length
			},
			[DATA.positions.buffer, DATA.angles.buffer, DATA.colors.buffer]
		);
	};

	// UPDATE
	$effect(() => {
		if (!query || isProcessing) return;
		untrack(() => {
			if (colorMode === 'none') return;
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
	let overlay: MapboxOverlay;
	let worker: Worker;
	let metadata: MetaData[];
	onMount(() => {
		const origin = new URL(window.location.href).searchParams.get('motis');
		worker = new Worker(new URL('tripsWorker.ts', import.meta.url), { type: 'module' });
		worker.postMessage({ type: 'init', origin });
		worker.onmessage = (e) => {
			if (e.data.type == 'fetch-complete') {
				metadata = e.data.metadata;
				isProcessing = false;
				animate();
				return;
			}
			const { positions, angles, length, colors } = e.data;
			DATA.positions = new Float64Array(positions.buffer);
			DATA.angles = new Float32Array(angles.buffer);
			DATA.colors = new Uint8Array(colors.buffer);
			DATA.length = length;
			overlay.setProps({ layers: [createLayer()] });
			animationId = requestAnimationFrame(animate);
		};
		overlay = new MapboxOverlay({
			interleaved: true,
			onHover,
			onClick
		});
	});
	/*
	$effect(() => {
		if (colorMode === 'none' || cancel) {
			cancelAnimationFrame(animationId);
		}
	});
	let cancel = $state(false);
	*/
	$effect(() => {
		if (!map || !overlay) return;
		map.addControl(overlay);
		/*
		untrack(() => {
			map.on('movestart', () => {
				cancel = true;
			});
			map.on('moveend', () => {
				cancel = false;
			});
		});
		 */
	});
	onDestroy(() => {
		if (animationId) cancelAnimationFrame(animationId);
		if (overlay) map?.removeControl(overlay);
		worker.terminate();
		popup.remove();
	});
</script>
