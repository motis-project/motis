<script lang="ts">
	import { lngLatToStr } from '$lib/lngLatToStr';
	import { MapboxOverlay } from '@deck.gl/mapbox';
	import { IconLayer } from '@deck.gl/layers';
	import maplibregl from 'maplibre-gl';
	import { onMount, untrack } from 'svelte';
	import { stops } from '@motis-project/motis-client';
	let {
		overlay,
		layers,
		zoom,
		bounds
	}: {
		overlay: MapboxOverlay;
		layers: IconLayer[];
		zoom: number;
		bounds: maplibregl.LngLatBoundsLike | undefined;
	} = $props();

	//QUERY
	let query = $derived.by(() => {
		if (!bounds) return null;
		const b = maplibregl.LngLatBounds.convert(bounds);
		const max = lngLatToStr(b.getNorthWest());
		const min = lngLatToStr(b.getSouthEast());
		return {
			min,
			max
		};
	});

	//DATA
	const STOPS_NUM = 2000;
	const positions = new Float64Array(STOPS_NUM * 2);
	const stopsData = {
		length: STOPS_NUM,
		positions
	};

	//LAYER
	const createStopIcon = (size: number) => {
		const canvas = document.createElement('canvas');
		canvas.width = size;
		canvas.height = size;
		const ctx = canvas.getContext('2d')!;
		const center = size / 2;
		const radius = size * 0.4;
		const border = (2 / 64) * size;

		// Main circle with semi-transparent red fill
		ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
		ctx.beginPath();
		ctx.arc(center, center, radius, 0, Math.PI * 2);
		ctx.fill();

		// Gray border
		ctx.strokeStyle = 'rgba(120, 120, 120, 1.0)';
		ctx.lineWidth = border;
		ctx.stroke();

		// White ring inside
		ctx.strokeStyle = '#ffffff';
		ctx.lineWidth = size * 0.04;
		ctx.beginPath();
		ctx.arc(center, center, radius * 0.65, 0, Math.PI * 2);
		ctx.stroke();

		// White dot in center
		ctx.fillStyle = '#ffffff';
		ctx.beginPath();
		ctx.arc(center, center, radius * 0.35, 0, Math.PI * 2);
		ctx.fill();

		return canvas;
	};
	const StopIcon = createStopIcon(50);
	const IconMapping = {
		marker: {
			x: 0,
			y: 0,
			width: 128,
			height: 128,
			anchorY: 64,
			anchorX: 64,
			mask: false
		}
	};
	const createLayer = () => {
		return new IconLayer({
			id: 'stops-view-layer',
			beforeId: 'trips-layer',
			data: {
				length: stopsData.length,
				attributes: {
					getPosition: { value: positions, size: 2 }
				}
			},
			// @ts-expect-error: canvas element seems to work fine
			iconAtlas: StopIcon,
			iconMapping: IconMapping,
			sizeScale: 5,
			getSize: 10,
			getIcon: (_) => 'marker',
			visible: zoom >= 12
		});
	};
	$effect(() => {});
	//SETUP
	onMount(() => {
		updateOverlayLayers(createLayer());
	});

	//UPDATE
	const zoomToImportance = $derived(0.01 / (zoom - 0.2));
	const updateOverlayLayers = (l: IconLayer) => {
		layers[1] = l;
		overlay.setProps({ layers: [...layers] });
	};
	$effect(() => {
		if (!query) return;
		untrack(async () => {
			if (zoom >= 12) {
				const { data } = await stops({ query });
				if (!data) return;
				let index = 0;
				for (let i = 0; i < data.length; ++i) {
					console.log(data[i].importance);
					if (!data[i].importance || data[i].importance! >= zoomToImportance) {
						positions[2 * index] = data[i].lon;
						positions[2 * index + 1] = data[i].lat;
						index++;
					}
				}
				stopsData.length = index;
			}
			updateOverlayLayers(createLayer());
		});
	});
</script>
