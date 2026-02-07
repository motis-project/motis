<script lang="ts">
	import { lngLatToStr } from '$lib/lngLatToStr';
	import { MapboxOverlay } from '@deck.gl/mapbox';
	import { IconLayer } from '@deck.gl/layers';
	import maplibregl, { type LngLatLike } from 'maplibre-gl';
	import { onMount, untrack } from 'svelte';
	import { stops } from '@motis-project/motis-client';
	import { type PickingInfo } from '@deck.gl/core';
	import { onClickStop } from '$lib/utils';
	let {
		map,
		overlay,
		layers,
		zoom,
		bounds,
		stopMode
	}: {
		map: maplibregl.Map | undefined;
		overlay: MapboxOverlay;
		layers: IconLayer[];
		zoom: number;
		bounds: maplibregl.LngLatBoundsLike | undefined;
		stopMode: 'all' | 'parent' | 'none';
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
	type MetaData = {
		name: string;
		stopId: string | undefined;
		parentId: string | undefined;
	};
	const metadata: MetaData[] = [];

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
	const ICON_SIZE = 50;
	const StopIcon = createStopIcon(ICON_SIZE);

	const IconMapping = {
		marker: {
			x: 0,
			y: 0,
			width: ICON_SIZE,
			height: ICON_SIZE,
			anchorX: ICON_SIZE / 2,
			anchorY: ICON_SIZE / 2,
			mask: false
		}
	};

	const popup = new maplibregl.Popup({
		closeButton: false,
		closeOnClick: false,
		maxWidth: 'none'
	});

	const onHover = (info: PickingInfo) => {
		if (info.picked && info.index != -1) {
			const data = metadata[info.index];
			const content = `<strong>${data.name}</strong><br>`;
			popup
				.setLngLat(info.coordinate as LngLatLike)
				.setHTML(content)
				.addTo(map!);
		} else {
			popup.remove();
		}
	};

	const onClick = (info: PickingInfo) => {
		if (info.picked && info.index != -1) {
			const data = metadata[info.index];
			onClickStop(data.name, data.stopId!, new Date(Date.now()));
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
			visible: stopMode !== 'none',
			sizeScale: 5,
			getSize: 4,
			pickable: true,
			useDevicePixels: false,
			parameters: { depthTest: false },
			getIcon: (_) => 'marker',
			onHover,
			onClick
		});
	};

	//SETUP
	onMount(() => {
		updateOverlayLayers(createLayer());
	});

	//UPDATE
	const zoomToImportance = $derived(Math.pow(10, 6 - 0.5 * zoom));
	const updateOverlayLayers = (l: IconLayer) => {
		layers[1] = l;
		overlay.setProps({ layers: [...layers] });
	};
	$effect(() => {
		if (stopMode) {
			updateOverlayLayers(createLayer());
		}
	});
	$effect(() => {
		if (!query || stopMode == 'none') return;
		untrack(async () => {
			if (zoom >= 12) {
				const { data } = await stops({ query });
				if (!data) return;
				let index = 0;
				for (let i = 0; i < data.length; ++i) {
					if (!data[i].importance || data[i].importance! >= zoomToImportance) {
						if (stopMode == 'parent' && data[i].parentId != data[i].stopId) continue;
						metadata[index] = {
							name: data[i].name,
							stopId: data[i].stopId,
							parentId: data[i].parentId
						};
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
