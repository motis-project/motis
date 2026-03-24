<script lang="ts">
	import { t } from '$lib/i18n/translation';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import { MapboxOverlay } from '@deck.gl/mapbox';
	import { IconLayer } from '@deck.gl/layers';
	import maplibregl, { type LngLatLike } from 'maplibre-gl';
	import { onMount, untrack } from 'svelte';
	import { stops } from '@motis-project/motis-client';
	import { type PickingInfo } from '@deck.gl/core';
	import { onClickStop } from '$lib/utils';
	import { updateOverlayLayers } from '$lib/updateOverlay';
	import { createStopIcon } from '../createIcon';

	let {
		map,
		overlay,
		layers,
		zoom,
		stopsMode,
		bounds
	}: {
		map: maplibregl.Map | undefined;
		overlay: MapboxOverlay;
		layers: IconLayer[];
		zoom: number;
		stopsMode: 'none' | 'all' | 'grouped';
		bounds: maplibregl.LngLatBoundsLike | undefined;
	} = $props();

	//QUERY
	let query = $derived.by(() => {
		if (!bounds) return null;
		const b = maplibregl.LngLatBounds.convert(bounds);
		const max = lngLatToStr(b.getNorthWest());
		const min = lngLatToStr(b.getSouthEast());
		const grouped = stopsMode == 'grouped';
		return {
			min,
			max,
			zoom: zoom < 13 ? 9 : zoom,
			grouped
		};
	});

	//DATA
	const STOPS_NUM = 2048;
	const positions = new Float64Array(STOPS_NUM * 2);
	const stopsData = {
		length: STOPS_NUM,
		positions
	};
	type MetaData = {
		name: string;
		stopId?: string;
		track?: string;
	};
	const metadata: MetaData[] = [];

	//LAYER
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
			const content = `<strong>${data.name}</strong><br>
							${data.track ? `<strong>${t.track}: ${data.track}</strong>` : ''}`;

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
			getSize: 15,
			pickable: true,
			visible: stopsMode !== 'none',
			useDevicePixels: false,
			parameters: { depthTest: false },
			getIcon: (_) => 'marker',
			onHover,
			onClick
		});
	};

	//SETUP
	onMount(() => {
		updateOverlayLayers(createLayer(), layers, overlay);
	});

	//UPDATE
	$effect(() => {
		if (stopsMode) {
			updateOverlayLayers(createLayer(), layers, overlay);
			stopsData.length = 0;
		}
	});
	$effect(() => {
		if (!query || stopsMode == 'none') return;
		untrack(async () => {
			const { data } = await stops({ query });
			if (!data) {
				stopsData.length = 0;
				updateOverlayLayers(createLayer(), layers, overlay);
				return;
			}

			let index = 0;
			for (let i = 0; i < data.length; ++i) {
				metadata[index] = {
					name: data[i].name,
					stopId: data[i].stopId,
					track: data[i].track
				};
				positions[2 * index] = data[i].lon;
				positions[2 * index + 1] = data[i].lat;
				index++;
			}
			stopsData.length = index;
			updateOverlayLayers(createLayer(), layers, overlay);
		});
	});
</script>
