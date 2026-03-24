<script lang="ts">
	import { t } from '$lib/i18n/translation';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import { MapboxOverlay } from '@deck.gl/mapbox';
	import { IconLayer } from '@deck.gl/layers';
	import maplibregl, { type LngLatLike } from 'maplibre-gl';
	import { onMount, untrack } from 'svelte';
	import { stops, type Mode } from '@motis-project/motis-client';
	import { type PickingInfo } from '@deck.gl/core';
	import { onClickStop } from '$lib/utils';
	import { updateOverlayLayers } from '$lib/updateOverlay';
	import { createStopIcon } from '../createIcon';
	import { hexToRgb } from '$lib/Color';

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
	const colors = new Uint8Array(STOPS_NUM * 3);
	const positions = new Float64Array(STOPS_NUM * 2);
	const stopsData = {
		length: STOPS_NUM,
		positions,
		colors
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
			mask: true
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
			beforeId: 'road-name-text',
			data: {
				length: stopsData.length,
				attributes: {
					getPosition: { value: positions, size: 2 },
					getColor: { value: colors, size: 3, normalized: true }
				}
			},
			// @ts-expect-error: canvas element seems to work fine
			iconAtlas: StopIcon,
			iconMapping: IconMapping,
			getSize: 15,
			pickable: true,
			colorFormat: 'RGB',
			visible: stopsMode !== 'none',
			useDevicePixels: false,
			autoHighlight: true,
			parameters: { depthTest: false },
			getIcon: (_) => 'marker',
			onHover,
			onClick
		});
	};

	const getModeColor = (m: Mode): [number, number, number, number] => {
		switch (m) {
			case 'BUS':
				return hexToRgb('#ff9800');
			case 'COACH':
				return hexToRgb('#9ccc65');
			case 'TRAM':
				return hexToRgb('#ebe717');
			case 'SUBURBAN':
				return hexToRgb('#4caf50');
			case 'SUBWAY':
				return hexToRgb('#3f51b5');
			case 'HIGHSPEED_RAIL':
				return hexToRgb('#9c27b0');
			case 'LONG_DISTANCE':
				return hexToRgb('#e91e63');
			case 'REGIONAL_FAST_RAIL':
			case 'REGIONAL_RAIL':
			case 'RAIL':
				return hexToRgb('#f44336');
			case 'ODM':
				return hexToRgb('#fdb813');
			default:
				return hexToRgb('#000000');
		}
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
				const mode = data[i].modes ? data[i].modes![0] : undefined;
				const color = mode ? getModeColor(mode) : hexToRgb('#000000');
				positions[2 * index] = data[i].lon;
				positions[2 * index + 1] = data[i].lat;
				colors[3 * index] = color[0];
				colors[3 * index + 1] = color[1];
				colors[3 * index + 2] = color[2];
				if (color[0] == 0 && color[1] == 0 && color[2] == 0) {
					console.log(data[i].modes);
				}
				index++;
			}
			stopsData.length = index;
			updateOverlayLayers(createLayer(), layers, overlay);
		});
	});
</script>
