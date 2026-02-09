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
	import Control from '../Control.svelte';
	import { createStopIcon } from '../createIcon';

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
		stopId?: string;
		track?: string;
	};
	const metadata: MetaData[] = [];
	let status = $state();

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
			visible: stopMode !== 'none' && zoom >= 12,
			getSize: 15,
			pickable: true,
			useDevicePixels: false,
			parameters: { depthTest: false },
			getIcon: (_) => 'marker',
			onHover,
			onClick
		});
	};

	//INTERACTION
	const popup = new maplibregl.Popup({
		closeButton: false,
		closeOnClick: true,
		closeOnMove: true,
		maxWidth: 'none'
	});
	const onHover = (info: PickingInfo) => {
		if (info.picked && info.index != -1) {
			const data = metadata[info.index];
			const content = `<strong>${data.name}</strong><br>
			${data.track ? `<strong>${t.track}: ${data.track}</strong><br>` : ''}`;
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

	//SETUP
	onMount(() => {
		updateOverlayLayers(createLayer());
	});

	//UPDATE
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
				const { data, response } = await stops({ query });
				status = response.status;
				if (!data) return;
				let index = 0;
				for (let i = 0; i < data.length; ++i) {
					if (data[i].parentId === data[i].stopId) {
						metadata[index] = {
							name: data[i].name,
							stopId: data[i].stopId,
							track: data[i].track
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

{#if status && status !== 200}
	<Control position="bottom-left">stops response status: {status}</Control>
{/if}
