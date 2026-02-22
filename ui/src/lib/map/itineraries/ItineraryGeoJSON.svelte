<script lang="ts">
	import Layer from '$lib/map/Layer.svelte';
	import GeoJSON from '$lib/map/GeoJSON.svelte';
	import type { Itinerary, Mode } from '@motis-project/motis-client';
	import { getColor } from '$lib/modeStyle';
	import polyline from '@mapbox/polyline';
	import { getDecorativeColors } from '$lib/map/colors';
	import { layers } from './itineraryLayers';
	export const PRECISION = 6;

	const {
		itinerary,
		id,
		selected,
		selectItinerary,
		level,
		theme
	}: {
		itinerary: Itinerary;
		id?: string;
		selected: boolean;
		selectItinerary?: () => void;
		level: number;
		theme: 'light' | 'dark';
	} = $props();

	function isIndividualTransport(m: Mode): boolean {
		return m == 'WALK' || m == 'BIKE' || m == 'CAR';
	}

	function getIndividualModeColor(m: Mode): string {
		switch (m) {
			case 'CAR':
				return '#bf75ff';
			default:
				return '#42a5f5';
		}
	}

	function itineraryToGeoJSON(i: Itinerary): GeoJSON.GeoJSON {
		return {
			type: 'FeatureCollection',
			features: i.legs.flatMap((l) => {
				if (l.steps) {
					const color = isIndividualTransport(l.mode)
						? getIndividualModeColor(l.mode)
						: `${getColor(l)[0]}`;
					const { outlineColor, chevronColor } = getDecorativeColors(color);
					return l.steps.map((p) => {
						return {
							type: 'Feature',
							properties: {
								color,
								outlineColor,
								chevronColor,
								fromLevel: p.fromLevel,
								toLevel: p.toLevel,
								level: level,
								way: p.osmWay
							},
							geometry: {
								type: 'LineString',
								coordinates: polyline.decode(p.polyline.points, PRECISION).map(([x, y]) => [y, x])
							}
						};
					});
				} else {
					const color = `${getColor(l)[0]}`;
					const { outlineColor, chevronColor } = getDecorativeColors(color);
					return {
						type: 'Feature',
						properties: {
							outlineColor,
							color,
							chevronColor
						},
						geometry: {
							type: 'LineString',
							coordinates: polyline.decode(l.legGeometry.points, PRECISION).map(([x, y]) => [y, x])
						}
					};
				}
			})
		};
	}
	const geojson = $derived(itineraryToGeoJSON(itinerary));
</script>

<GeoJSON id="route-{id}" data={geojson}>
	{#each layers as layer (layer.id)}
		{#if !('line-gradient' in layer.paint) && selected}
			<Layer
				id={layer.id}
				type={layer.type}
				layout={layer.layout}
				filter={['all', ['has', 'fromLevel'], layer.filter]}
				paint={layer.paint}
			></Layer>
		{/if}
	{/each}
	<Layer
		id="path-{id}"
		type="line"
		layout={{
			'line-join': 'round',
			'line-cap': 'round'
		}}
		filter={['!', ['has', 'fromLevel']]}
		onclick={selectItinerary
			? (_) => {
					selectItinerary();
				}
			: undefined}
		paint={{
			'line-color': selected ? ['get', 'color'] : theme == 'dark' ? '#777' : '#bbb',
			'line-width': 7.5,
			'line-opacity': 1
		}}
	/>
	<Layer
		id="path-outline-{id}"
		type="line"
		layout={{
			'line-join': 'round',
			'line-cap': 'round'
		}}
		filter={['!', ['has', 'fromLevel']]}
		paint={{
			'line-color': selected ? ['get', 'outlineColor'] : theme == 'dark' ? '#444' : '#999',
			'line-width': 1.5,
			'line-gap-width': 7.5,
			'line-opacity': 1
		}}
	/>
	<Layer
		id="path-chevrons-{id}"
		type="symbol"
		layout={{
			'symbol-placement': 'line',
			'symbol-spacing': 40,
			'text-field': '›',
			'text-size': 24,
			'text-font': ['Noto Sans Bold'],
			'text-keep-upright': false,
			'text-allow-overlap': true,
			'text-rotation-alignment': 'map',
			'text-offset': [0, -0.1]
		}}
		filter={['!', ['has', 'fromLevel']]}
		paint={{
			'text-color': selected ? ['get', 'chevronColor'] : theme == 'dark' ? '#999' : '#ddd',
			'text-opacity': 0.85,
			'text-halo-color': selected ? ['get', 'outlineColor'] : theme == 'dark' ? '#444' : '#999',
			'text-halo-width': 0.5,
			'text-halo-blur': 0.2
		}}
	/>
</GeoJSON>
<GeoJSON id="route-{id}-metrics" data={geojson} lineMetrics={true}>
	{#each layers as layer (layer.id)}
		{#if 'line-gradient' in layer.paint && selected}
			<Layer
				id="{layer.id}-metrics"
				type={layer.type}
				layout={layer.layout}
				filter={layer.filter}
				paint={layer.paint}
			></Layer>
		{/if}
	{/each}
</GeoJSON>
