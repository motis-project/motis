<script lang="ts">
	import Layer from '$lib/map/Layer.svelte';
	import GeoJSON from '$lib/map/GeoJSON.svelte';
	import type { Itinerary, Mode } from '@motis-project/motis-client';
	import { getColor } from '$lib/modeStyle';
	import polyline from '@mapbox/polyline';
	import { colord } from 'colord';
	import { layers } from './itineraryLayers';
	import type { FilterSpecification } from 'maplibre-gl';
	import {
		_isLowerLevelRoutingFilter,
		_isUpperLevelRoutingFilter,
		_isCurrentLevelRoutingFilter,
		_leadsToLowerLevelRoutingFilter,
		_leadsUpToCurrentLevelRoutingFilter,
		_leadsDownToCurrentLevelRoutingFilter,
		_leadsToUpperLevelRoutingFilter,
		_connectsToCurrentLevelRoutingFilter,
		_isCurrentLevelFilter,
		_ceilFromLevel,
		_ceilToLevel,
		_floorFromLevel,
		_floorToLevel
	} from './layerFilters';
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
					const outlineColor = colord(color).darken(0.2).toHex();
					return l.steps.map((p) => {
						return {
							type: 'Feature',
							properties: {
								color,
								outlineColor,
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
					const outlineColor = colord(color).darken(0.2).toHex();
					return {
						type: 'Feature',
						properties: {
							outlineColor,
							color
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
				filter={layer.filter as FilterSpecification}
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
			'line-opacity': 0.8
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
			'line-opacity': 0.8
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
				filter={layer.filter as FilterSpecification}
				paint={layer.paint}
			></Layer>
		{/if}
	{/each}
</GeoJSON>
