<script lang="ts">
	import Layer from '$lib/map/Layer.svelte';
	import GeoJSON from '$lib/map/GeoJSON.svelte';
	import type { Itinerary, Mode } from '@motis-project/motis-client';
	import { getColor } from '$lib/modeStyle';
	import polyline from '@mapbox/polyline';
	import { colord } from 'colord';
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
								level: p.fromLevel,
								way: p.osmWay,
								levelDiff: p.fromLevel - level
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
	const opacityExpression = [
		'case',
		['==', ['abs', ['get', 'levelDiff']], 0],
		1,
		['<=', ['abs', ['get', 'levelDiff']], 1],
		0.6,
		['<=', ['abs', ['get', 'levelDiff']], 2],
		0.3,
		0.1
	];
</script>

<GeoJSON id="route-{id}" data={geojson}>
	<Layer
		id="path-outline-solid-{id}"
		type="line"
		layout={{
			'line-join': 'round',
			'line-cap': 'round'
		}}
		filter={['>=', ['get', 'levelDiff'], 0]}
		paint={{
			'line-color': selected ? ['get', 'outlineColor'] : theme == 'dark' ? '#444' : '#999',
			'line-width': ['case', ['<', ['get', 'levelDiff'], 0], 3, 9],
			'line-opacity': opacityExpression
		}}
	/>
	<Layer
		id="path-outline-dashed-{id}"
		type="line"
		layout={{
			'line-join': 'round',
			'line-cap': 'butt'
		}}
		filter={['<', ['get', 'levelDiff'], 0]}
		paint={{
			'line-color': selected ? ['get', 'outlineColor'] : theme == 'dark' ? '#444' : '#999',
			'line-width': ['case', ['<', ['get', 'levelDiff'], 0], 3, 9],
			'line-opacity': opacityExpression,
			'line-gap-width': ['case', ['<', ['get', 'levelDiff'], 0], 3, 0],
			'line-dasharray': [3, 1]
		}}
	/>
	<Layer
		id="path-{id}"
		type="line"
		layout={{
			'line-join': 'round',
			'line-cap': 'round'
		}}
		filter={['>=', ['get', 'levelDiff'], 0]}
		onclick={selectItinerary
			? (_) => {
					selectItinerary();
				}
			: undefined}
		paint={{
			'line-color': selected ? ['get', 'color'] : theme == 'dark' ? '#777' : '#bbb',
			'line-width': 9,
			'line-opacity': opacityExpression
		}}
	/>
</GeoJSON>
