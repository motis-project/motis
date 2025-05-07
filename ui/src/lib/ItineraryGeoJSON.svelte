<script lang="ts">
	import Layer from '$lib/map/Layer.svelte';
	import GeoJSON from '$lib/map/GeoJSON.svelte';
	import type { Itinerary, Mode } from '$lib/openapi';
	import { getColor } from '$lib/modeStyle';
	import polyline from 'polyline';
	import { colord } from 'colord';

	const PRECISION = 6;

	function isIndividualTransport(m: Mode): boolean {
		return m == 'WALK' || m == 'BIKE' || m == 'CAR';
	}

	function itineraryToGeoJSON(i: Itinerary): GeoJSON.GeoJSON {
		return {
			type: 'FeatureCollection',
			features: i.legs.flatMap((l) => {
				if (l.steps) {
					const color = isIndividualTransport(l.mode) ? '#42a5f5' : `${getColor(l)[0]}`;
					const outlineColor = colord(color).darken(0.2).toHex();
					return l.steps.map((p) => {
						return {
							type: 'Feature',
							properties: {
								color,
								outlineColor,
								level: p.fromLevel,
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

	const {
		itinerary,
		level
	}: {
		itinerary: Itinerary;
		level: number;
	} = $props();

	const geojson = $derived(itineraryToGeoJSON(itinerary));
</script>

<GeoJSON id="route" data={geojson}>
	<Layer
		id="path-outline"
		type="line"
		layout={{
			'line-join': 'round',
			'line-cap': 'round'
		}}
		filter={['any', ['!has', 'level'], ['==', 'level', level]]}
		paint={{
			'line-color': ['get', 'outlineColor'],
			'line-width': 10,
			'line-opacity': 0.8
		}}
	/>
	<Layer
		id="path"
		type="line"
		layout={{
			'line-join': 'round',
			'line-cap': 'round'
		}}
		filter={['any', ['!has', 'level'], ['==', 'level', level]]}
		paint={{
			'line-color': ['get', 'color'],
			'line-width': 7.5,
			'line-opacity': 0.8
		}}
	/>
</GeoJSON>
