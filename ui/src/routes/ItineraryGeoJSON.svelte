<script lang="ts">
	import Layer from '$lib/map/Layer.svelte';
	import GeoJSON from '$lib/map/GeoJSON.svelte';
	import type { Itinerary } from '$lib/openapi';
	import { getColor } from '$lib/modeStyle';
	import polyline from 'polyline';
	import { colord } from 'colord';

	const PRECISION = 7;

	function itineraryToGeoJSON(i: Itinerary): GeoJSON.GeoJSON {
		return {
			type: 'FeatureCollection',
			features: i.legs.flatMap((l) => {
				if (l.legGeometryWithLevels) {
					return l.legGeometryWithLevels.map((p) => {
						return {
							type: 'Feature',
							properties: {
								color: '#42a5f5',
								outlineColor: '#1966a4',
								level: p.from_level,
								way: p.osm_way
							},
							geometry: {
								type: 'LineString',
								coordinates: polyline.decode(p.polyline.points, PRECISION).map(([x, y]) => [y, x])
							}
						};
					});
				} else {
					const color = `${getColor(l)}`;
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
