<script lang="ts">
	import Layer from '$lib/map/Layer.svelte';
	import GeoJSON from '$lib/map/GeoJSON.svelte';
	import type { Itinerary, Leg } from '@motis-project/motis-client';
	import { onClickStop } from '../../utils';
	import { getColor } from '../../modeStyle';

	let {
		itinerary = $bindable(),
		theme
	}: {
		itinerary: Itinerary;
		theme: 'dark' | 'light';
	} = $props();

	function stopsToGeoJSON(legs: Leg[]): GeoJSON.GeoJSON {
		return {
			type: 'FeatureCollection',
			//@ts-expect-error: type is safe
			features: legs
				.filter((l) => {
					return l.mode !== 'WALK' && l.mode !== 'BIKE' && l.mode !== 'CAR';
				})
				.flatMap((l) => {
					const stops = [
						{
							type: 'Feature',
							geometry: { type: 'Point', coordinates: [l.from.lon, l.from.lat] },
							properties: {
								stopId: l.from.stopId,
								name: l.from.name,
								time: l.from.arrival ?? l.from.departure,
								color: getColor(l)[0]
							}
						},
						{
							type: 'Feature',
							geometry: { type: 'Point', coordinates: [l.to.lon, l.to.lat] },
							properties: {
								stopId: l.to.stopId,
								name: l.to.name,
								time: l.to.arrival,
								color: getColor(l)[0]
							}
						}
					];
					const intermediateStops = l.intermediateStops
						? l.intermediateStops.map((s) => ({
								type: 'Feature',
								geometry: {
									type: 'Point',
									coordinates: [s.lon, s.lat]
								},
								properties: {
									stopId: s.stopId,
									name: s.name,
									time: s.arrival,
									color: getColor(l)[0]
								}
							}))
						: [];

					return [...stops, ...intermediateStops];
				})
		};
	}
	const geojson = $derived(stopsToGeoJSON(itinerary.legs));
</script>

<GeoJSON id="stops" data={geojson}>
	<Layer
		id="stops"
		type="circle"
		layout={{}}
		filter={['all']}
		paint={{
			'circle-radius': 5,
			'circle-color': 'white',
			'circle-stroke-width': 4,
			'circle-stroke-color': ['get', 'color']
		}}
		onclick={(e) => {
			const s = e.features?.[0];
			if (!s?.properties?.stopId) {
				return;
			}
			console.log('Clicked Stop:', s.properties.name);
			onClickStop(s.properties.name, s.properties.stopId, new Date(s.properties.time));
		}}
		onmousemove={(_, map) => (map.getCanvas().style.cursor = 'pointer')}
		onmouseleave={(_, map) => (map.getCanvas().style.cursor = '')}
	/>
	<Layer
		id="intermediate-stops-name"
		type="symbol"
		layout={{
			'text-field': ['get', 'name'],
			'text-font': ['Noto Sans Display Regular'],
			'text-size': 12,
			'text-offset': [0, 1],
			'text-anchor': 'top'
		}}
		filter={true}
		paint={{
			'text-halo-width': 2,
			// Use a darker hue of the line color for dark theme and a lighter hue for light theme
			'text-halo-color': [
				'interpolate-hcl',
				['linear'],
				0.6,
				0,
				['get', 'color'],
				1,
				theme == 'dark' ? '#000' : '#fff'
			],
			'text-color': theme == 'dark' ? '#fff' : '#000'
		}}
	/>
</GeoJSON>
