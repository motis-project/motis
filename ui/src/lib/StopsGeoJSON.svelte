<script lang="ts">
	import Layer from '$lib/map/Layer.svelte';
	import GeoJSON from '$lib/map/GeoJSON.svelte';
	import type { Itinerary, Leg } from './api/openapi';
	import { onClickStop } from './utils';

	let {
		itinerary = $bindable()
	}: {
		itinerary: Itinerary;
	} = $props();

	function stopsToGeoJSON(legs: Leg[]): GeoJSON.GeoJSON {
		return {
			type: 'FeatureCollection',
			//@ts-expect-error: type is safe
			features: legs.flatMap((l) => {
				const stops = [
					{
						
						type: 'Feature',
						geometry: { type: 'Point', coordinates: [l.from.lon, l.from.lat] },
						properties: { stopId: l.from.stopId, name: l.from.name, time: l.from.arrival }
					},
					{
						type: 'Feature',
						geometry: { type: 'Point', coordinates: [l.to.lon, l.to.lat] },
						properties: { stopId: l.to.stopId, name: l.to.name, time: l.to.arrival }
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
								time: s.arrival
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
		paint={{ 'circle-radius': 6, 'circle-color': 'black' }}
		onclick={(e) => {
			if (!e.features || e.features.length === 0) return;
			const s = e.features[0];
			console.log('Clicked Stop:', s.properties.name);
			onClickStop(s.properties.name, s.properties.stopId, new Date(s.properties.time));
		}}
	/>
</GeoJSON>
