<script lang="ts">
	import { posToLocation, type Location } from '$lib/Location';
	import maplibregl from 'maplibre-gl';
	import { getContext, onDestroy } from 'svelte';

	let ctx: { map: maplibregl.Map | null } = getContext('map'); // from Map component

	let {
		color,
		draggable,
		location = $bindable()
	}: {
		color: string;
		draggable: boolean;
		location: Location;
	} = $props();

	let initialized = false;
	let marker: maplibregl.Marker | undefined;

	$effect(() => {
		if (ctx.map && location.value.match) {
			if (!initialized) {
				marker = new maplibregl.Marker({
					draggable,
					color
				})
					.setLngLat(location.value.match)
					.addTo(ctx.map)
					.on('dragend', () => {
						if (marker && location.value.match) {
							let x = posToLocation(marker.getLngLat());
							location.value = x.value;
							location.label = x.label;
						}
					});
				initialized = true;
			}
		}
	});

	$effect(() => {
		if (marker && location.value && location.value.match) {
			marker.setLngLat(location.value.match);
		}
	});

	onDestroy(() => {
		if (marker) {
			marker.remove();
			marker = undefined;
		}
	});
</script>
