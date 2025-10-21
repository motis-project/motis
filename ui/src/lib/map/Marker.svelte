<script lang="ts">
	import { posToLocation, type Location } from '$lib/Location';
	import maplibregl from 'maplibre-gl';
	import { getContext, onDestroy } from 'svelte';

	let ctx: { map: maplibregl.Map | null } = getContext('map'); // from Map component

	let {
		color,
		draggable,
		level,
		location = $bindable(),
		marker = $bindable()
	}: {
		color: string;
		draggable: boolean;
		level?: number;
		location: Location;
		marker?: maplibregl.Marker;
	} = $props();

	let initialized = false;

	$effect(() => {
		if (ctx.map && location.match) {
			if (!initialized) {
				marker = new maplibregl.Marker({
					draggable,
					color
				})
					.setLngLat(location.match)
					.addTo(ctx.map);
				marker.on('dragend', () => {
					if (marker && location.match) {
						let x = posToLocation(marker.getLngLat(), level ?? 0);
						location = x;
						location.label = x.label;
					}
				});
				initialized = true;
			}
		}
	});

	$effect(() => {
		if (marker && location && location.match && location.match.lat && location.match.lon) {
			marker.setLngLat(location.match);
		}
	});

	onDestroy(() => {
		if (marker) {
			marker.remove();
			marker = undefined;
		}
	});
</script>
