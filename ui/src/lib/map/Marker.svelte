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
							let x = posToLocation(marker.getLngLat(), level ?? 0);
							location.value = x.value;
							location.label = x.label;
						}
					});
				initialized = true;
			}
		}
	});

	$effect(() => {
		if (
			marker &&
			location.value &&
			location.value.match &&
			location.value.match.lat &&
			location.value.match.lon
		) {
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
