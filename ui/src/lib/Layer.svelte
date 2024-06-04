<script lang="ts">
	import { onDestroy, getContext } from 'svelte';

	let { id, type, filter, layout, paint } = $props();

	let ctx = getContext('map');
	let source = getContext('source');

	let initialized = false;
	$effect(() => {
		if (ctx.map && source.id) {
			if (initialized) {
			} else {
				ctx.map.addLayer({
					source: source.id,
					id,
					type,
					filter,
					layout,
					paint
				});
				initialized = true;
			}
		}
	});

	onDestroy(() => {
		if (initialized) {
			ctx.map?.removeLayer(id);
		}
	});
</script>
