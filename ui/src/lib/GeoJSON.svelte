<script lang="ts">
	import { getContext, onDestroy, setContext, type Snippet } from 'svelte';

	class Props {
		id!: string;
		data!: GeoJSON.GeoJSON;
		children!: Snippet<[any]>;
	}

	let { id, data, children }: Props = $props();

	let ctx: { map: maplibregl.Map | null } = getContext('map');

	let sourceId = $state<{ id: null | string }>({ id: null });
	setContext('source', sourceId);

	let initialized = false;

	$effect(() => {
		console.log('RUN EFFECT GEOJSON');
		if (ctx.map && id && data) {
			if (initialized) {
				ctx.map.getSource(id)?.setData(data);
			} else {
				ctx.map.addSource(id, {
					type: 'geojson',
					data
				});
				sourceId.id = id;
				initialized = true;
			}
		}
	});

	onDestroy(() => {
		if (initialized) {
			ctx.map?.removeSource(id);
		}
	});
</script>

{@render children()}
