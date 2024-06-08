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

	let currData = null;
	const updateSource = () => {
		if (!ctx.map || data == null) {
			sourceId.id = null;
			return;
		}
		const src = ctx.map.getSource(id);
		const d = $state.snapshot(data);
		if (src) {
			if (d !== currData) {
				src.setData(data);
			}
		} else {
			ctx.map.addSource(id, {
				type: 'geojson',
				data
			});
		}
		currData = d;
		sourceId.id = id;
	};

	let initialized = false;
	$effect(() => {
		if (ctx.map && id && data) {
			updateSource();
			if (!initialized) {
				ctx.map.on('styledata', updateSource);
				sourceId.id = id;
				initialized = true;
			}
		}
	});

	onDestroy(() => {
		if (initialized) {
			const src = ctx.map.getSource(id);
			if (src) {
				ctx.map?.removeSource(id);
			}
		}
	});
</script>

{@render children()}
