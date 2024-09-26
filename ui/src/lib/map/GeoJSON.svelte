<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import GeoJSON from 'geojson';
	import { getContext, onDestroy, setContext, type Snippet } from 'svelte';

	class Props {
		id!: string;
		data!: GeoJSON.GeoJSON;
		children!: Snippet;
	}

	let { id, data, children }: Props = $props();

	let ctx: { map: maplibregl.Map | null } = getContext('map');

	let sourceId = $state<{ id: null | string }>({ id: null });
	setContext('source', sourceId);

	const updateSource = () => {
		if (!ctx.map || data == null) {
			return;
		}
		const src = ctx.map!.getSource(id);
		if (src) {
			console.log('UPDATE DATA', id);

			// @ts-expect-error: setData exists and does what it should
			src.setData(data);
		} else {
			console.log('ADD SOURCE', id);
			ctx.map!.addSource(id, {
				type: 'geojson',
				data
			});
		}
		sourceId.id = id;
	};

	let initialized = false;
	$effect(() => {
		if (ctx.map && id && data) {
			updateSource();
			if (!initialized) {
				ctx.map!.on('styledata', updateSource);
				sourceId.id = id;
				initialized = true;
			}
		}
	});

	onDestroy(() => {
		if (initialized) {
			ctx.map?.off('styledata', updateSource);
			sourceId.id = null;
			const src = ctx.map!.getSource(id);
			if (src) {
				console.log('DESTROY SOURCE', id);
				ctx.map?.removeSource(id);
			} else {
				console.log('DESTROY SOURCE --- NO SOURCE!', id);
			}
		}
	});
</script>

{@render children()}
