<script lang="ts">
	import maplibregl, { type GeoJSONSourceSpecification } from 'maplibre-gl';
	import GeoJSON from 'geojson';
	import { getContext, onDestroy, setContext, type Snippet } from 'svelte';

	class Props {
		id!: string;
		data!: GeoJSON.GeoJSON;
		lineMetrics?: boolean;
		children!: Snippet;
		options?: Omit<GeoJSONSourceSpecification, 'type' | 'data'>;
	}

	let { id, data, lineMetrics = false, children, options }: Props = $props();

	let ctx: { map: maplibregl.Map | null } = getContext('map');

	let sourceId = $state<{ id: null | string }>({ id: null });
	setContext('source', sourceId);

	const updateSource = () => {
		if (!ctx.map || data == null) {
			return;
		}
		const src = ctx.map!.getSource(id) as maplibregl.GeoJSONSource | undefined;
		if (src) {
			src.setData(data);
		} else {
			ctx.map!.addSource(id, {
				type: 'geojson',
				lineMetrics,
				data,
				...(options ?? {})
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
				ctx.map?.removeSource(id);
			}
		}
	});
</script>

{@render children()}
