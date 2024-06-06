<script lang="ts">
	import { onDestroy, getContext } from 'svelte';

	let {
		id,
		type,
		filter,
		layout,
		paint
	}: { id: string; type: string; filter: Object; layout: Object; paint: Object } = $props();

	let ctx: { map: maplibregl.Map | null } = getContext('map');
	let source = getContext('source');

	let initialized = false;
	let currFilter = filter;
	let currLayout = layout;
	let currPaint = paint;

	let updateLayer = () => {
		const l = ctx.map?.getLayer(id);
		if (!source.id) {
			if (l) {
				ctx.map?.removeLayer(id);
			}
			return;
		}

		if (l && filter == currFilter && layout == currLayout && paint == currPaint) {
			return;
		}

		if (!l) {
			console.log('ADD LAYER', id);
			ctx.map.addLayer({
				source: source.id,
				id,
				type,
				filter,
				layout,
				paint
			});
			currFilter = filter;
			currLayout = layout;
			currPaint = paint;
			return;
		}

		if (currFilter != filter) {
			console.log('UPDATE FILTER', id);
			ctx.map!.setFilter(id, filter);
			currFilter = filter;
		}
	};

	$effect(() => {
		if (ctx.map && source.id) {
			if (initialized) {
			} else {
				ctx.map.on('styledata', updateLayer);
				updateLayer();
				initialized = true;
			}
		}
	});

	onDestroy(() => {
		const l = ctx.map?.getLayer(id);
		if (l) {
			ctx.map?.removeLayer(id);
		}
	});
</script>
