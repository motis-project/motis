<script lang="ts">
	import { onDestroy, getContext, setContext } from 'svelte';

	let {
		id,
		type,
		filter,
		layout,
		paint
	}: {
		id: string;
		type: string;
		filter: maplibregl.FilterSpecification;
		layout: Object;
		paint: Object;
	} = $props();

	let layer = $state<{ id: null | string }>({ id: null });
	setContext('layer', layer);

	let ctx: { map: maplibregl.Map | null } = getContext('map'); // from Map component
	let source: { id: string | null } = getContext('source'); // from GeoJSON component

	let initialized = false;
	let currFilter = filter;
	let currLayout = layout;
	let currPaint = paint;

	let updateLayer = () => {
		const l = ctx.map?.getLayer(id);
		if (!source.id) {
			if (l) {
				console.log('REMOVE', id);
				layer.id = null;
				ctx.map?.removeLayer(id);
			}
			return;
		}

		if (l && filter == currFilter && layout == currLayout && paint == currPaint) {
			return;
		}

		if (!l) {
			console.log('ADD LAYER', source.id, id, type, filter, layout, paint);
			ctx.map.addLayer({
				source: source.id,
				id,
				type,
				filter,
				layout,
				paint
			});
			currFilter = $state.snapshot(filter);
			currLayout = $state.snapshot(layout);
			currPaint = $state.snapshot(paint);
			layer.id = id;
			return;
		}

		if (currFilter != filter) {
			console.log('UPDATE FILTER', id, filter);
			currFilter = $state.snapshot(filter);
			ctx.map!.setFilter(id, filter);
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
		ctx.map?.off('styledata', updateLayer);
		if (l) {
			// console.log('ON DESTROY LAYER', id, ctx.map);
			ctx.map?.removeLayer(id);
		} else {
			// console.log('ON DESTROY LAYER --- NO LAYER FOUND!!', id);
		}
	});
</script>
