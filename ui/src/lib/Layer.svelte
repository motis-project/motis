<script lang="ts">
	import { onDestroy, getContext } from 'svelte';

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

	let ctx: { map: maplibregl.Map | null } = getContext('map');
	let source: { id: string | null } = getContext('source');

	let initialized = false;
	let currFilter = filter;
	let currLayout = layout;
	let currPaint = paint;

	let updateLayer = () => {
		const l = ctx.map?.getLayer(id);
		if (!source.id) {
			if (l) {
				console.log('REMOVE', id);
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
			return;
		}

		if (currFilter != filter) {
			console.log(this, id, currFilter, filter);
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
		if (l) {
			console.log('ON DESTROY LAYER', id, ctx.map);
			ctx.map?.off('styledata', updateLayer);
			ctx.map?.removeLayer(id);
		} else {
			console.log('ON DESTROY LAYER --- NO LAYER FOUND!!', id);
		}
	});
</script>
