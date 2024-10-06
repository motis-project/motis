<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import { onDestroy, getContext, setContext, type Snippet } from 'svelte';
	import type { MapMouseEvent, MapGeoJSONFeature } from 'maplibre-gl';

	type ClickHandler = (e: MapMouseEvent & { features?: MapGeoJSONFeature[] }) => void;

	let {
		id,
		type,
		filter,
		layout,
		paint,
		onclick,
		children
	}: {
		id: string;
		type: 'symbol' | 'fill' | 'line' | 'circle';
		filter: maplibregl.FilterSpecification;
		layout: Object; // eslint-disable-line
		paint: Object; // eslint-disable-line
		onclick?: ClickHandler;
		children?: Snippet;
	} = $props();

	function click(e: MapMouseEvent & { features?: MapGeoJSONFeature[] }) {
		onclick && onclick(e);
	}

	let layer = $state<{ id: null | string }>({ id });
	setContext('layer', layer);

	let ctx: { map: maplibregl.Map | null } = getContext('map'); // from Map component
	let source: { id: string | null } = getContext('source'); // from GeoJSON component

	let initialized = false;
	let currFilter = $state.snapshot(filter);
	let currLayout = $state.snapshot(layout);
	let currPaint = $state.snapshot(paint);

	let updateLayer = () => {
		const l = ctx.map?.getLayer(id);
		if (!source.id) {
			if (l) {
				layer.id = null;
				ctx.map?.removeLayer(id);
			}
			return;
		}

		if (l && filter == currFilter && layout == currLayout && paint == currPaint) {
			return;
		}

		if (!l) {
			// @ts-expect-error not assignable
			ctx.map!.addLayer({
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
			currFilter = $state.snapshot(filter);
			ctx.map!.setFilter(id, filter);
		}
	};

	$effect(() => {
		if (ctx.map && source.id) {
			if (!initialized) {
				if (onclick) {
					ctx.map.on('click', id, click);
				}
				ctx.map.on('styledata', updateLayer);
				updateLayer();
				initialized = true;
			}
		}
	});

	onDestroy(() => {
		const l = ctx.map?.getLayer(id);
		ctx.map?.off('styledata', updateLayer);
		if (onclick) {
			ctx.map?.off('click', id, click);
		}
		if (l) {
			ctx.map?.removeLayer(id);
		}
	});
</script>

{#if children}
	{@render children()}
{/if}
