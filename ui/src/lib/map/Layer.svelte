<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import { onDestroy, getContext, setContext, type Snippet } from 'svelte';
	import type { MapMouseEvent, MapGeoJSONFeature } from 'maplibre-gl';

	type ClickHandler = (
		e: MapMouseEvent & { features?: MapGeoJSONFeature[] },
		map: maplibregl.Map
	) => void;

	let {
		id,
		type,
		filter,
		layout,
		paint,
		beforeLayerId = 'road-ref-shield',
		onclick,
		onmousemove,
		onmouseenter,
		onmouseleave,
		children
	}: {
		id: string;
		type: 'symbol' | 'fill' | 'line' | 'circle';
		filter: maplibregl.FilterSpecification;
		layout: Object; // eslint-disable-line
		paint: Object; // eslint-disable-line
		beforeLayerId?: string;
		onclick?: ClickHandler;
		onmousemove?: ClickHandler;
		onmouseenter?: ClickHandler;
		onmouseleave?: ClickHandler;
		children?: Snippet;
	} = $props();

	type PendingState = {
		moves: Map<string, string>;
		frame: number | null;
	};

	const pendingMoves = new WeakMap<maplibregl.Map, PendingState>();

	const ensurePendingState = (map: maplibregl.Map): PendingState => {
		let state = pendingMoves.get(map);
		if (!state) {
			state = { moves: new Map(), frame: null };
			pendingMoves.set(map, state);
		}
		return state;
	};

	function click(e: MapMouseEvent & { features?: MapGeoJSONFeature[] }) {
		if (onclick) {
			onclick(e, ctx.map!);
		}
	}

	function mousemove(e: MapMouseEvent & { features?: MapGeoJSONFeature[] }) {
		if (onmousemove) {
			onmousemove(e, ctx.map!);
		}
	}

	function mouseenter(e: MapMouseEvent & { features?: MapGeoJSONFeature[] }) {
		if (onmouseenter) {
			onmouseenter(e, ctx.map!);
		}
	}

	function mouseleave(e: MapMouseEvent & { features?: MapGeoJSONFeature[] }) {
		if (onmouseleave) {
			onmouseleave(e, ctx.map!);
		}
	}

	const processPendingMoves = (map: maplibregl.Map) => {
		const state = pendingMoves.get(map);
		if (!state) {
			return;
		}
		for (const [layerId, beforeId] of Array.from(state.moves.entries())) {
			if (!map.getLayer(layerId)) {
				state.moves.delete(layerId);
				continue;
			}
			if (!map.getLayer(beforeId)) {
				continue;
			}
			map.moveLayer(layerId, beforeId);
			state.moves.delete(layerId);
		}
		if (state.moves.size === 0) {
			if (state.frame !== null) {
				cancelAnimationFrame(state.frame);
			}
			pendingMoves.delete(map);
		}
	};

	const scheduleMoveLayer = (
		map: maplibregl.Map,
		layerId: string,
		beforeId: string | undefined
	) => {
		if (!beforeId || beforeId === layerId) {
			return;
		}
		const state = ensurePendingState(map);
		state.moves.set(layerId, beforeId);
		if (state.frame !== null) {
			return;
		}
		state.frame = requestAnimationFrame(() => {
			state.frame = null;
			processPendingMoves(map);
		});
	};

	const clearPendingForLayer = (map: maplibregl.Map | null | undefined, layerId: string) => {
		if (!map) {
			return;
		}
		const state = pendingMoves.get(map);
		if (!state) {
			return;
		}
		state.moves.delete(layerId);
		if (state.moves.size === 0) {
			if (state.frame !== null) {
				cancelAnimationFrame(state.frame);
			}
			pendingMoves.delete(map);
		}
	};

	let layer = $state<{ id: null | string }>({ id });
	setContext('layer', layer);

	let ctx: { map: maplibregl.Map | null } = getContext('map'); // from Map component
	let source: { id: string | null } = getContext('source'); // from GeoJSON component

	let initialized = false;
	/* eslint-disable-next-line @typescript-eslint/ban-ts-comment */
	// @ts-ignore
	let currFilter = $state.snapshot(filter);
	let currLayout = $state.snapshot(layout);
	let currPaint = $state.snapshot(paint);
	let currBefore = beforeLayerId;

	let updateLayer = () => {
		const map = ctx.map;
		const l = map?.getLayer(id);
		if (!source.id) {
			if (l) {
				layer.id = null;
				map?.removeLayer(id);
			}
			clearPendingForLayer(map, id);
			return;
		}

		if (l && filter == currFilter && layout == currLayout && paint == currPaint) {
			if (map) {
				processPendingMoves(map);
			}
			return;
		}

		if (!l) {
			const before = beforeLayerId && map?.getLayer(beforeLayerId) ? beforeLayerId : undefined;
			map!.addLayer(
				// @ts-expect-error not assignable
				{
					source: source.id,
					id,
					type,
					filter,
					layout,
					paint
				},
				before
			);
			currFilter = $state.snapshot(filter);
			currLayout = $state.snapshot(layout);
			currPaint = $state.snapshot(paint);
			currBefore = beforeLayerId;
			layer.id = id;
			if (beforeLayerId && !before) {
				scheduleMoveLayer(map!, id, beforeLayerId);
			}
			processPendingMoves(map!);
			return;
		}

		if (currFilter != filter) {
			currFilter = $state.snapshot(filter);
			map!.setFilter(id, filter);
		}
		if (currBefore !== beforeLayerId) {
			currBefore = beforeLayerId;
			if (beforeLayerId && map?.getLayer(beforeLayerId)) {
				map!.moveLayer(id, beforeLayerId);
			} else if (beforeLayerId) {
				scheduleMoveLayer(map!, id, beforeLayerId);
			}
		}
		if (map) {
			processPendingMoves(map);
		}
	};

	$effect(() => {
		if (ctx.map && source.id) {
			if (!initialized) {
				if (onclick) {
					ctx.map.on('click', id, click);
				}
				if (onmousemove) {
					ctx.map.on('mousemove', id, mousemove);
				}
				if (onmouseenter) {
					ctx.map.on('mouseenter', id, mouseenter);
				}
				if (onmouseleave) {
					ctx.map.on('mouseleave', id, mouseleave);
				}
				ctx.map.on('styledata', updateLayer);
				updateLayer();
				initialized = true;
			}
			processPendingMoves(ctx.map);
		}
	});

	$effect(() => {
		if (initialized) {
			updateLayer();
		}
	});

	onDestroy(() => {
		const l = ctx.map?.getLayer(id);
		ctx.map?.off('styledata', updateLayer);
		if (onclick) {
			ctx.map?.off('click', id, click);
		}
		if (onmousemove) {
			ctx.map?.off('mousemove', id, mousemove);
		}
		if (onmouseenter) {
			ctx.map?.off('mouseenter', id, mouseenter);
		}
		if (onmouseleave) {
			ctx.map?.off('mouseleave', id, mouseleave);
		}
		if (l) {
			ctx.map?.removeLayer(id);
		}
		clearPendingForLayer(ctx.map, id);
	});
</script>

{#if children}
	{@render children()}
{/if}
