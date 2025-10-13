<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import { onDestroy, getContext, setContext, type Snippet } from 'svelte';
	import type { MapMouseEvent, MapGeoJSONFeature } from 'maplibre-gl';

	type ClickHandler = (e: MapMouseEvent & { features?: MapGeoJSONFeature[] }) => void;
	type LayerEvents = Partial<
		Record<'click' | 'mouseenter' | 'mouseleave' | 'mousemove', ClickHandler>
	>;

	let {
		id,
		type,
		filter,
		layout,
		paint,
		beforeLayerId = 'road-ref-shield',
		onclick,
		events,
		children
	}: {
		id: string;
		type: 'symbol' | 'fill' | 'line' | 'circle';
		filter: maplibregl.FilterSpecification;
		layout: Object; // eslint-disable-line
		paint: Object; // eslint-disable-line
		beforeLayerId?: string;
		onclick?: ClickHandler;
		events?: LayerEvents;
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
	let currFilter = $state.snapshot(filter);
	let currLayout = $state.snapshot(layout);
	let currPaint = $state.snapshot(paint);
	let currBefore = beforeLayerId;
	let registeredEvents: Array<[keyof LayerEvents, ClickHandler]> = [];

	const clearEventListeners = () => {
		if (!ctx.map) {
			return;
		}
		for (const [eventName, handler] of registeredEvents) {
			ctx.map.off(eventName, id, handler);
		}
		registeredEvents = [];
	};

	const resolveEvents = (): LayerEvents => {
		const combined: LayerEvents = { ...(events ?? {}) };
		if (onclick) {
			combined.click = onclick;
		}
		return combined;
	};

	const updateEventListeners = () => {
		if (!ctx.map || !layer.id) {
			return;
		}
		clearEventListeners();
		const combined = resolveEvents();
		for (const eventName of Object.keys(combined) as (keyof LayerEvents)[]) {
			const handler = combined[eventName];
			if (!handler) {
				continue;
			}
			ctx.map.on(eventName, id, handler);
			registeredEvents.push([eventName, handler]);
		}
	};

	let updateLayer = () => {
		const map = ctx.map;
		const l = map?.getLayer(id);
		if (!source.id) {
			if (l) {
				layer.id = null;
				clearEventListeners();
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
			updateEventListeners();
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
		updateEventListeners();
		if (map) {
			processPendingMoves(map);
		}
	};

	$effect(() => {
		if (ctx.map && source.id) {
			if (!initialized) {
				ctx.map.on('styledata', updateLayer);
				updateLayer();
				initialized = true;
			}
			processPendingMoves(ctx.map);
		}
	});

	onDestroy(() => {
		const l = ctx.map?.getLayer(id);
		ctx.map?.off('styledata', updateLayer);
		clearEventListeners();
		if (l) {
			ctx.map?.removeLayer(id);
		}
		clearPendingForLayer(ctx.map, id);
	});
</script>

{#if children}
	{@render children()}
{/if}
