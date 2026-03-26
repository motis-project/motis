<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import { getContext, onDestroy, type Snippet } from 'svelte';

	type PopupSnapshot = {
		lngLat: maplibregl.LngLatLike;
		event: maplibregl.MapMouseEvent;
		features?: maplibregl.MapGeoJSONFeature[];
	};

	type PopupController = {
		open?: (snapshot: PopupSnapshot) => void;
		close?: () => void;
		getSnapshot?: () => PopupSnapshot | null;
		onSnapshotChange?: (snapshot: PopupSnapshot | null) => void;
	};

	let {
		children,
		class: className,
		trigger,
		controller
	}: {
		children?: Snippet<
			[maplibregl.MapMouseEvent, () => void, maplibregl.MapGeoJSONFeature[] | undefined]
		>;
		class?: string;
		trigger: 'click' | 'contextmenu';
		controller?: PopupController;
	} = $props();

	let ctx: { map: maplibregl.Map | null } = getContext('map'); // from Map component
	let layer: { id: string } | null = getContext('layer'); // from Layer component (optional)

	let popupEl = $state<HTMLDivElement>();
	let popup = $state<maplibregl.Popup>();
	let event = $state.raw<maplibregl.MapMouseEvent>();
	let features = $state.raw<maplibregl.MapGeoJSONFeature[]>();

	const clearPopupState = () => {
		popup = undefined;
		event = undefined;
		features = undefined;
		controller?.onSnapshotChange?.(null);
	};

	const openPopup = (snapshot: PopupSnapshot) => {
		if (!ctx.map) {
			return;
		}
		if (popup) {
			popup.remove();
		}
		const nextPopup = new maplibregl.Popup({
			anchor: 'top-left',
			closeButton: false,
			maxWidth: 'none'
		});
		nextPopup.on('close', () => {
			if (popup === nextPopup) {
				clearPopupState();
			}
		});
		nextPopup.setLngLat(snapshot.lngLat);
		nextPopup.addTo(ctx.map);
		popup = nextPopup;
		event = snapshot.event;
		features = snapshot.features;
		controller?.onSnapshotChange?.(snapshot);
	};

	const close = () => popup?.remove();

	const onTrigger = (e: maplibregl.MapLayerMouseEvent) => {
		openPopup({ lngLat: e.lngLat, event: e, features: e.features });
	};

	const onMouseEnter = () => {
		if (ctx.map) {
			ctx.map.getCanvas().style.cursor = 'pointer';
		}
	};

	const onMouseLeave = () => {
		if (ctx.map) {
			ctx.map.getCanvas().style.cursor = '';
		}
	};

	let initialized = false;
	$effect(() => {
		if (ctx.map) {
			if (!initialized) {
				if (layer) {
					ctx.map.on(trigger, layer.id, onTrigger);
					ctx.map.on('mouseenter', layer.id, onMouseEnter);
					ctx.map.on('mouseleave', layer.id, onMouseLeave);
				} else {
					ctx.map.on(trigger, onTrigger);
				}
			}
			initialized = true;
		}
	});

	$effect(() => {
		if (popup && popupEl) {
			popup.setDOMContent(popupEl);
		}
	});

	$effect(() => {
		if (controller) {
			controller.open = openPopup;
			controller.close = close;
			controller.getSnapshot = () => {
				if (!popup || !event) {
					return null;
				}
				return {
					lngLat: popup.getLngLat(),
					event,
					features
				};
			};
		}
	});

	onDestroy(() => {
		if (popup) {
			popup.remove();
			clearPopupState();
		}
		if (ctx.map && initialized) {
			if (layer) {
				ctx.map.off(trigger, layer.id, onTrigger);
				ctx.map.off('mouseenter', layer.id, onMouseEnter);
				ctx.map.off('mouseleave', layer.id, onMouseLeave);
			} else {
				ctx.map.off(trigger, onTrigger);
			}
		}
	});
</script>

{#if popup && event}
	<div bind:this={popupEl} class={className}>
		{#if children}
			{@render children(event, close, features)}
		{/if}
	</div>
{/if}
