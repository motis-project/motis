<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import { getContext, onDestroy, type Snippet } from 'svelte';

	let {
		children,
		class: className,
		trigger
	}: {
		children?: Snippet<
			[maplibregl.MapMouseEvent, () => void, maplibregl.MapGeoJSONFeature[] | undefined]
		>;
		class?: string;
		trigger: 'click' | 'contextmenu';
	} = $props();

	let ctx: { map: maplibregl.Map | null } = getContext('map'); // from Map component
	let layer: { id: string } | null = getContext('layer'); // from Layer component (optional)

	let popupEl = $state<HTMLDivElement>();
	let popup = $state<maplibregl.Popup>();
	let event = $state.raw<maplibregl.MapMouseEvent>();
	let features = $state.raw<maplibregl.MapGeoJSONFeature[]>();

	const close = () => {
		if (popup) {
			popup.remove();
			popup = undefined;
		}
	};

	const onTrigger = (e: maplibregl.MapLayerMouseEvent) => {
		if (ctx.map) {
			if (popup) {
				popup.remove();
			}
			popup = new maplibregl.Popup({
				anchor: 'top-left',
				closeButton: false,
				maxWidth: 'none'
			});
			popup.setLngLat(e.lngLat);
			popup.addTo(ctx.map!);
			event = e;
			features = e.features;
		}
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

	onDestroy(() => {
		if (popup) {
			popup.remove();
			popup = undefined;
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
