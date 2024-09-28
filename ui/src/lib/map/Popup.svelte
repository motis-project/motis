<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import { getContext, onDestroy, type Snippet } from 'svelte';

	let {
		children,
		class: className,
		trigger
	}: {
		children?: Snippet<[maplibregl.MapMouseEvent, () => void]>;
		class?: string;
		trigger: string;
	} = $props();

	let ctx: { map: maplibregl.Map | null } = getContext('map'); // from Map component

	let popupEl = $state<HTMLDivElement>();
	let popup = $state<maplibregl.Popup>();
	let event = $state<maplibregl.MapMouseEvent>();

	const close = () => {
		if (popup) {
			popup.remove();
			popup = undefined;
		}
	};

	const onContextMenu = (e: maplibregl.MapMouseEvent) => {
		if (ctx.map) {
			if (popup) {
				popup.remove();
			}
			popup = new maplibregl.Popup({
				anchor: 'top-left',
				closeButton: false
			});
			popup.setLngLat(e.lngLat);
			popup.addTo(ctx.map!);
			event = e;
		}
	};

	let initialized = false;
	$effect(() => {
		if (ctx.map) {
			if (!initialized) {
				ctx.map.on(trigger, onContextMenu);
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
			ctx.map.off('contextmenu', onContextMenu);
		}
	});
</script>

{#if popup && event}
	<div bind:this={popupEl} class={className}>
		{#if children}
			{@render children(event, close)}
		{/if}
	</div>
{/if}
