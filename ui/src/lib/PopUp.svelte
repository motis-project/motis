<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import { getContext, type Snippet } from 'svelte';

	let {
		children
	}: {
		children: Snippet;
	} = $props();

	let ctx: { id: null | string } = getContext('layer'); // from Layer component

	let popupEl = $state<HTMLDivElement | undefined>(undefined);
	let popup: null | maplibregl.Popup = null;
	$effect(() => {
		if (popupEl === undefined) {
			return;
		}

		if (ctx.id === null && popup !== null) {
			popup.remove();
		} else if (ctx.id !== null && popup === null) {
			new maplibregl.Popup().setLngLat(e.lngLat).setDOMContent(popupEl).addTo(map!);
		}
	});
</script>

<div bind:this={popupEl}>
	{@render children()}
</div>
