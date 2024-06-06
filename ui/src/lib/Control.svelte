<script lang="ts">
	import type { Map, ControlPosition } from 'maplibre-gl';
	import { getContext, onDestroy } from 'svelte';
	import { get } from 'svelte/store';

	let { children, ...props } = $props();

	let el: HTMLElement | null = null;

	let initialized = $state(false);

	class Control implements maplibregl.IControl {
		onAdd(map: Map): HTMLElement {
			return el!;
		}
		onRemove(map: Map): void {
			el?.parentNode?.removeChild(el);
		}
		getDefaultPosition?: (() => ControlPosition) | undefined;
	}

	let ctrl = new Control();

	let ctx: { map: maplibregl.Map | null } = getContext('map');

	$effect(() => {
		if (ctx.map && el) {
			ctx.map.addControl(ctrl, 'top-right');
			initialized = true;
		}
	});

	onDestroy(() => ctx.map?.removeControl(ctrl));
</script>

<div class:hidden={!initialized} class="maplibregl-ctrl" bind:this={el}>
	<div class="maplibregl-ctrl-group">
		<button {...props}>
			{@render children()}
		</button>
	</div>
</div>

<style>
	.ctrl-btn-center {
		display: grid !important;
		height: 100%;
		width: 100%;
		place-items: center;
	}
</style>
