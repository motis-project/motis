<script lang="ts">
	import type { Map, ControlPosition } from 'maplibre-gl';
	import { getContext, onDestroy, type Snippet } from 'svelte';

	class Props {
		children!: Snippet;
		position?: maplibregl.ControlPosition = 'top-right';
	}
	let { children, position, ...props }: Props = $props();

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
			ctx.map.addControl(ctrl, position);
			initialized = true;
		}
	});

	onDestroy(() => ctx.map?.removeControl(ctrl));
</script>

<div class:hidden={!initialized} class="maplibregl-ctrl" {...props} bind:this={el}>
	{@render children()}
</div>

<style>
	.ctrl-btn-center {
		display: grid !important;
		height: 100%;
		width: 100%;
		place-items: center;
	}
</style>
