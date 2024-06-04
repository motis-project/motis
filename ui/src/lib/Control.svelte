<script lang="ts">
	import { getContext, onDestroy } from 'svelte';
	import { get } from 'svelte/store';

	let { children, ...props } = $props();

	let position = 'top-right';

	let el: HTMLElement | null = null;

	let control = {
		onAdd() {
			return el;
		},
		onRemove() {
			el?.parentNode?.removeChild(el);
		}
	};

	let ctx = getContext('map');

	$effect(() => {
		if (ctx.map && el) {
			ctx.map.addControl(control, 'top-right');
		}
	});

	onDestroy(() => ctx.map?.removeControl(control));
</script>

<div class="maplibregl-ctrl" bind:this={el}>
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
