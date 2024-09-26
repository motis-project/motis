<script lang="ts">
	import type { Map, ControlPosition, IControl } from 'maplibre-gl';
	import { getContext, onDestroy, type Snippet } from 'svelte';

	class Props {
		children?: Snippet;
		position?: ControlPosition = 'top-right';
	}

	let { children, position, ...props }: Props = $props();
	let el: HTMLElement | null = null;
	let initialized = $state(false);

	class Control implements IControl {
		/* eslint-disable-next-line */
		onAdd(map: Map): HTMLElement {
			return el!;
		}
		/* eslint-disable-next-line */
		onRemove(map: Map): void {
			el?.parentNode?.removeChild(el);
		}
		getDefaultPosition?: (() => ControlPosition) | undefined;
	}

	let ctrl = new Control();
	let ctx: { map: Map | null } = getContext('map');

	$effect(() => {
		if (ctx.map && el) {
			ctx.map.addControl(ctrl, position);
			initialized = true;
		}
	});

	onDestroy(() => ctx.map?.removeControl(ctrl));
</script>

<div
	class:hidden={!initialized}
	class="clear-both pointer-events-auto p-4"
	{...props}
	bind:this={el}
>
	{#if children}
		{@render children()}
	{/if}
</div>
