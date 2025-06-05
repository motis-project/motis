<script lang="ts">
	import { cn } from '$lib/utils';
	import type { Map, ControlPosition, IControl } from 'maplibre-gl';
	import { getContext, onDestroy, type Snippet } from 'svelte';

	let {
		children,
		position,
		class: className
	}: {
		children?: Snippet;
		position?: ControlPosition;
		class?: string;
	} = $props();
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
		if (ctx.map && el && position != undefined) {
			ctx.map.addControl(ctrl, position);
			initialized = true;
		}
	});

	onDestroy(() => ctx.map?.removeControl(ctrl));
</script>

<div
	class:hidden={!initialized && position != undefined}
	class={cn('clear-both pointer-events-auto pt-2 md:pt-4 px-2 md:px-4 max-w-full', className)}
	bind:this={el}
>
	{#if children}
		{@render children()}
	{/if}
</div>
