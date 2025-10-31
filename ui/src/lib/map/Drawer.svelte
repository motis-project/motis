<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { ChevronDown } from '@lucide/svelte';
	import { onMount, type Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	import { restoreScroll, resetScroll } from './handleScroll';

	let {
		ref = $bindable(null),
		showMap = $bindable(),
		class: className,
		children,
		...restProps
	}: {
		ref?: HTMLDivElement | null;
		showMap?: boolean;
		class?: string;
		children?: Snippet;
	} & HTMLAttributes<HTMLDivElement> = $props();

	let expanded = $state(true);
	const maxTranslate = 0.7;
	const minTranslate = 0;
	let startY = 0;
	let currentY = 0;
	let isDragging = false;
	let fromHandle = false;
	let container: HTMLElement | null;

	onMount(() => {
		const cleanup = restoreScroll(container!);
		return cleanup;
	});

	$effect(() => resetScroll(container!));

	const getScrollableElement = (element: Element): Element | null => {
		while (element && element !== document.documentElement) {
			const style = window.getComputedStyle(element);
			const overflowY = style.overflowY;
			const hasScrollableY =
				overflowY !== 'visible' &&
				overflowY !== 'hidden' &&
				element.scrollHeight > element.clientHeight;

			if (hasScrollableY) {
				return element;
			}

			element = element.parentElement as HTMLElement;
		}

		return null;
	};

	const ontouchstart = (e: TouchEvent) => {
		const target = !fromHandle ? (e.target as Element) : null;
		const scrollableElement = target ? (getScrollableElement(target) as HTMLElement) : null;
		startY = e.touches[0].clientY;
		isDragging = expanded ? (scrollableElement ? scrollableElement.scrollTop === 0 : true) : true;
	};

	const ontouchmove = (e: TouchEvent) => {
		if (!isDragging) return;
		currentY = e.touches[0].clientY;
		const delta = currentY - startY;
		const baseTranslate = expanded ? 0 : window.innerHeight * maxTranslate;
		if ((expanded && delta < 0) || (!expanded && delta > 0)) return;
		showMap = true;
		ref!.style.transition = 'none';
		ref!.style.transform = `translateY(${baseTranslate + delta}px)`;
	};
	const ontouchend = (e: TouchEvent) => {
		if (!isDragging) return;
		isDragging = false;
		ref!.style.transition = '';
		const delta = e.changedTouches[0].clientY - startY;
		if ((70 < delta || (delta == 0 && fromHandle)) && expanded) {
			ref!.style.transform = `translateY(${window.innerHeight * maxTranslate}px)`;
			expanded = false;
		} else if ((delta < -70 || (delta == 0 && fromHandle)) && !expanded) {
			ref!.style.transform = `translateY(${minTranslate}px)`;
			expanded = true;
		} else {
			ref!.style.transform = `translateY(${expanded ? minTranslate : window.innerHeight * maxTranslate}px)`;
		}
		fromHandle = false;
	};
</script>

<div
	bind:this={ref}
	class={cn(
		'bg-card text-card-foreground rounded-xl pb-5 h-full border shadow max-w-full transition-transform duration-200 ease-out',
		className
	)}
	{...restProps}
	{ontouchstart}
	{ontouchmove}
	{ontouchend}
>
	<div
		class="mx-auto my-5 relative before:content-[''] before:absolute before:inset-[-20px] before:inset-x-[-50vw] flex items-center justify-center"
		ontouchstart={(e) => {
			fromHandle = true;
			showMap = true;
			ontouchstart(e);
		}}
	>
		<div
			class="absolute transition-all duration-200"
			class:opacity-0={!expanded}
			class:opacity-100={expanded}
		>
			<ChevronDown class="size-12 text-gray-300 scale-x-150" strokeWidth={3.5} />
		</div>
		<div
			class="w-14 h-2 bg-gray-300 rounded-full transition-all duration-200"
			class:opacity-0={expanded}
			class:opacity-100={!expanded}
		></div>
	</div>

	<div bind:this={container} class:overflow-auto={expanded}>
		{@render children?.()}
	</div>
</div>
