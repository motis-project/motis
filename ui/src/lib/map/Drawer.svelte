<script lang="ts">
	import { cn } from '$lib/utils.js';
	import ChevronDown from 'lucide-svelte/icons/chevron-down';
	import { type Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

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

	const ontouchstart = (e: TouchEvent) => {
		startY = e.touches[0].clientY;
		isDragging = true;
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
		if ((70 < delta || delta == 0) && expanded) {
			ref!.style.transform = `translateY(${window.innerHeight * maxTranslate}px)`;
			expanded = false;
		} else if ((delta < -70 || delta == 0) && !expanded) {
			ref!.style.transform = `translateY(${minTranslate}px)`;
			expanded = true;
		} else {
			ref!.style.transform = `translateY(${expanded ? minTranslate : window.innerHeight * maxTranslate}px)`;
		}
	};
</script>

<div
	bind:this={ref}
	class={cn(
		'bg-card text-card-foreground rounded-xl pb-2 h-full border shadow max-w-full transition-transform duration-200 ease-out',
		className
	)}
	{...restProps}
	{ontouchmove}
	{ontouchend}
>
	<div
		class="mx-auto my-5 relative before:content-[''] before:absolute before:inset-[-20px] before:inset-x-[-50vw] flex items-center justify-center"
		{ontouchstart}
	>
		<div
			class="absolute transition-all duration-200"
			class:opacity-0={!expanded}
			class:opacity-100={expanded}
		>
			<ChevronDown class="size-14 text-gray-300 scale-x-150" strokeWidth={3} />
		</div>
		<div
			class="w-14 h-2 bg-gray-300 rounded-full transition-all duration-200"
			class:opacity-0={expanded}
			class:opacity-100={!expanded}
		></div>
	</div>

	<div class:overflow-auto={expanded}>
		{@render children?.()}
	</div>
</div>
