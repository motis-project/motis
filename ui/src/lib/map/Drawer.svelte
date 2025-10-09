<script lang="ts">
    import type { WithElementRef } from "bits-ui";
    import type { HTMLAttributes } from "svelte/elements";
    import { cn } from "$lib/utils.js";
    import { onMount } from "svelte";

    let {
        ref = $bindable(null),
        showMap = $bindable(),
        class: className,
        children,
        ...restProps
    }: any = $props();

    let expanded = true;
    let threshold = 0;
    const maxTranslate = 0.7;
    const minTranslate = 0;
    let startY = 0;
    let currentY = 0;

    onMount(() => {
        threshold = (window.innerHeight) / 2;
    });

    const touchStart = (e : TouchEvent) => {
        startY = e.touches[0].clientY;
    };

    const touchMove = (e : TouchEvent) => {
        currentY = e.touches[0].clientY;
        const delta = e.touches[0].clientY - startY;
        const baseTranslate = expanded ? 0 : window.innerHeight * maxTranslate;
        if (expanded && delta < 0 || !expanded && delta > 0) return;
        showMap = true
        ref.style.transition = 'none';
        ref.style.transform = `translateY(${baseTranslate + delta}px)`;
    };
    const touchEnd = (e : TouchEvent) => {
        ref.style.transition = '';
        const delta = e.changedTouches[0].clientY - startY;
        const baseTranslate = expanded ? 0 : window.innerHeight * maxTranslate;
        if (delta + baseTranslate > threshold && expanded) {
            ref.style.transform = `translateY(${window.innerHeight * maxTranslate}px)`;
            expanded = false;
        } else if (delta + baseTranslate <= threshold && !expanded) {
            ref.style.transform = `translateY(${minTranslate}px)`;
            expanded = true;
        } else {
            ref.style.transform = `translateY(${expanded ? minTranslate : window.innerHeight * maxTranslate}px)`;
        }
    };
    
</script>


<div
	bind:this={ref}
	class={cn("bg-card text-card-foreground rounded-xl border shadow max-w-full transition-transform duration-200 ease-out", className)}
	{...restProps}
    ontouchstart={touchStart}
    ontouchmove={touchMove}
    ontouchend={touchEnd}
>
    <div 
        class="w-14 mx-auto my-5 h-2 bg-gray-300 rounded-full"
    ></div>
	{@render children?.()}
</div>
