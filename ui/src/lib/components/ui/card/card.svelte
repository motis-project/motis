<script lang="ts">
	import type { WithElementRef } from "bits-ui";
	import type { HTMLAttributes } from "svelte/elements";
	import { cn } from "$lib/utils.js";
	import {onMount } from "svelte";
	import { restoreScroll} from "$lib/map/handleScroll";
	import { browser } from "$app/environment";

	let {
		ref = $bindable(null),
		class: className,
		children,
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLDivElement>> = $props();
	
	onMount(
		() => {
			if (!(browser && window.innerWidth < 768) && ref!.classList.contains("scrollable")) {
				const cleanup = restoreScroll(ref!);
				return cleanup;
			}	
		}
	);

</script>

<div
	bind:this={ref}
	class={cn("bg-card text-card-foreground rounded-xl border shadow max-w-full", className)}
	{...restProps}
>
	{@render children?.()}
</div>
