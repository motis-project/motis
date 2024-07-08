<script lang="ts">
	import ChevronRight from 'lucide-svelte/icons/chevron-right';
	import type { Itinerary } from '$lib/openapi';
	import Time from './Time.svelte';

	class Props {
		itinerary!: Itinerary;
	}

	let { itinerary }: Props = $props();
</script>

<div class="p-2">
	{#each itinerary.legs as l}
		{#if l.routeShortName}
			<div class="w-full">
				<div class="w-full flex justify-between items-center space-x-1">
					<div class="text-nowrap bg-black text-white rounded-full px-2 py-1 font-bold">
						{l.routeShortName}
					</div>
					<div class="border-t w-full h-0"></div>
					<div class="text-nowrap border rounded-lg">
						{l.from.track}
					</div>
				</div>

				<div class="border-l-2">
					<div class="flex">
						<Time timestamp={l.startTime} />
						<span>{l.from.name}</span>
					</div>
				</div>
			</div>
		{:else}
			{l.mode}
		{/if}
	{/each}
</div>
