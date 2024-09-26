<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { Separator } from '$lib/components/ui/separator';
	import { formatDurationSec } from '$lib/formatDuration';
	import { getModeStyle, routeColor } from '$lib/modeStyle';
	import { plan, type Itinerary, type PlanData, type PlanResponse } from '$lib/openapi';
	import Time from '$lib/Time.svelte';
	import LoaderCircle from 'lucide-svelte/icons/loader-circle';

	let {
		routingResponses,
		baseQuery,
		selectedItinerary = $bindable()
	}: {
		routingResponses: Array<Promise<PlanResponse>>;
		baseQuery: PlanData;
		selectedItinerary: Itinerary | undefined;
	} = $props();
</script>

<div class="flex flex-col space-y-8 px-4 py-8">
	{#each routingResponses as r, rI}
		{#await r}
			<div class="flex items-center justify-center w-full">
				<LoaderCircle class="animate-spin w-12 h-12 m-20" />
			</div>
		{:then r}
			{#if rI === 0}
				<div class="w-full flex justify-between items-center space-x-4">
					<div class="border-t w-full h-0"></div>
					<button
						onclick={() => {
							routingResponses.splice(
								0,
								0,
								plan({ query: { ...baseQuery.query, pageCursor: r.previousPageCursor } }).then(
									(x) => x.data!
								)
							);
						}}
						class="px-2 py-1 bg-blue-600 hover:!bg-blue-700 text-white font-bold text-sm border rounded-lg"
					>
						früher
					</button>
					<div class="border-t w-full h-0"></div>
				</div>
			{/if}
			{#each r.itineraries as it}
				<button
					onclick={() => {
						selectedItinerary = it;
					}}
				>
					<Card class="p-4">
						<div class="text-base h-8 flex justify-between items-center space-x-4 w-full">
							<div>
								<div class="text-xs font-bold uppercase text-slate-400">Departure</div>
								<Time timestamp={it.startTime} />
							</div>
							<Separator orientation="vertical" />
							<div>
								<div class="text-xs font-bold uppercase text-slate-400">Arrival</div>
								<Time timestamp={it.endTime} />
							</div>
							<Separator orientation="vertical" />
							<div>
								<div class="text-xs font-bold uppercase text-slate-400">Transfers</div>
								<div class="flex justify-center w-full">{it.transfers}</div>
							</div>
							<Separator orientation="vertical" />
							<div>
								<div class="text-xs font-bold uppercase text-slate-400">Duration</div>
								<div class="flex justify-center w-full">
									{formatDurationSec(it.duration)}
								</div>
							</div>
						</div>
						<Separator class="my-2" />
						<div class="mt-4 flex flex-wrap gap-x-4 gap-y-4">
							{#each it.legs.filter((l) => l.routeShortName) as l}
								<div
									class="flex items-center py-1 px-2 rounded-lg font-bold text-sm h-8 text-nowrap"
									style={routeColor(l)}
								>
									<svg class="relative mr-1 w-4 h-4 fill-white rounded-full">
										<use xlink:href={`#${getModeStyle(l.mode)[0]}`}></use>
									</svg>
									{l.routeShortName}
								</div>
							{/each}
						</div>
					</Card>
				</button>
			{/each}
			{#if rI === routingResponses.length - 1}
				<div class="w-full flex justify-between items-center space-x-4">
					<div class="border-t w-full h-0"></div>
					<button
						onclick={() => {
							routingResponses.push(
								plan({ query: { ...baseQuery.query, pageCursor: r.nextPageCursor } }).then(
									(x) => x.data!
								)
							);
						}}
						class="px-2 py-1 bg-blue-600 hover:!bg-blue-700 text-white text-sm font-bold border rounded-lg"
					>
						später
					</button>
					<div class="border-t w-full h-0"></div>
				</div>
			{/if}
		{:catch e}
			<div>Error: {e}</div>
		{/await}
	{/each}
</div>
