<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Card } from '$lib/components/ui/card';
	import { Separator } from '$lib/components/ui/separator';
	import { formatDurationSec } from '$lib/formatDuration';
	import { getModeStyle, routeColor } from '$lib/modeStyle';
	import { plan, type Itinerary, type Leg, type PlanData, type PlanResponse } from '$lib/openapi';
	import Time from '$lib/Time.svelte';
	import LoaderCircle from 'lucide-svelte/icons/loader-circle';
	import { t } from '$lib/i18n/translation';

	let {
		routingResponses,
		baseResponse,
		baseQuery,
		selectedItinerary = $bindable()
	}: {
		routingResponses: Array<Promise<PlanResponse>>;
		baseResponse: Promise<PlanResponse> | undefined;
		baseQuery: PlanData | undefined;
		selectedItinerary: Itinerary | undefined;
	} = $props();
</script>

{#snippet legSummary(l: Leg)}
	<div
		class="flex items-center py-1 px-2 rounded-lg font-bold text-sm h-8 text-nowrap"
		style={routeColor(l)}
	>
		<svg class="relative mr-1 w-4 h-4 rounded-full">
			<use xlink:href={`#${getModeStyle(l.mode)[0]}`}></use>
		</svg>
		{#if l.routeShortName}
			{l.routeShortName}
		{:else}
			{formatDurationSec(l.duration)}
		{/if}
	</div>
{/snippet}

{#if baseResponse}
	{#await baseResponse}
		<div class="flex items-center justify-center w-full">
			<LoaderCircle class="animate-spin w-12 h-12 m-20" />
		</div>
	{:then r}
		{#if r.direct.length !== 0}
			<div class="my-4 flex flex-wrap gap-x-3 gap-y-3">
				{#each r.direct as d}
					<Button
						variant="link"
						onclick={() => {
							selectedItinerary = d;
						}}
					>
						{@render legSummary(d.legs[0]!)}
					</Button>
				{/each}
			</div>
		{/if}

		{#if r.itineraries.length !== 0}
			<div class="flex flex-col space-y-8 px-4 py-8">
				{#each routingResponses as r, rI}
					{#await r}
						<div class="flex items-center justify-center w-full">
							<LoaderCircle class="animate-spin w-12 h-12 m-20" />
						</div>
					{:then r}
						{#if rI === 0 && baseQuery}
							<div class="w-full flex justify-between items-center space-x-4">
								<div class="border-t w-full h-0"></div>
								<button
									onclick={() => {
										routingResponses.splice(
											0,
											0,
											plan({
												query: { ...baseQuery.query, pageCursor: r.previousPageCursor }
											}).then((x) => x.data!)
										);
									}}
									class="px-2 py-1 bg-blue-600 hover:!bg-blue-700 text-white font-bold text-sm border rounded-lg"
								>
									{t.earlier}
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
											<div class="text-xs font-bold uppercase text-slate-400">{t.departure}</div>
											<Time
												isRealtime={it.legs[0].realTime}
												timestamp={it.startTime}
												scheduledTimestamp={it.legs[0].scheduledStartTime}
												variant={'realtime-show-always'}
											/>
										</div>
										<Separator orientation="vertical" />
										<div>
											<div class="text-xs font-bold uppercase text-slate-400">{t.arrival}</div>
											<Time
												isRealtime={it.legs[it.legs.length - 1].realTime}
												timestamp={it.endTime}
												scheduledTimestamp={it.legs[it.legs.length - 1].scheduledStartTime}
												variant={'realtime-show-always'}
											/>
										</div>
										<Separator orientation="vertical" />
										<div>
											<div class="text-xs font-bold uppercase text-slate-400">Transfers</div>
											<div class="flex justify-center w-full">{it.transfers}</div>
										</div>
										<Separator orientation="vertical" />
										<div>
											<div class="text-xs font-bold uppercase text-slate-400">{t.duration}</div>
											<div class="flex justify-center w-full">
												{formatDurationSec(it.duration)}
											</div>
										</div>
									</div>
									<Separator class="my-2" />
									<div class="mt-4 flex flex-wrap gap-x-3 gap-y-3">
										{#each it.legs.filter((l, i) => (i == 0 && l.duration > 1) || (i == it.legs.length - 1 && l.duration > 1) || l.routeShortName || l.mode != 'WALK') as l}
											{@render legSummary(l)}
										{/each}
									</div>
								</Card>
							</button>
						{/each}
						{#if rI === routingResponses.length - 1 && baseQuery}
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
									{t.later}
								</button>
								<div class="border-t w-full h-0"></div>
							</div>
						{/if}
					{:catch e}
						<div>Error: {e}</div>
					{/await}
				{/each}
			</div>
		{/if}
	{/await}
{/if}
