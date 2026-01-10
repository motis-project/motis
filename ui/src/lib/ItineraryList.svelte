<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import ErrorMessage from '$lib/ErrorMessage.svelte';
	import { Separator } from '$lib/components/ui/separator';
	import { formatDurationSec } from '$lib/formatDuration';
	import { getModeStyle, routeColor } from '$lib/modeStyle';
	import {
		plan,
		type Itinerary,
		type Leg,
		type PlanData,
		type PlanError,
		type PlanResponse,
		type Error as ApiError
	} from '@motis-project/motis-client';
	import Time from '$lib/Time.svelte';
	import { LoaderCircle } from '@lucide/svelte';
	import { t } from '$lib/i18n/translation';
	import DirectConnection from '$lib/DirectConnection.svelte';
	import type { RequestResult } from '@hey-api/client-fetch';

	let {
		routingResponses,
		baseResponse,
		baseQuery,
		selectItinerary,
		updateStartDest
	}: {
		routingResponses: Array<Promise<PlanResponse>>;
		baseResponse: Promise<PlanResponse> | undefined;
		baseQuery: PlanData | undefined;
		selectItinerary: (it: Itinerary) => void;
		updateStartDest: (r: Awaited<RequestResult<PlanResponse, ApiError, false>>) => PlanResponse;
	} = $props();

	const throwOnError = (promise: RequestResult<PlanResponse, PlanError, false>) =>
		promise.then((res) => {
			if (res.error) {
				throw { error: res.error.error, status: res.response.status };
			}
			return res;
		});
</script>

{#snippet legSummary(l: Leg)}
	<div
		class="flex items-center py-1 px-2 rounded-lg font-bold text-sm h-8 text-nowrap"
		style={routeColor(l)}
	>
		<svg class="relative mr-1 w-4 h-4 rounded-full">
			<use xlink:href={`#${getModeStyle(l)[0]}`}></use>
		</svg>
		{#if l.displayName}
			{l.displayName}
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
				{#each r.direct as d, i (i)}
					<DirectConnection
						{d}
						onclick={() => {
							selectItinerary(d);
						}}
					/>
				{/each}
			</div>
		{/if}

		{#if r.itineraries.length !== 0}
			<div class="flex flex-col space-y-6 px-4 py-8">
				{#each routingResponses as r, rI (rI)}
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
											throwOnError(
												plan({
													query: { ...baseQuery.query, pageCursor: r.previousPageCursor }
												})
											).then(updateStartDest)
										);
									}}
									class="px-2 py-1 bg-blue-600 hover:!bg-blue-700 text-white font-bold text-sm border rounded-lg text-nowrap"
								>
									{t.earlier}
								</button>
								<div class="border-t w-full h-0"></div>
							</div>
						{/if}
						{#each r.itineraries as it, i (i)}
							<button
								onclick={() => {
									selectItinerary(it);
								}}
							>
								<Card class="p-4">
									<div class="text-base flex justify-around items-start space-x-1 w-full">
										<div class="overflow-hidden basis-1/4 h-full flex flex-col">
											<div class="text-xs font-bold uppercase text-slate-400">{t.departure}</div>
											<Time
												isRealtime={it.legs[0].realTime}
												timestamp={it.startTime}
												scheduledTimestamp={it.legs[0].scheduledStartTime}
												variant="realtime-show-always"
												queriedTime={baseQuery?.query.time}
												timeZone={it.legs[0].from.tz}
											/>
										</div>
										<Separator orientation="vertical" />
										<div class="overflow-hidden basis-1/4 h-full flex flex-col">
											<div class="text-xs font-bold uppercase text-slate-400">{t.arrival}</div>
											<Time
												isRealtime={it.legs[it.legs.length - 1].realTime}
												timestamp={it.endTime}
												scheduledTimestamp={it.legs[it.legs.length - 1].scheduledEndTime}
												variant="realtime-show-always"
												queriedTime={it.startTime}
												timeZone={it.legs[it.legs.length - 1].to.tz}
											/>
										</div>
										<Separator orientation="vertical" />
										<div class="overflow-hidden basis-1/4 h-full flex flex-col">
											<div class="text-xs font-bold uppercase text-slate-400">
												{t.transfers}
											</div>
											<div class="text-center text-nowrap">
												{it.transfers}
											</div>
										</div>
										<Separator orientation="vertical" />
										<div class="overflow-hidden basis-1/4 h-full flex flex-col">
											<div class="text-xs font-bold uppercase text-slate-400">
												{t.duration}
											</div>
											<div class="text-center text-nowrap">
												{formatDurationSec(it.duration)}
											</div>
										</div>
									</div>
									<Separator class="my-2" />
									<div class="mt-4 flex flex-wrap gap-x-3 gap-y-3">
										{#each it.legs.filter((l, i) => (i == 0 && l.duration > 1) || (i == it.legs.length - 1 && l.duration > 1) || l.displayName || l.mode != 'WALK') as l, i (i)}
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
											throwOnError(
												plan({
													query: { ...baseQuery.query, pageCursor: r.nextPageCursor }
												})
											).then(updateStartDest)
										);
									}}
									class="px-2 py-1 bg-blue-600 hover:!bg-blue-700 text-white text-sm font-bold border rounded-lg text-nowrap"
								>
									{t.later}
								</button>
								<div class="border-t w-full h-0"></div>
							</div>
						{/if}
					{:catch e}
						<ErrorMessage message={e.error ?? e} status={e.status ?? 404} />
					{/await}
				{/each}
			</div>
		{:else if r.direct.length === 0}
			<ErrorMessage message={t.noItinerariesFound} status={200} />
		{/if}
	{:catch e}
		<ErrorMessage message={e.error ?? e} status={e.status ?? 404} />
	{/await}
{/if}
