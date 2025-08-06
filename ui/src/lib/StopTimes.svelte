<script lang="ts">
	import { stoptimes, type StoptimesError, type StoptimesResponse } from '$lib/api/openapi';
	import LoaderCircle from 'lucide-svelte/icons/loader-circle';
	import ArrowRight from 'lucide-svelte/icons/arrow-right';
	import CircleX from 'lucide-svelte/icons/circle-x';
	import Info from 'lucide-svelte/icons/info';
	import ErrorMessage from '$lib/ErrorMessage.svelte';
	import Time from '$lib/Time.svelte';
	import Route from '$lib/Route.svelte';
	import { Button } from '$lib/components/ui/button';
	import { language, t } from '$lib/i18n/translation';
	import type { RequestResult } from '@hey-api/client-fetch';
	import { onClickStop, onClickTrip } from '$lib/utils';

	let {
		stopId,
		stopName,
		time: queryTime,
		stopNameFromResponse = $bindable(),
		arriveBy
	}: {
		stopId: string;
		stopName: string;
		time: Date;
		arriveBy?: boolean;
		stopNameFromResponse: string;
	} = $props();

	let query = $derived({ stopId, time: queryTime.toISOString(), arriveBy, n: 10, language });
	/* eslint-disable svelte/prefer-writable-derived */
	let responses = $state<Array<Promise<StoptimesResponse>>>([]);
	$effect(() => {
		responses = [throwOnError(stoptimes({ query }))];
	});
	/* eslint-enable svelte/prefer-writable-derived */

	const throwOnError = (promise: RequestResult<StoptimesResponse, StoptimesError, false>) =>
		promise.then((response) => {
			if (response.error) {
				console.log(response.error);
				throw new Error('HTTP ' + response.response?.status);
			}
			stopNameFromResponse = response.data?.place?.name || '';
			return response.data!;
		});
</script>

<div
	class="text-base grid gap-y-2 gap-x-2 grid-cols-[repeat(3,max-content)_auto] auto-rows-fr items-center"
>
	<div class="col-span-full w-full flex items-center justify-center">
		<Button
			class="font-bold"
			variant="outline"
			onclick={() => {
				onClickStop(stopName, stopId, queryTime, !arriveBy);
			}}
		>
			{#if arriveBy}
				{t.switchToDepartures}
			{:else}
				{t.switchToArrivals}
			{/if}
		</Button>
	</div>
	{#each responses as r, rI (rI)}
		{#await r}
			<div class="col-span-full w-full flex items-center justify-center">
				<LoaderCircle class="animate-spin w-12 h-12 m-20" />
			</div>
		{:then r}
			{#if rI === 0 && r.previousPageCursor.length}
				<div class="col-span-full w-full flex justify-between items-center space-x-4">
					<div class="border-t w-full h-0"></div>
					<button
						onclick={() => {
							responses.splice(
								0,
								0,
								throwOnError(stoptimes({ query: { ...query, pageCursor: r.previousPageCursor } }))
							);
						}}
						class="px-2 py-1 bg-blue-600 hover:!bg-blue-700 text-white font-bold text-sm border rounded-lg text-nowrap"
					>
						{t.earlier}
					</button>
					<div class="border-t w-full h-0"></div>
				</div>
			{/if}

			{#each r.stopTimes as stopTime, i (i)}
				{@const timestamp = arriveBy ? stopTime.place.arrival! : stopTime.place.departure!}
				{@const scheduledTimestamp = arriveBy
					? stopTime.place.scheduledArrival!
					: stopTime.place.scheduledDeparture!}
				<Route class="w-fit max-w-32 text-ellipsis overflow-hidden" l={stopTime} {onClickTrip} />
				<Time
					variant="schedule"
					isRealtime={stopTime.realTime}
					{timestamp}
					{scheduledTimestamp}
					queriedTime={queryTime.toISOString()}
				/>
				<Time variant="realtime" isRealtime={stopTime.realTime} {timestamp} {scheduledTimestamp} />
				<span>
					<div class="flex items-center text-muted-foreground min-w-0">
						<div><ArrowRight class="stroke-muted-foreground h-4 w-4" /></div>
						<span class="ml-1 leading-tight text-ellipsis overflow-hidden">{stopTime.headsign}</span
						>
					</div>
					{#if stopTime.pickupDropoffType == 'NOT_ALLOWED'}
						<div class="flex items-center text-destructive text-sm">
							<CircleX class="stroke-destructive h-4 w-4" />
							<span class="ml-1 leading-none">
								{stopTime.tripCancelled
									? t.tripCancelled
									: stopTime.cancelled
										? t.stopCancelled
										: arriveBy
											? t.outDisallowed
											: t.inDisallowed}
							</span>
						</div>
					{/if}
					{#if stopTime.place.alerts}
						<div class="flex items-center text-destructive text-sm">
							<Info class="stroke-destructive h-4 w-4" />
							<span class="ml-1 leading-none">
								{t.alertsAvailable}
							</span>
						</div>
					{/if}
				</span>
			{/each}
			{#if !r.stopTimes.length}
				<div class="col-span-full w-full flex items-center justify-center">
					<ErrorMessage e={t.noItinerariesFound} />
				</div>
			{/if}

			{#if rI === responses.length - 1 && r.nextPageCursor.length}
				<div class="col-span-full w-full flex justify-between items-center space-x-4">
					<div class="border-t w-full h-0"></div>
					<button
						onclick={() => {
							responses.push(
								throwOnError(stoptimes({ query: { ...query, pageCursor: r.nextPageCursor } }))
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
			<div class="col-span-full w-full flex items-center justify-center">
				<ErrorMessage {e} />
			</div>
		{/await}
	{/each}
</div>
