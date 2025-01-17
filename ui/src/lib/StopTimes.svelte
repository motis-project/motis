<script lang="ts">
	import { stoptimes, type StoptimesError, type StoptimesResponse } from '$lib/openapi';
	import LoaderCircle from 'lucide-svelte/icons/loader-circle';
	import ArrowRight from 'lucide-svelte/icons/arrow-right';
	import ErrorMessage from '$lib/ErrorMessage.svelte';
	import Time from '$lib/Time.svelte';
	import Route from '$lib/Route.svelte';
	import { Button } from '$lib/components/ui/button';
	import { t } from '$lib/i18n/translation';
	import type { RequestResult } from '@hey-api/client-fetch';

	let {
		stopId,
		time: queryTime,
		arriveBy = $bindable(),
		onClickTrip
	}: {
		stopId: string;
		time: Date;
		arriveBy?: boolean;
		onClickTrip: (tripId: string) => void;
	} = $props();

	let query = $derived({ stopId, time: queryTime.toISOString(), arriveBy, n: 10 });
	let responses = $state<Array<Promise<StoptimesResponse>>>([]);
	$effect(() => {
		responses = [throwOnError(stoptimes({ query }))];
	});

	const throwOnError = (promise: RequestResult<StoptimesResponse, StoptimesError, false>) =>
		promise.then((response) => {
			if (response.error) {
				console.log(response.error);
				throw new Error('HTTP ' + response.response?.status);
			}
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
				arriveBy = !arriveBy;
			}}
		>
			{#if arriveBy}
				{t.switchToDepartures}
			{:else}
				{t.switchToArrivals}
			{/if}
		</Button>
	</div>
	{#each responses as r, rI}
		{#await r}
			<div class="col-span-full w-full flex items-center justify-center">
				<LoaderCircle class="animate-spin w-12 h-12 m-20" />
			</div>
		{:then r}
			{#if rI === 0}
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

			{#each r.stopTimes as t}
				{@const timestamp = arriveBy ? t.place.arrival! : t.place.departure!}
				{@const scheduledTimestamp = arriveBy
					? t.place.scheduledArrival!
					: t.place.scheduledDeparture!}
				<Route class="w-fit max-w-32 text-ellipsis overflow-hidden" l={t} {onClickTrip} />
				<Time variant="schedule" isRealtime={t.realTime} {timestamp} {scheduledTimestamp} />
				<Time variant="realtime" isRealtime={t.realTime} {timestamp} {scheduledTimestamp} />
				<div class="flex items-center text-muted-foreground min-w-0">
					<div><ArrowRight class="stroke-muted-foreground h-4 w-4" /></div>
					<span class="ml-1 leading-tight text-ellipsis overflow-hidden">{t.headsign}</span>
				</div>
			{/each}

			{#if rI === responses.length - 1}
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
