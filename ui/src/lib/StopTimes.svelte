<script lang="ts">
	import {
		stoptimes,
		type StoptimesError,
		type StoptimesResponse
	} from '@motis-project/motis-client';
	import { LoaderCircle, ArrowRight, CircleX } from '@lucide/svelte';
	import ErrorMessage from '$lib/ErrorMessage.svelte';
	import Time from '$lib/Time.svelte';
	import Route from '$lib/Route.svelte';
	import { Button } from '$lib/components/ui/button';
	import { language, t } from '$lib/i18n/translation';
	import type { RequestResult } from '@hey-api/client-fetch';
	import { onClickStop, onClickTrip } from '$lib/utils';
	import { getModeLabel } from './map/getModeLabel';
	import { posToLocation } from './Location';
	import type { Location } from './Location';
	import maplibregl from 'maplibre-gl';
	import Alerts from './Alerts.svelte';

	let {
		stopId,
		stopName,
		time: queryTime,
		stopNameFromResponse = $bindable(),
		stop = $bindable(),
		stopMarker = $bindable(),
		arriveBy
	}: {
		stopId: string;
		stopName: string;
		time: Date;
		arriveBy?: boolean;
		stopNameFromResponse: string;
		stop: Location | undefined;
		stopMarker: maplibregl.Marker | undefined;
	} = $props();

	let query = $derived({
		stopId,
		time: queryTime.toISOString(),
		arriveBy,
		n: 10,
		exactRadius: false,
		radius: 200,
		language: [language]
	});
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
				throw { error: response.error.error, status: response.response.status };
			}
			stopNameFromResponse = response.data?.place?.name || '';
			let placeFromResponse = response.data?.place;
			stop = posToLocation(
				maplibregl.LngLat.convert([placeFromResponse.lon, placeFromResponse.lat])
			);
			stopMarker?.setLngLat(stop.match!);
			return response.data!;
		});
</script>

<div class="flex justify-center mb-4">
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
		<div class="flex items-center justify-center">
			<LoaderCircle
				class="animate-spin w-20 h-20 my-60
				"
			/>
		</div>
	{:then r}
		{#if rI === 0 && r.previousPageCursor.length}
			<div class="col-span-full flex justify-center items-center border-b pb-4">
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
			</div>
		{/if}
		{#each r.stopTimes as stopTime, i (i)}
			{@const timestamp = arriveBy ? stopTime.place.arrival! : stopTime.place.departure!}
			{@const scheduledTimestamp = arriveBy
				? stopTime.place.scheduledArrival!
				: stopTime.place.scheduledDeparture!}

			<div
				class="gap-y-2 p-3 text-base grid grid-cols-[auto_1fr] border-b hover:bg-slate-100 dark:hover:bg-slate-800 duration-500 ease-out transition-all"
			>
				<div class="flex flex-col w-24">
					<Time
						variant="schedule"
						timeZone={stopTime.place.tz}
						isRealtime={stopTime.realTime}
						{timestamp}
						{scheduledTimestamp}
						queriedTime={queryTime.toISOString()}
						{arriveBy}
					/>
					<Time
						variant="realtime"
						timeZone={stopTime.place.tz}
						isRealtime={stopTime.realTime}
						{timestamp}
						{scheduledTimestamp}
						{arriveBy}
					/>
				</div>
				<div class="flex-col text-base">
					<div class="flex justify-between">
						<Route class="text-ellipsis mb-2" l={stopTime} {onClickTrip} />
						{#if stopTime.place.track}
							<span class="text-nowrap ml-3 text-sm py-1 px-3 rounded-lg">
								{getModeLabel(stopTime.mode) == 'Track' ? t.trackAbr : t.platformAbr}
								{stopTime.place.track}
							</span>
						{/if}
					</div>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-3">
							<ArrowRight class="shrink-0 stroke-muted-foreground h-4 w-4" />
							<span class="leading-none">
								{stopTime.headsign}
								{#if !stopTime.headsign}
									{stopTime.tripTo.name}
								{:else if !stopTime.tripTo.name.startsWith(stopTime.headsign)}
									<span class="stroke-muted-foreground">({stopTime.tripTo.name})</span>
								{/if}
							</span>
						</div>
						<Alerts tz={stopTime.place.tz} alerts={stopTime.place.alerts} />
					</div>
				</div>
				{#if stopTime.pickupDropoffType == 'NOT_ALLOWED'}
					<div class="flex items-center col-span-full text-destructive text-sm">
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
			</div>
		{/each}
		{#if !r.stopTimes.length}
			<div class="col-span-full w-full flex items-center justify-center">
				<ErrorMessage message={t.noItinerariesFound} status={404} />
			</div>
		{/if}

		{#if rI === responses.length - 1 && r.nextPageCursor.length}
			<div class="col-span-full flex mt-4 justify-center items-center">
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
			</div>
		{/if}
	{:catch e}
		<div class="col-span-full w-full flex items-center justify-center">
			<ErrorMessage message={e.error} status={e.status} />
		</div>
	{/await}
{/each}
