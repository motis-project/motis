<script lang="ts">
	import ArrowRight from 'lucide-svelte/icons/arrow-right';
	import type { Itinerary, Leg } from '$lib/openapi';
	import Time from '../lib/Time.svelte';
	import { routeBorderColor, routeColor } from '$lib/modeStyle';
	import { formatDurationSec, formatDistanceMeters } from '$lib/formatDuration';
	import { Button } from '$lib/components/ui/button';
	import Route from '$lib/Route.svelte';
	import { getModeName } from '$lib/getModeName';

	const {
		itinerary,
		onClickStop,
		onClickTrip
	}: {
		itinerary: Itinerary;
		onClickStop: (name: string, stopId: string, time: Date) => void;
		onClickTrip: (tripId: string) => void;
	} = $props();

	const lastLeg = $derived(itinerary.legs.findLast((l) => l.duration !== 0));
</script>

{#snippet stopTimes(
	timestamp: string,
	scheduledTimestamp: string,
	isRealtime: boolean,
	name: string,
	stopId?: string
)}
	<Time
		variant="schedule"
		class="font-semibold mr-2"
		{isRealtime}
		{timestamp}
		{scheduledTimestamp}
	/>
	<Time variant="realtime" class="font-semibold" {isRealtime} {timestamp} {scheduledTimestamp} />
	{#if stopId}
		<Button
			class="col-span-5 mr-6 text-lg justify-normal text-wrap text-left"
			variant="link"
			on:click={() => {
				onClickStop(name, stopId, new Date(timestamp));
			}}
		>
			{name}
		</Button>
	{:else}
		<span class="col-span-5 mr-6">{name}</span>
	{/if}
{/snippet}

{#snippet streetLeg(l: Leg)}
	<div class="py-12 pl-8 flex flex-col text-muted-foreground">
		<span class="ml-6">
			{formatDurationSec(l.duration)}
			{getModeName(l.mode)}
			{formatDistanceMeters(Math.round(l.distance!))}
		</span>
		{#if l.rental && l.rental.systemName}
			<span class="ml-6">
				Anbieter: <a href={l.rental.url} target="_blank">{l.rental.systemName}</a>
			</span>
		{/if}
	</div>
{/snippet}

<div class="text-lg">
	{#each itinerary.legs as l, i}
		{@const isLast = i == itinerary.legs.length - 1}
		{@const isLastPred = i == itinerary.legs.length - 2}
		{@const pred = i == 0 ? undefined : itinerary.legs[i - 1]}
		{@const next = isLast ? undefined : itinerary.legs[i + 1]}

		{#if l.routeShortName}
			<div class="w-full flex justify-between items-center space-x-1">
				<Route {onClickTrip} {l} />
				{#if pred && (pred.from.track || pred.duration !== 0)}
					<div class="border-t w-full h-0"></div>
					<div class="text-sm text-muted-foreground text-nowrap px-2">
						{#if pred.from.track}
							Ankunft auf Gleis {pred.from.track}{pred.duration ? ',' : ''}
						{/if}
						{#if pred.duration}
							{formatDurationSec(pred.duration)} Fußweg
						{/if}
						{#if pred.distance}
							({Math.round(pred.distance)} m)
						{/if}
					</div>
				{/if}
				<div class="border-t w-full h-0"></div>
				{#if l.from.track}
					<div class="text-nowrap border rounded-xl px-2">
						Gleis {l.from.track}
					</div>
				{/if}
			</div>

			<div class="pt-4 pl-6 border-l-4 left-4 relative" style={routeBorderColor(l)}>
				<div class="grid gap-y-6 grid-cols-7 items-center">
					{@render stopTimes(
						l.startTime,
						l.scheduledStartTime,
						l.realTime,
						l.from.name,
						l.from.stopId
					)}
				</div>
				<div class="mt-2 flex items-center text-muted-foreground">
					<ArrowRight class="stroke-muted-foreground h-4 w-4" />
					<span class="ml-1">{l.headsign}</span>
				</div>
				{#if l.intermediateStops?.length === 0}
					<div class="py-12 pl-8 flex items-center text-muted-foreground">
						Fahrt ohne Zwischenhalt
					</div>
				{:else}
					<details class="[&_svg]:open:-rotate-180">
						<summary class="py-12 pl-8 flex items-center text-muted-foreground">
							<svg
								class="rotate-0 transform transition-all duration-300"
								fill="none"
								height="20"
								width="20"
								stroke="currentColor"
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								viewBox="0 0 24 24"
							>
								<polyline points="6 9 12 15 18 9"></polyline>
							</svg>
							<span class="ml-2 cursor-pointer">
								Fahrt {l.intermediateStops?.length} Station ({formatDurationSec(l.duration)})
							</span>
						</summary>
						<div class="mb-6 grid gap-y-6 grid-cols-7 items-center">
							{#each l.intermediateStops! as s}
								{@render stopTimes(s.arrival!, s.scheduledArrival!, l.realTime, s.name!, s.stopId)}
							{/each}
						</div>
					</details>
				{/if}

				{#if !isLast && !(isLastPred && next!.duration === 0)}
					<div class="grid gap-y-6 grid-cols-7 items-center pb-3">
						{@render stopTimes(
							l.endTime!,
							l.scheduledEndTime!,
							l.realTime!,
							l.to.name,
							l.to.stopId
						)}
					</div>
				{/if}

				{#if isLast}
					<!-- fill visual gap -->
					<div class="pb-1"></div>
				{/if}
			</div>
		{:else if !(isLast && l.duration === 0) && ((i == 0 && l.duration !== 0) || !next || !next.routeShortName || l.mode != 'WALK' || (pred && (pred.mode == 'BIKE' || pred.mode == 'BIKE_RENTAL')))}
			<Route {onClickTrip} {l} />
			<div class="pt-4 pl-6 border-l-4 left-4 relative" style={routeBorderColor(l)}>
				<div class="grid gap-y-6 grid-cols-7 items-center">
					{@render stopTimes(
						l.startTime,
						l.scheduledStartTime,
						l.realTime,
						l.from.name,
						l.from.stopId
					)}
				</div>
				{@render streetLeg(l)}
				{#if !isLast}
					<div class="grid gap-y-6 grid-cols-7 items-center pb-4">
						{@render stopTimes(l.endTime, l.scheduledEndTime, l.realTime, l.to.name, l.to.stopId)}
					</div>
				{/if}
			</div>
		{/if}
	{/each}
	<div class="flex">
		<div class="relative left-[13px] w-3 h-3 rounded-full" style={routeColor(lastLeg!)}></div>
		<div class="relative left-3 bottom-[7px] pl-6 grid gap-y-6 grid-cols-7 items-center">
			{@render stopTimes(
				lastLeg!.endTime,
				lastLeg!.scheduledEndTime,
				lastLeg!.realTime,
				lastLeg!.to.name,
				lastLeg!.to.stopId
			)}
		</div>
	</div>
</div>
