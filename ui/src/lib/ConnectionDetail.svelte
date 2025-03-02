<script lang="ts">
	import ArrowRight from 'lucide-svelte/icons/arrow-right';
	import type { Itinerary, Leg } from '$lib/openapi';
	import Time from '$lib/Time.svelte';
	import { routeBorderColor, routeColor } from '$lib/modeStyle';
	import { formatDurationSec, formatDistanceMeters } from '$lib/formatDuration';
	import { Button } from '$lib/components/ui/button';
	import Route from '$lib/Route.svelte';
	import { getModeName } from '$lib/getModeName';
	import { t } from '$lib/i18n/translation';

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
		class="font-semibold w-16"
		{isRealtime}
		{timestamp}
		{scheduledTimestamp}
	/>
	<Time
		variant="realtime"
		class="font-semibold w-16"
		{isRealtime}
		{timestamp}
		{scheduledTimestamp}
	/>
	{#if stopId}
		<Button
			class="text-[length:inherit] leading-none justify-normal text-wrap text-left"
			variant="link"
			onclick={() => {
				onClickStop(name, stopId, new Date(timestamp));
			}}
		>
			{name}
		</Button>
	{:else}
		<span>{name}</span>
	{/if}
{/snippet}

{#snippet streetLeg(l: Leg)}
	<div class="py-12 pl-8 flex flex-col gap-y-4 text-muted-foreground">
		<span class="ml-6">
			{formatDurationSec(l.duration)}
			{getModeName(l)}
			{formatDistanceMeters(l.distance)}
		</span>
		{#if l.rental && l.rental.systemName}
			<span class="ml-6">
				{t.sharingProvider}: <a href={l.rental.url} target="_blank">{l.rental.systemName}</a>
			</span>
		{/if}
		{#if l.rental?.returnConstraint == 'ROUNDTRIP_STATION'}
			<span class="ml-6">
				{t.roundtripStationReturnConstraint}
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
		{@const prevTransitLeg = itinerary.legs.slice(0, i).find((l) => l.tripId)}

		{#if l.routeShortName}
			<div class="w-full flex justify-between items-center space-x-1">
				<Route {onClickTrip} {l} />
				{#if pred && (pred.from.track || pred.duration !== 0) && (i != 1 || pred.routeShortName)}
					<div class="border-t h-0 grow shrink"></div>
					<div class="text-sm text-muted-foreground leading-none px-2">
						{#if pred.from.track}
							{t.arrivalOnTrack} {pred.from.track}{pred.duration ? ',' : ''}
						{/if}
						{#if pred.duration}
							{formatDurationSec(pred.duration)} {t.walk}
						{/if}
						{#if pred.distance}
							({Math.round(pred.distance)} m)
						{/if}
					</div>
				{/if}
				<div class="border-t h-0 grow shrink"></div>
				{#if l.from.track}
					<div class="text-nowrap border rounded-xl px-2">
						{t.track}
						{l.from.track}
					</div>
				{/if}
			</div>

			<div class="pt-4 pl-6 border-l-4 left-4 relative" style={routeBorderColor(l)}>
				<div class="grid gap-y-6 grid-cols-[max-content_max-content_auto] items-center">
					{@render stopTimes(
						l.startTime,
						l.scheduledStartTime,
						l.realTime,
						l.from.name,
						l.from.stopId
					)}
				</div>
				<div class="mt-2 flex items-center text-muted-foreground leading-none">
					<ArrowRight class="stroke-muted-foreground h-4 w-4" />
					<span class="ml-1">{l.headsign}</span>
				</div>
				{#if l.intermediateStops?.length === 0}
					<div class="py-8 pl-1 md:pl-4 flex items-center text-muted-foreground">
						{t.tripIntermediateStops(0)}
					</div>
					{#if itinerary.fareTransfers != undefined && l.fareTransferIndex != undefined && l.effectiveFareLegIndex != undefined}
						{@const fareTransfer = itinerary.fareTransfers[l.fareTransferIndex]}
						{@const includedInTransfer =
							fareTransfer.rule == 'AB' ||
							(fareTransfer.rule == 'A_AB' && l.effectiveFareLegIndex !== 0)}
						<div class="list-inside pl-1 md:pl-4 mb-8 text-xs font-bold">
							{#if includedInTransfer || (prevTransitLeg && prevTransitLeg.fareTransferIndex === l.fareTransferIndex && prevTransitLeg.effectiveFareLegIndex === l.effectiveFareLegIndex)}
								Included in ticket.
							{:else}
								{@const productOptions =
									fareTransfer.effectiveFareLegProducts[l.effectiveFareLegIndex]}
								{#if productOptions.length > 1}
									<div class="mb-1">Ticket options:</div>
								{/if}
								<ul
									class:list-disc={productOptions.length > 1}
									class:list-inside={productOptions.length > 1}
								>
									{#each productOptions as product}
										<li>{product.name}</li>
									{/each}
								</ul>
							{/if}
						</div>
					{/if}
				{:else}
					<details class="[&_svg]:open:-rotate-180 my-2">
						<summary class="py-8 pl-1 md:pl-4 flex items-center text-muted-foreground">
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
								{t.tripIntermediateStops(l.intermediateStops?.length ?? 0)}
								({formatDurationSec(l.duration)})
							</span>
						</summary>
						<div class="mb-1 grid gap-y-4 grid-cols-[max-content_max-content_auto] items-center">
							{#each l.intermediateStops! as s}
								{@render stopTimes(s.arrival!, s.scheduledArrival!, l.realTime, s.name!, s.stopId)}
							{/each}
						</div>
					</details>
				{/if}

				{#if !isLast && !(isLastPred && next!.duration === 0)}
					<div class="grid gap-y-6 grid-cols-[max-content_max-content_auto] items-center pb-3">
						{@render stopTimes(
							l.endTime!,
							l.scheduledEndTime!,
							l.realTime!,
							l.to.name,
							l.to.stopId
						)}
					</div>
				{/if}

				{#if isLast || (isLastPred && next!.duration === 0)}
					<!-- fill visual gap -->
					<div class="pb-2"></div>
				{/if}
			</div>
		{:else if !(isLast && l.duration === 0) && ((i == 0 && l.duration !== 0) || !next || !next.routeShortName || l.mode != 'WALK' || (pred && (pred.mode == 'BIKE' || pred.mode == 'RENTAL')))}
			<Route {onClickTrip} {l} />
			<div class="pt-4 pl-6 border-l-4 left-4 relative" style={routeBorderColor(l)}>
				<div class="grid gap-y-6 grid-cols-[max-content_max-content_auto] items-center">
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
					<div class="grid gap-y-6 grid-cols-[max-content_max-content_auto] items-center pb-4">
						{@render stopTimes(l.endTime, l.scheduledEndTime, l.realTime, l.to.name, l.to.stopId)}
					</div>
				{/if}
			</div>
		{/if}
	{/each}
	<div class="relative pl-6 left-4">
		<div
			class="absolute left-[-6px] top-[0px] w-[15px] h-[15px] rounded-full"
			style={routeColor(lastLeg!)}
		></div>
		<div
			class="relative left-[2.5px] bottom-[7px] grid gap-y-6 grid-cols-[max-content_max-content_auto] items-center"
		>
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
