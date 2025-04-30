<script lang="ts">
	import ArrowRight from 'lucide-svelte/icons/arrow-right';
	import CircleX from 'lucide-svelte/icons/circle-x';
	import { type FareProduct, type Itinerary, type Leg, type PickupDropoffType } from '$lib/openapi';
	import Time from '$lib/Time.svelte';
	import { routeBorderColor, routeColor } from '$lib/modeStyle';
	import { formatDurationSec, formatDistanceMeters } from '$lib/formatDuration';
	import { Button } from '$lib/components/ui/button';
	import Route from '$lib/Route.svelte';
	import { getModeName } from '$lib/getModeName';
	import { t } from '$lib/i18n/translation';
	import { onClickStop, onClickTrip } from '$lib/utils';

	const {
		itinerary
	}: {
		itinerary: Itinerary;
	} = $props();

	const isRelevantLeg = (l: Leg) => l.duration !== 0 || l.routeShortName;
	const lastLeg = $derived(itinerary.legs.findLast(isRelevantLeg));
</script>

{#snippet stopTimes(
	timestamp: string,
	scheduledTimestamp: string,
	isRealtime: boolean,
	name: string,
	stopId?: string,
	pickupType?: PickupDropoffType,
	dropoffType?: PickupDropoffType
)}
	<Time
		variant="schedule"
		class="font-semibold w-16"
		queriedTime={timestamp}
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
	<span>
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
			{#if pickupType == 'NOT_ALLOWED' || dropoffType == 'NOT_ALLOWED'}
				<div class="ml-4 flex items-center text-destructive text-sm">
					<CircleX class="stroke-destructive h-4 w-4" />
					<span class="ml-1 leading-none">
						{pickupType == 'NOT_ALLOWED' && dropoffType == 'NOT_ALLOWED'
							? t.inOutDisallowed
							: pickupType == 'NOT_ALLOWED'
								? t.inDisallowed
								: t.outDisallowed}
					</span>
				</div>
			{/if}
		{:else}
			<span>{name}</span>
		{/if}
	</span>
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

{#snippet productInfo(product: FareProduct)}
	{product.name}
	({product.amount}
	{product.currency})
	{#if product.riderCategory}
		for
		{#if product.riderCategory.eligibilityUrl}
			<a
				class:italic={product.riderCategory.isDefaultFareCategory}
				class="underline"
				href={product.riderCategory.eligibilityUrl}
			>
				{product.riderCategory.riderCategoryName}
			</a>
		{:else}
			<span class:italic={product.riderCategory.isDefaultFareCategory}>
				{product.riderCategory.riderCategoryName}
			</span>
		{/if}
	{/if}
	{#if product.media}
		as
		{#if product.media.fareMediaName}
			{product.media.fareMediaName}
		{:else}
			{product.media.fareMediaType}
		{/if}
	{/if}
{/snippet}

{#snippet ticketInfo(prevTransitLeg: Leg | undefined, l: Leg)}
	{#if itinerary.fareTransfers != undefined && l.fareTransferIndex != undefined && l.effectiveFareLegIndex != undefined}
		{@const fareTransfer = itinerary.fareTransfers[l.fareTransferIndex]}
		{@const includedInTransfer =
			fareTransfer.rule == 'AB' || (fareTransfer.rule == 'A_AB' && l.effectiveFareLegIndex !== 0)}
		<div class="list-inside pl-1 md:pl-4 my-8 text-xs font-bold">
			{#if includedInTransfer || (prevTransitLeg && prevTransitLeg.fareTransferIndex === l.fareTransferIndex && prevTransitLeg.effectiveFareLegIndex === l.effectiveFareLegIndex)}
				{t.includedInTicket}
			{:else}
				{@const productOptions = fareTransfer.effectiveFareLegProducts[l.effectiveFareLegIndex]}
				{#if productOptions.length > 1}
					<div class="mb-1">{t.ticketOptions}:</div>
				{/if}
				<ul
					class:list-disc={productOptions.length > 1}
					class:list-inside={productOptions.length > 1}
				>
					{#each productOptions as product}
						<li>
							{#if productOptions.length == 1}
								{t.ticket}
							{/if}
							{@render productInfo(product)}
						</li>
					{/each}
				</ul>
			{/if}
		</div>
	{/if}
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
				{#if pred && (pred.from.track || isRelevantLeg(pred)) && (i != 1 || pred.routeShortName)}
					<div class="border-t h-0 grow shrink"></div>
					<div class="text-sm text-muted-foreground leading-none px-2 text-center">
						{#if pred.from.track}
							{t.arrivalOnTrack} {pred.from.track}{pred.duration ? ',' : ''}
						{/if}
						{#if pred.duration}
							{formatDurationSec(pred.duration)} {t.walk}
						{/if}
						{#if pred.distance}
							({Math.round(pred.distance)} m)
						{/if}
						{#if prevTransitLeg?.fareTransferIndex != undefined && itinerary.fareTransfers && itinerary.fareTransfers[prevTransitLeg.fareTransferIndex].transferProduct}
							{@const transferProduct =
								itinerary.fareTransfers[prevTransitLeg.fareTransferIndex].transferProduct!}
							{#if prevTransitLeg.effectiveFareLegIndex === 0 && l.effectiveFareLegIndex === 1}
								<br />
								<span class="text-xs font-bold text-foreground">
									Ticket: {pred.effectiveFareLegIndex}
									{@render productInfo(transferProduct)}
								</span>
							{/if}
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
						l.from.stopId,
						l.from.pickupType,
						'NORMAL'
					)}
				</div>
				<div class="mt-2 flex items-center text-muted-foreground leading-none">
					<ArrowRight class="stroke-muted-foreground h-4 w-4" />
					<span class="ml-1">{l.headsign}</span>
				</div>
				{#if l.cancelled}
					<div class="mt-2 flex items-center text-destructive leading-none">
						<CircleX class="stroke-destructive h-4 w-4" />
						<span class="ml-1 font-bold">{t.tripCancelled}</span>
					</div>
				{/if}
				{#if !l.scheduled}
					<div class="mt-2 flex items-center text-green-600 leading-none">
						<span class="ml-1">{t.unscheduledTrip}</span>
					</div>
				{/if}
				{#if l.alerts}
					{#each l.alerts as alert}
						<div class="text-destructive text-sm font-bold">
							{alert.headerText}
						</div>
					{/each}
				{/if}
				{#if l.intermediateStops?.length === 0}
					<div class="py-8 pl-1 md:pl-4 flex items-center text-muted-foreground">
						{t.tripIntermediateStops(0)}
					</div>
					{@render ticketInfo(prevTransitLeg, l)}
				{:else}
					{@render ticketInfo(prevTransitLeg, l)}
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
								{@render stopTimes(
									s.arrival!,
									s.scheduledArrival!,
									l.realTime,
									s.name!,
									s.stopId,
									s.pickupType,
									s.dropoffType
								)}
							{/each}
						</div>
					</details>
				{/if}

				{#if !isLast && !(isLastPred && !isRelevantLeg(next!))}
					<div class="grid gap-y-6 grid-cols-[max-content_max-content_auto] items-center pb-3">
						{@render stopTimes(
							l.endTime!,
							l.scheduledEndTime!,
							l.realTime!,
							l.to.name,
							l.to.stopId,
							'NORMAL',
							l.to.dropoffType
						)}
					</div>
				{/if}

				{#if isLast || (isLastPred && !isRelevantLeg(next!))}
					<!-- fill visual gap -->
					<div class="pb-2"></div>
				{/if}
			</div>
		{:else if !(isLast && !isRelevantLeg(l)) && ((i == 0 && isRelevantLeg(l)) || !next || !next.routeShortName || l.mode != 'WALK' || (pred && (pred.mode == 'BIKE' || pred.mode == 'RENTAL')))}
			<Route {onClickTrip} {l} />
			<div class="pt-4 pl-6 border-l-4 left-4 relative" style={routeBorderColor(l)}>
				<div class="grid gap-y-6 grid-cols-[max-content_max-content_auto] items-center">
					{@render stopTimes(
						l.startTime,
						l.scheduledStartTime,
						l.realTime,
						l.from.name,
						l.from.stopId,
						l.from.pickupType,
						'NORMAL'
					)}
				</div>
				{@render streetLeg(l)}
				{#if !isLast}
					<div class="grid gap-y-6 grid-cols-[max-content_max-content_auto] items-center pb-4">
						{@render stopTimes(
							l.endTime,
							l.scheduledEndTime,
							l.realTime,
							l.to.name,
							l.to.stopId,
							'NORMAL',
							l.to.dropoffType
						)}
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
				lastLeg!.to.stopId,
				'NORMAL',
				lastLeg!.to.dropoffType
			)}
		</div>
	</div>
</div>
