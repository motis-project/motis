<script lang="ts">
	import ArrowRight from 'lucide-svelte/icons/arrow-right';
	import ArrowUp from 'lucide-svelte/icons/arrow-up';
	import ArrowDown from 'lucide-svelte/icons/arrow-down';
	import DollarSign from 'lucide-svelte/icons/dollar-sign';
	import CircleX from 'lucide-svelte/icons/circle-x';
	import type { FareProduct, Itinerary, Leg, Place, StepInstruction } from '$lib/api/openapi';
	import Time from '$lib/Time.svelte';
	import { routeBorderColor, routeColor } from '$lib/modeStyle';
	import { formatDurationSec, formatDistanceMeters } from '$lib/formatDuration';
	import { Button } from '$lib/components/ui/button';
	import Route from '$lib/Route.svelte';
	import { getModeName } from '$lib/getModeName';
	import { language, t } from '$lib/i18n/translation';
	import { onClickStop, onClickTrip } from '$lib/utils';
	import { formatDate, formatTime } from './toDateTime';

	const {
		itinerary
	}: {
		itinerary: Itinerary;
	} = $props();

	const isRelevantLeg = (l: Leg) => l.duration !== 0 || l.displayName;
	const lastLeg = $derived(itinerary.legs.findLast(isRelevantLeg));
</script>

{#snippet stopTimes(
	timestamp: string,
	scheduledTimestamp: string,
	isRealtime: boolean,
	p: Place,
	isStartOrEnd: number
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
		{#if p.stopId}
			<Button
				class="text-[length:inherit] leading-none justify-normal text-wrap text-left"
				variant="link"
				onclick={() => onClickStop(p.name, p.stopId!, new Date(timestamp))}
			>
				{p.name}
			</Button>
			{@const pickupNotAllowedOrEnd = p.pickupType == 'NOT_ALLOWED' && isStartOrEnd != -1}
			{@const dropoffNotAllowedOrStart = p.dropoffType == 'NOT_ALLOWED' && isStartOrEnd != 1}
			{#if (p as Place & { switchTo?: Leg }).switchTo}
				{@const switchTo = (p as Place & { switchTo: Leg }).switchTo}
				<div class="ml-4 flex items-center text-sm">
					{t.continuesAs}
					{switchTo.displayName!}
					<ArrowRight class="mx-1 size-4" />
					{switchTo.headsign}
				</div>
			{/if}
			{#if pickupNotAllowedOrEnd || dropoffNotAllowedOrStart}
				<div class="ml-4 flex items-center text-destructive text-sm">
					<CircleX class="stroke-destructive size-4" />
					<span class="ml-1 leading-none">
						{pickupNotAllowedOrEnd && dropoffNotAllowedOrStart
							? t.inOutDisallowed
							: pickupNotAllowedOrEnd
								? t.inDisallowed
								: t.outDisallowed}
					</span>
				</div>
			{/if}
			{#if isStartOrEnd && p.alerts}
				{#each p.alerts as alert, i (i)}
					<div class="ml-4 text-destructive text-sm">
						{alert.headerText}
					</div>
				{/each}
			{/if}
		{:else}
			<span class="px-4 py-2">{p.name || p.flex}</span>
		{/if}
	</span>
{/snippet}

{#snippet streetLeg(l: Leg)}
	{@const stepsWithElevation = l.steps?.filter(
		(s: StepInstruction) => s.elevationUp || s.elevationDown
	)}
	{@const stepsWithToll = l.steps?.filter((s: StepInstruction) => s.toll)}
	{@const stepsWithAccessRestriction = l.steps?.filter((s: StepInstruction) => s.accessRestriction)}

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
		{#if stepsWithElevation && stepsWithElevation.length > 0}
			<div class="ml-6 flex items-center gap-2 text-xs">
				{t.incline}
				<div class="flex items-center">
					<ArrowUp class="size-4" />
					{stepsWithElevation.reduce((acc: number, s: StepInstruction) => acc + s.elevationUp!, 0)} m
				</div>
				<div class="flex items-center">
					<ArrowDown class="size-4" />
					{stepsWithElevation.reduce(
						(acc: number, s: StepInstruction) => acc + s.elevationDown!,
						0
					)} m
				</div>
			</div>
		{/if}
		{#if stepsWithToll && stepsWithToll.length > 0}
			<div class="ml-6 flex items-center gap-2 text-sm text-orange-500">
				<DollarSign class="size-4" />
				{t.toll}
			</div>
		{/if}
		{#if stepsWithAccessRestriction && stepsWithAccessRestriction.length > 0}
			<div class="ml-6 flex items-center gap-2 text-sm text-orange-500">
				<CircleX class="size-4" />
				{t.accessRestriction}
				({stepsWithAccessRestriction
					.map((s) => s.accessRestriction)
					.filter((value, index, array) => array.indexOf(value) === index)
					.join(', ')})
			</div>
		{/if}
	</div>
{/snippet}

{#snippet productInfo(product: FareProduct)}
	{product.name}
	{new Intl.NumberFormat(language, { style: 'currency', currency: product.currency }).format(
		product.amount
	)}
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
					{#each productOptions as products, i (i)}
						{#each products as product, j (j)}
							<li>
								{@render productInfo(product)}
							</li>
						{/each}
					{/each}
				</ul>
			{/if}
		</div>
	{/if}
{/snippet}

<div class="text-lg">
	{#each itinerary.legs as l, i (i)}
		{@const isLast = i == itinerary.legs.length - 1}
		{@const isLastPred = i == itinerary.legs.length - 2}
		{@const pred = i == 0 ? undefined : itinerary.legs[i - 1]}
		{@const next = isLast ? undefined : itinerary.legs[i + 1]}
		{@const prevTransitLeg = itinerary.legs.slice(0, i).find((l) => l.tripId)}

		{#if l.displayName}
			<div class="w-full flex justify-between items-center space-x-1">
				<Route {onClickTrip} {l} />
				{#if pred && (pred.from.track || isRelevantLeg(pred)) && (i != 1 || pred.displayName)}
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
						{#if prevTransitLeg?.fareTransferIndex != undefined && itinerary.fareTransfers && itinerary.fareTransfers[prevTransitLeg.fareTransferIndex].transferProducts}
							{@const transferProducts =
								itinerary.fareTransfers[prevTransitLeg.fareTransferIndex].transferProducts!}
							{#if prevTransitLeg.effectiveFareLegIndex === 0 && l.effectiveFareLegIndex === 1}
								<br />
								<span class="text-xs font-bold text-foreground">
									Ticket: {pred.effectiveFareLegIndex}
									{#each transferProducts as transferProduct (transferProduct.name)}
										{@render productInfo(transferProduct)}
									{/each}
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
					{@render stopTimes(l.startTime, l.scheduledStartTime, l.realTime, l.from, 1)}
				</div>
				<div class="mt-2 mb-2 flex items-center text-muted-foreground leading-none">
					<ArrowRight class="stroke-muted-foreground size-4" />
					<span class="ml-1">{l.headsign}</span>
				</div>
				{#if l.loopedCalendarSince}
					<div class="mt-2 flex items-center text-destructive leading-none">
						{t.dataExpiredSince}
						{formatDate(new Date(l.loopedCalendarSince))}
					</div>
				{/if}
				{#if l.cancelled}
					<div class="mt-2 flex items-center text-destructive leading-none">
						<CircleX class="stroke-destructive size-4" />
						<span class="ml-1 font-bold">{t.tripCancelled}</span>
					</div>
				{/if}
				{#if !l.scheduled}
					<div class="mt-2 flex items-center text-green-600 leading-none">
						<span class="ml-1">{t.unscheduledTrip}</span>
					</div>
				{/if}
				{#if l.alerts}
					<ul class="mt-2">
						{#each l.alerts as alert, i (i)}
							<li class="text-destructive text-sm font-bold">
								{alert.headerText}
							</li>
						{/each}
					</ul>
				{/if}
				{#if l.intermediateStops?.length === 0}
					<div class="pt-16 pb-8 pl-1 md:pl-4 flex items-center text-muted-foreground">
						{t.tripIntermediateStops(0)}
					</div>
					{@render ticketInfo(prevTransitLeg, l)}
				{:else}
					{@render ticketInfo(prevTransitLeg, l)}
					<details class="[&_.collapsible]:open:-rotate-180 my-2">
						<summary class="py-8 pl-1 md:pl-4 flex items-center text-muted-foreground">
							<svg
								class="collapsible rotate-0 transform transition-all duration-300"
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
							{#each l.intermediateStops! as s, i (i)}
								{@render stopTimes(s.arrival!, s.scheduledArrival!, l.realTime, s, 0)}
							{/each}
						</div>
					</details>
				{/if}

				{#if !isLast && !(isLastPred && !isRelevantLeg(next!))}
					<div class="grid gap-y-6 grid-cols-[max-content_max-content_auto] items-center pb-3">
						{@render stopTimes(l.endTime!, l.scheduledEndTime!, l.realTime!, l.to, -1)}
					</div>
				{/if}

				{#if isLast || (isLastPred && !isRelevantLeg(next!))}
					<!-- fill visual gap -->
					<div class="pb-2"></div>
				{/if}
			</div>
		{:else if !(isLast && !isRelevantLeg(l)) && ((i == 0 && isRelevantLeg(l)) || !next || !next.displayName || l.mode != 'WALK' || (pred && (pred.mode == 'BIKE' || (l.mode == 'WALK' && pred.mode == 'CAR') || pred.mode == 'RENTAL')))}
			<Route {onClickTrip} {l} />
			<div class="pt-4 pl-6 border-l-4 left-4 relative" style={routeBorderColor(l)}>
				<div class="grid gap-y-6 grid-cols-[max-content_max-content_auto] items-center">
					{@render stopTimes(l.startTime, l.scheduledStartTime, l.realTime, l.from, 1)}
				</div>
				{#if l.mode == 'FLEX'}
					<div class="mt-2 flex items-center leading-none">
						<span class="ml-1 text-sm">
							{formatTime(new Date(l.from.flexStartPickupDropOffWindow!))} -
							{formatTime(new Date(l.from.flexEndPickupDropOffWindow!))}
						</span>
					</div>
				{/if}
				{@render streetLeg(l)}
				{#if !isLast}
					<div class="grid gap-y-6 grid-cols-[max-content_max-content_auto] items-center pb-4">
						{@render stopTimes(l.endTime, l.scheduledEndTime, l.realTime, l.to, -1)}
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
				lastLeg!.to,
				-1
			)}
		</div>
	</div>
</div>
