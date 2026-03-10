<script lang="ts">
	import { ArrowRight, ArrowUp, ArrowDown, DollarSign, CircleX } from '@lucide/svelte';
	import type {
		FareProduct,
		Itinerary,
		Leg,
		Mode,
		Place,
		StepInstruction
	} from '@motis-project/motis-client';
	import Time from '$lib/Time.svelte';
	import { routeBorderColor, routeColor } from '$lib/modeStyle';
	import { formatDurationSec, formatDistanceMeters } from '$lib/formatDuration';
	import { Button } from '$lib/components/ui/button';
	import Route from '$lib/Route.svelte';
	import Alerts from '$lib/Alerts.svelte';
	import { getModeName } from '$lib/getModeName';
	import { language, t } from '$lib/i18n/translation';
	import { onClickStop, onClickTrip } from '$lib/utils';
	import { formatDate, formatTime } from './toDateTime';
	import { getModeLabel } from './map/getModeLabel';
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
	mode: Mode,
	isStartOrEnd: number,
	hidePlatform?: boolean
)}
	{@const arriveBy = isStartOrEnd == 0 || isStartOrEnd == 1}
	{@const textColor = isStartOrEnd == 0 ? '' : 'font-semibold'}
	<div class="flex items-baseline justify-between w-full {textColor}">
		<div class="flex justify-between">
			<Time
				variant="schedule"
				class="w-14 md:w-16"
				queriedTime={timestamp}
				timeZone={p.tz}
				{isRealtime}
				{timestamp}
				{scheduledTimestamp}
				{arriveBy}
			/>
			<Time
				variant="realtime"
				class="w-14 md:w-16"
				timeZone={p.tz}
				{isRealtime}
				{timestamp}
				{scheduledTimestamp}
				{arriveBy}
			/>
		</div>
		<div class="w-full">
			{#if p.stopId}
				{@const pickupNotAllowedOrEnd = p.pickupType == 'NOT_ALLOWED' && isStartOrEnd != 1}
				{@const dropoffNotAllowedOrStart = p.dropoffType == 'NOT_ALLOWED' && isStartOrEnd != -1}
				<div class="flex items-center justify-between">
					<div class="flex flex-row items-center justify-center">
						<Button
							class="text-[length:inherit] leading-none justify-normal text-wrap p-0 text-left {textColor}"
							variant="link"
							onclick={() => onClickStop(p.name, p.stopId!, new Date(timestamp))}
						>
							{p.name}
						</Button>

						{#if isStartOrEnd != 0}
							<Alerts alerts={p.alerts} tz={p.tz} variant="icon" />
						{/if}
					</div>
					{#if p.track && !hidePlatform}
						<span class="text-nowrap px-2 border rounded-xl ml-1 mr-4">
							{getModeLabel(mode) == 'Track' ? t.trackAbr : t.platformAbr}
							{p.track}
						</span>
					{/if}
				</div>
				<div>
					{#if (p as Place & { switchTo?: Leg }).switchTo}
						{@const switchTo = (p as Place & { switchTo: Leg }).switchTo}
						<div class="flex items-center text-sm mt-1">
							{t.continuesAs}
							{switchTo.displayName!}
							<ArrowRight class="mx-1 size-4" />
							{switchTo.headsign}
						</div>
					{/if}
					{#if pickupNotAllowedOrEnd || dropoffNotAllowedOrStart}
						<div class="flex items-center text-destructive text-sm mt-1">
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
				</div>
			{:else}
				<span>{p.name || p.flex}</span>
			{/if}
		</div>
	</div>
{/snippet}

{#snippet streetLeg(l: Leg)}
	{@const stepsWithElevation = l.steps?.filter(
		(s: StepInstruction) => s.elevationUp || s.elevationDown
	)}
	{@const stepsWithToll = l.steps?.filter((s: StepInstruction) => s.toll)}
	{@const stepsWithAccessRestriction = l.steps?.filter((s: StepInstruction) => s.accessRestriction)}

	<div class="py-12 flex flex-col gap-y-4 text-muted-foreground">
		<span class="ml-6">
			{formatDurationSec(l.duration)}
			{getModeName(l)}
			{formatDistanceMeters(l.distance)}
		</span>
		{#if l.rental && l.rental.systemName}
			<span class="ml-6">
				{t.sharingProvider}:
				<a href={l.rental.url} target="_blank" class="hover:underline">{l.rental.systemName}</a>
			</span>
		{/if}
		{#if l.rental?.returnConstraint == 'ROUNDTRIP_STATION'}
			<span class="ml-6">
				{t.roundtripStationReturnConstraint}
			</span>
		{/if}
		{#if l.rental?.rentalUriWeb}
			<span class="ml-6">
				<Button class="font-bold" variant="outline" href={l.rental.rentalUriWeb} target="_blank">
					{t.rent}
				</Button>
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
		{#if includedInTransfer || fareTransfer.effectiveFareLegProducts[l.effectiveFareLegIndex].length > 0}
			<div class="pl-1 md:pl-4 my-8 text-xs font-bold">
				{#if includedInTransfer || (prevTransitLeg && prevTransitLeg.fareTransferIndex === l.fareTransferIndex && prevTransitLeg.effectiveFareLegIndex === l.effectiveFareLegIndex)}
					{t.includedInTicket}
				{:else}
					{@const productOptions = fareTransfer.effectiveFareLegProducts[l.effectiveFareLegIndex]}
					{#if productOptions.length > 1}
						<div class="mb-1">{t.ticketOptions}:</div>
					{/if}
					<ul
						class:list-disc={productOptions.length > 1}
						class:list-outside={productOptions.length > 1}
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
	{/if}
{/snippet}

<div class="text-lg max-w-full">
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
								<span class="text-xs font-bold text-foreground text-left">
									{#if transferProducts.length > 1}
										<div class="mb-1">{t.ticketOptions}:</div>
									{/if}
									<ul
										class:list-disc={transferProducts.length > 1}
										class:list-outside={transferProducts.length > 1}
									>
										{#each transferProducts as product, j (j)}
											<li>
												{@render productInfo(product)}
											</li>
										{/each}
									</ul>
								</span>
							{/if}
						{/if}
					</div>
				{/if}
				<div class="border-t h-0 grow shrink"></div>
			</div>

			<div class="pt-4 pb-2 pl-4 sm:pl-6 border-l-4 left-4 relative" style={routeBorderColor(l)}>
				{@render stopTimes(l.startTime, l.scheduledStartTime, l.realTime, l.from, l.mode, -1)}
				<div class="flex items-center">
					<ArrowRight class="stroke-muted-foreground size-4" />
					<span class="ml-1">
						{#if l.tripTo}
							<Button
								class="text-[length:inherit] text-muted-foreground leading-none justify-normal text-wrap text-left "
								variant="link"
								onclick={() =>
									onClickStop(
										l.tripTo!.name,
										l.tripTo!.stopId!,
										new Date(l.tripTo!.arrival!),
										true
									)}
							>
								{l.headsign}
								{#if !l.headsign || !l.tripTo.name.startsWith(l.headsign)}
									<br />({l.tripTo.name})
								{/if}
							</Button>
						{:else}
							{l.headsign}
						{/if}
					</span>
				</div>

				<Alerts alerts={l.alerts} tz={l.from.tz || l.to.tz} variant="full" />

				{#if l.routeUrl}
					<div class="mt-2 mr-4">
						<Button
							variant="secondary"
							href={l.routeUrl}
							target="_blank"
							class="overflow-hidden text-ellipsis whitespace-nowrap w-full px-4 inline-block underline"
						>
							{l.routeUrl}
						</Button>
					</div>
				{/if}

				{#if l.loopedCalendarSince}
					<div class="mt-2 flex items-center text-destructive leading-none">
						{t.dataExpiredSince}
						{formatDate(new Date(l.loopedCalendarSince), l.from.tz)}
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
				{#if l.intermediateStops?.length === 0}
					<div class="py-10 pl-4 md:pl-4 flex items-center text-muted-foreground">
						{t.tripIntermediateStops(0)} ({formatDurationSec(l.duration)})
					</div>
					{@render ticketInfo(prevTransitLeg, l)}
				{:else}
					{@render ticketInfo(prevTransitLeg, l)}
					<details class="[&_.collapsible]:open:-rotate-180 my-2">
						<summary class="py-10 pl-4 md:pl-4 flex items-center text-muted-foreground">
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
						<div class="grid gap-2 items-start content-start pb-2">
							{#each l.intermediateStops! as s, i (i)}
								{@render stopTimes(s.arrival!, s.scheduledArrival!, l.realTime, s, l.mode, 0)}
							{/each}
						</div>
					</details>
				{/if}

				{#if !isLast && !(isLastPred && !isRelevantLeg(next!))}
					{@render stopTimes(l.endTime!, l.scheduledEndTime!, l.realTime!, l.to, l.mode, 1)}
				{/if}

				{#if isLast || (isLastPred && !isRelevantLeg(next!))}
					<!-- fill visual gap -->
					<div class="pb-2"></div>
				{/if}
			</div>
		{:else if !(isLast && !isRelevantLeg(l)) && ((i == 0 && isRelevantLeg(l)) || !next || !next.displayName || l.mode != 'WALK' || (pred && (pred.mode == 'BIKE' || (l.mode == 'WALK' && pred.mode == 'CAR') || pred.mode == 'RENTAL')))}
			<Route {onClickTrip} {l} />
			<div class="pt-2 pb-2 pl-4 sm:pl-6 border-l-4 left-4 relative" style={routeBorderColor(l)}>
				{@render stopTimes(l.startTime, l.scheduledStartTime, l.realTime, l.from, l.mode, -1, true)}
				{#if l.mode == 'FLEX'}
					<div class="mt-2 flex items-center leading-none">
						<span class="ml-1 text-sm">
							{formatTime(new Date(l.from.flexStartPickupDropOffWindow!), l.from.tz)} -
							{formatTime(new Date(l.from.flexEndPickupDropOffWindow!), l.from.tz)}
						</span>
					</div>
				{/if}
				{@render streetLeg(l)}
				{#if !isLast}
					{@render stopTimes(l.endTime, l.scheduledEndTime, l.realTime, l.to, l.mode, 1, true)}
				{/if}
			</div>
		{/if}
	{/each}
	<div class="relative pl-4 md:pl-6 left-5">
		<div
			class="absolute left-[-9px] w-[15px] h-[15px] rounded-full"
			style={routeColor(lastLeg!)}
		></div>
		<div class="relative top-[-6px] mb-[-6px]">
			{@render stopTimes(
				lastLeg!.endTime,
				lastLeg!.scheduledEndTime,
				lastLeg!.realTime,
				lastLeg!.to,
				lastLeg!.mode,
				1
			)}
		</div>
	</div>
</div>
