<script lang="ts">
	import ArrowRight from 'lucide-svelte/icons/arrow-right';
	import type { Itinerary } from '$lib/openapi';
	import Time from '../lib/Time.svelte';
	import { routeBorderColor, routeColor } from '$lib/modeStyle';
	import { getModeStyle } from '$lib/modeStyle';
	import { formatDurationSec } from '$lib/formatDuration';

	const { itinerary }: { itinerary: Itinerary } = $props();
	const lastLeg = itinerary.legs.findLast((l) => l.duration !== 0);
</script>

<div class="p-2 text-lg">
	{#each itinerary.legs as l, i}
		{@const isLast = i == itinerary.legs.length - 1}
		{@const isLastPred = i == itinerary.legs.length - 2}
		{@const pred = i == 0 ? undefined : itinerary.legs[i - 1]}
		{@const next = isLast ? undefined : itinerary.legs[i + 1]}
		{@const modeIcon = getModeStyle(l.mode)[0]}
		{#if l.routeShortName}
			<div class="w-full flex justify-between items-center space-x-1">
				<div
					class="flex items-center text-nowrap rounded-full pl-2 pr-3 h-8 py-[1px] font-bold"
					style={routeColor(l)}
				>
					<svg class="relative mr-2 w-6 h-6 fill-white rounded-full">
						<use xlink:href={`#${modeIcon}`}></use>
					</svg>
					{l.routeShortName}
				</div>
				{#if i !== 1}
					<div class="border-t w-full h-0"></div>
					{#if pred && (pred.from.track || pred.duration !== 0)}
						<div class="text-sm text-muted-foreground text-nowrap px-2">
							{#if pred.from.track}
								Ankunft auf Gleis {pred.from.track},
							{/if}
							{#if pred.duration !== 0}
								{formatDurationSec(pred.duration)} Fußweg
							{/if}
						</div>
					{/if}
				{/if}
				<div class="border-t w-full h-0"></div>
				{#if l.from.track}
					<div class="text-nowrap border rounded-xl px-2">
						Gleis {l.from.track}
					</div>
				{/if}
			</div>

			<div class="pt-4 pl-6 border-l-4 left-4 relative" style={routeBorderColor(l)}>
				<div class="flex items-center">
					<Time class="font-semibold mr-2" timestamp={l.startTime} />
					<Time class="font-semibold" timestamp={l.startTime} delay={l.departureDelay} />
					<span class="ml-8">{l.from.name}</span>
				</div>
				<div class="mt-2 flex items-center text-muted-foreground">
					<ArrowRight class="stroke-slate-400 h-4 w-4" />
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
						{#each l.intermediateStops! as s}
							<div class="flex items-center mb-6">
								<Time class="font-semibold mr-2" timestamp={s.arrival!} />
								<Time class="font-semibold" timestamp={s.arrival!} delay={l.arrivalDelay} />
								<span class="ml-8">{s.name}</span>
							</div>
						{/each}
					</details>
				{/if}

				{#if !isLast && !(isLastPred && next!.duration === 0)}
					<div class="flex items-center pb-3">
						<Time class="font-semibold mr-2" timestamp={l.endTime} />
						<Time class="font-semibold" timestamp={l.endTime} delay={l.arrivalDelay} />
						<span class="ml-8">{l.to.name}</span>
					</div>
				{/if}
			</div>
		{:else if !(isLast && l.duration === 0) && ((i == 0 && l.duration !== 0) || !next || !next.routeShortName)}
			<div class="w-full flex justify-between items-center space-x-1">
				<div>
					<svg class="relative left-[7px] w-6 h-6 fill-background rounded-full bg-foreground p-1">
						<use xlink:href={`#${modeIcon}`}></use>
					</svg>
				</div>
				<div class="border-t w-full h-0"></div>
			</div>

			<div class="pt-4 pl-6 border-l-4 left-4 relative" style={routeBorderColor(l)}>
				<div class="flex items-center">
					<Time class="font-semibold mr-2" timestamp={l.startTime} />
					<Time class="font-semibold" timestamp={l.startTime} delay={l.departureDelay} />
					<span class="ml-8">{l.from.name}</span>
				</div>
				<div class="py-12 pl-8 flex items-center text-muted-foreground">
					<span class="ml-6">Fußweg ({formatDurationSec(l.duration)})</span>
				</div>
				{#if !isLast}
					<div class="flex pb-4">
						<Time class="font-semibold mr-2" timestamp={l.endTime} />
						<Time class="font-semibold" timestamp={l.endTime} delay={l.arrivalDelay} />
						<span class="ml-8">{l.to.name}</span>
					</div>
				{/if}
			</div>
		{/if}
	{/each}
	<div class="flex pb-4">
		<div class="relative left-[12px] w-3 h-3 rounded-full" style={routeColor(lastLeg!)}></div>
		<div class="relative left-2 bottom-[7px] pl-6 flex">
			<Time class="font-semibold mr-2" timestamp={lastLeg!.endTime} />
			<Time class="font-semibold" timestamp={lastLeg!.endTime} delay={lastLeg!.arrivalDelay} />
			<span class="ml-8">{lastLeg!.to.name}</span>
		</div>
	</div>
</div>
