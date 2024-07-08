<script lang="ts">
	import ArrowRight from 'lucide-svelte/icons/arrow-right';
	import type { Itinerary, Mode } from '$lib/openapi';
	import Time from './Time.svelte';
	import { routeColor } from '$lib/routeColor';
	import { getModeStyle } from '$lib/modeStyle';

	class Props {
		itinerary!: Itinerary;
	}

	let { itinerary }: Props = $props();

	const lastLeg = itinerary.legs.findLast((l) => l.duration !== 0);
	console.log(lastLeg);

	const formatDurationSec = (t: number) => {
		let hours = Math.floor(t / 3600);
		let minutes = (t - hours * 3600) / 60;
		let str = [hours !== 0 ? hours + 'h' : '', minutes !== 0 ? minutes + 'min' : ''].join(' ');
		return str;
	};
</script>

{#snippet train()}
	<svg viewBox="0 0 24 24">
		<path
			d="M4,15.5C4,17.43 5.57,19 7.5,19L6,20.5v0.5h12v-0.5L16.5,19c1.93,0 3.5,-1.57 3.5,-3.5L20,5c0,-3.5 -3.58,-4 -8,-4s-8,0.5 -8,4v10.5zM12,17c-1.1,0 -2,-0.9 -2,-2s0.9,-2 2,-2 2,0.9 2,2 -0.9,2 -2,2zM18,10L6,10L6,5h12v5z"
		/>
	</svg>
{/snippet}

<div class="p-2 text-lg">
	{#each itinerary.legs as l, i}
		{@const isLast = i == itinerary.legs.length - 1}
		{@const isLastPred = i == itinerary.legs.length - 2}
		{@const pred = i == 0 ? undefined : itinerary.legs[i - 1]}
		{@const next = isLast ? undefined : itinerary.legs[i + 1]}
		{@const [modeIcon, modeColor] = getModeStyle(l.mode)}
		{#if l.routeShortName}
			<div class="w-full flex justify-between items-center space-x-1">
				<div
					class="flex items-center text-nowrap rounded-full px-2 h-8 py-[1px] font-bold"
					style={routeColor(l, modeColor)}
				>
					<svg class="relative mr-2 w-6 h-6 fill-white rounded-full">
						<use xlink:href={`#${modeIcon}`}></use>
					</svg>
					{l.routeShortName}
				</div>
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
				<div class="border-t w-full h-0"></div>
				{#if l.from.track}
					<div class="text-nowrap border rounded-xl px-2">
						Gleis {l.from.track}
					</div>
				{/if}
			</div>

			<div class="pt-4 pl-6 border-l-4 left-4 relative" style={`border-color: #${modeColor}`}>
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
								<Time class="font-semibold mr-2" timestamp={s.arrival} />
								<Time class="font-semibold" timestamp={s.arrival} delay={l.arrivalDelay} />
								<span class="ml-8">{s.name}</span>
							</div>
						{/each}
					</details>
				{/if}

				{#if !isLast && !(isLastPred && next!.duration === 0)}
					<div class="flex items-center pb-4">
						<Time class="font-semibold mr-2" timestamp={l.endTime} />
						<Time class="font-semibold" timestamp={l.endTime} delay={l.arrivalDelay} />
						<span class="ml-8">{l.to.name}</span>
					</div>
				{/if}
			</div>
		{:else if !(isLast && l.duration === 0) && ((i == 0 && l.duration !== 0) || !next || !next.routeShortName)}
			<div class="w-full flex justify-between items-center space-x-1">
				<div>
					<svg class="relative left-[7px] w-6 h-6 fill-white rounded-full bg-black p-1">
						<use xlink:href={`#${modeIcon}`}></use>
					</svg>
				</div>
				<div class="border-t w-full h-0"></div>
			</div>

			<div class="pt-4 pl-6 border-l-4 left-4 relative" style={`border-color: #${modeColor}`}>
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
		<div
			class="relative left-[12px] w-3 h-3 rounded-full"
			style={`background-color: #${getModeStyle(lastLeg!.mode)[1]}`}
		></div>
		<div class="relative left-1 bottom-1 pl-6 flex">
			<Time class="font-semibold mr-2" timestamp={lastLeg.endTime} />
			<Time class="font-semibold" timestamp={lastLeg.endTime} delay={lastLeg.arrivalDelay} />
			<span class="ml-8">{lastLeg.to.name}</span>
		</div>
	</div>
</div>
