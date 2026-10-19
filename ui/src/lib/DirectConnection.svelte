<script lang="ts">
	import { Button, type ButtonProps } from '$lib/components/ui/button';
	import { formatDurationSec } from '$lib/formatDuration';
	import { getModeStyle, routeColor } from './modeStyle';
	import { getProviderName } from './getModeName';
	import { formatTime } from './toDateTime';
	import { directDepartureRange } from './flexShiftRange';
	import type { Itinerary } from '@motis-project/motis-client';
	import { language } from '$lib/i18n/translation';

	const {
		d,
		queriedTime,
		...restProps
	}: {
		d: Itinerary;
		queriedTime?: string;
	} & ButtonProps = $props();

	const modeStyles = [
		...new Map(d.legs.map((l) => [JSON.stringify(getModeStyle(l)), getModeStyle(l)])).values()
	];

	const leg = d.legs.find((leg) => leg.mode !== 'WALK') ?? d.legs[0]!;

	// A FLEX ride is not bound to the query time: show when it can depart.
	const departures = $derived(directDepartureRange(d.legs));

	// Like the itinerary cards: a departure on another day than the query
	// carries the weekday.
	const dayLabel = (t: number, tz: string | undefined) => {
		if (queriedTime === undefined) {
			return '';
		}
		const day = (x: Date) =>
			new Intl.DateTimeFormat('en-CA', {
				timeZone: tz,
				year: 'numeric',
				month: '2-digit',
				day: '2-digit'
			}).format(x);
		const x = new Date(t);
		return day(x) === day(new Date(queriedTime))
			? ''
			: new Intl.DateTimeFormat(language, { weekday: 'short', timeZone: tz }).format(x) + ' ';
	};
</script>

<Button variant="child" {...restProps}>
	<div class="flex flex-col items-stretch">
		<div
			class="relative z-10 flex items-center py-1 px-2 rounded-lg font-bold text-sm h-8 text-nowrap"
			style={routeColor(leg)}
		>
			{#each modeStyles as [icon, _color, _textColor], i (i)}
				<svg class="relative mr-1 w-4 h-4 rounded-full">
					<use xlink:href={`#${icon}`}></use>
				</svg>
			{/each}
			{#if getProviderName(leg)}
				<span class="mr-1.5 max-w-40 truncate">{getProviderName(leg)}</span>
			{/if}
			{formatDurationSec(d.duration)}
		</div>
		{#if departures}
			{@const tz = d.legs[0].from.tz}
			<!-- Tucked under the chip: the grey fills in behind its rounded corners. -->
			<div
				class="-mt-2 rounded-b-lg bg-muted px-2 pt-2.5 pb-0.5 text-center text-xs text-muted-foreground tabular-nums text-nowrap"
			>
				{dayLabel(departures.lo, tz)}{formatTime(new Date(departures.lo), tz)}
				{#if departures.hi > departures.lo}
					– {dayLabel(departures.hi, tz)}{formatTime(new Date(departures.hi), tz)}
				{/if}
			</div>
		{/if}
	</div>
</Button>
